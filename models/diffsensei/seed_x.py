import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import LogitsProcessorList

# from src.models.mllm.generation import AutoImageTokenGenerationProcessor
from generation import AutoImageTokenGenerationProcessor

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'


def cosine_loss(rec, target):
    target = target / target.norm(dim=-1, keepdim=True)
    rec = rec / rec.norm(dim=-1, keepdim=True)
    rec_loss = (1 - (target * rec).sum(-1)).mean()
    return rec_loss


class ContinuousLVLM(nn.Module):
    def __init__(self, llm, input_resampler, output_resampler, lm_loss_scale=1.0, rec_loss_scale=1.0, mse=True) -> None:
        super().__init__()
        self.llm = llm
        self.input_resampler = input_resampler
        self.output_resampler = output_resampler
        self.lm_loss_scale = lm_loss_scale
        self.rec_loss_scale = rec_loss_scale
        
        self.mse = mse
        if self.mse:
            self.mse_loss = torch.nn.MSELoss() 

    def forward(self, input_ids, attention_mask, labels, image_embeds, embeds_gen_mask, embeds_cmp_mask, ids_gen_mask, ids_cmp_mask):
        input_embeds = self.llm.get_input_embeddings()(input_ids)  # bz x seq_len x dim

        bz, sq, dim = input_embeds.shape

        if image_embeds is not None:
            image_embeds_cmp = image_embeds[embeds_cmp_mask]
            image_embeds_lm = self.input_resampler(image_embeds_cmp)
            input_embeds[ids_cmp_mask] = image_embeds_lm.reshape(-1, dim)
        else:
            image_embeds = torch.randn(bz, self.output_resampler.num_queries, self.output_resampler.embed_dim).to(input_embeds.device, dtype=input_embeds.dtype)
            image_embeds_lm = self.input_resampler(image_embeds)
            min_bz = min(input_embeds.shape[0], image_embeds_lm.shape[0])
            input_embeds[:min_bz, :self.input_resampler.num_queries, :] = input_embeds[:min_bz, :self.input_resampler.num_queries, :] + 0.0 * image_embeds_lm[:min_bz, :, :]

        has_image_output = image_embeds is not None and embeds_gen_mask.sum().item() > 0
            
        output_lm = self.llm(attention_mask=attention_mask,
                             inputs_embeds=input_embeds,
                             labels=labels,
                             output_hidden_states=True,
                             return_dict=True)
        lm_loss = output_lm['loss']

        last_hidden_state = output_lm.hidden_states[-1]  # 4 x 160 x 4096

        if has_image_output:
            target_embeds = image_embeds[embeds_gen_mask]  # num_imgs_gen_target x nq_in x dim_in, 2 x 256 x 4096

            num_imgs_for_rec = target_embeds.shape[0]
            output_image_embeds = last_hidden_state[ids_gen_mask].view(num_imgs_for_rec, -1, dim)  # 128 x 4096 -> 2 x 64 x 4096

            recon_image_embeds = self.output_resampler(output_image_embeds)  # 2 x 256 x 4096

            if self.mse:
                # rec_loss = self.mse_loss(recon_image_embeds, target_embeds.detach())
                rec_loss = F.mse_loss(recon_image_embeds, target_embeds.detach()) # for zero3 compatibility
            else:
                rec_loss = cosine_loss(recon_image_embeds, target_embeds.detach())
        else:
            output_image_embeds = torch.randn(bz, self.input_resampler.num_queries, self.input_resampler.embed_dim).to(input_embeds.device, dtype=input_embeds.dtype)
            recon_image_embeds = self.output_resampler(output_image_embeds)
            target_embeds = torch.randn(bz, self.output_resampler.num_queries, self.output_resampler.embed_dim).to(input_embeds.device, dtype=input_embeds.dtype)
            rec_loss = F.mse_loss(recon_image_embeds, target_embeds.detach()) * 0.0

        total_loss = self.lm_loss_scale * lm_loss + self.rec_loss_scale * rec_loss

        return {
            'total_loss': total_loss,
            'lm_loss': lm_loss,
            'rec_loss': rec_loss,
            'has_image_output': has_image_output,
            'image_embeds': recon_image_embeds,
        }

    def generate(
        self,
        tokenizer,
        prompt=None,
        input_ids=None,
        image_embeds=None,
        ids_cmp_mask=None,
        logits_processor=None,
        num_img_gen_tokens=64,
        temperature=0.7,
        num_beams=1,
        max_new_tokens=120,
        top_p=0.5,
    ):
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
            logits_processor.append(
                AutoImageTokenGenerationProcessor(tokenizer=tokenizer, num_img_gen_tokens=num_img_gen_tokens))

        if prompt is not None:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)

        input_embeds = self.llm.get_input_embeddings()(input_ids)
        bz, sq, dim = input_embeds.shape

        if image_embeds is not None:
            assert ids_cmp_mask is not None
            with torch.no_grad():
                image_embeds_lm = self.input_resampler(image_embeds)
            input_embeds[ids_cmp_mask] = image_embeds_lm.contiguous().view(-1, dim).to(dtype=self.llm.dtype)

        generation_config = {
            'temperature': temperature,
            'num_beams': num_beams,
            'max_new_tokens': max_new_tokens,
            'top_p': top_p,
            'do_sample': False
        }

        output = self.llm.generate(input_ids=input_ids,
                                   inputs_embeds=input_embeds,
                                   output_hidden_states=True,
                                   return_dict_in_generate=True,
                                   logits_processor=logits_processor,
                                   **generation_config)

        generate_ids = output.sequences[0][input_ids.shape[1]:]
        eoi_token_id = tokenizer.encode(EOI_TOKEN, add_special_tokens=False)[1]
        image_gen_ids = tokenizer.encode(''.join([IMG_TOKEN.format(int(item)) for item in range(num_img_gen_tokens)]), add_special_tokens=False)[1:]
        image_gen_ids = torch.Tensor(image_gen_ids).to(device=generate_ids.device, dtype=torch.int)

        last_hidden_states = torch.cat([hidden_state[-1] for hidden_state in output.hidden_states], dim=1)[0, input_ids.shape[1]:, :]

        eoi_indices = torch.where(generate_ids == eoi_token_id)[0].tolist()
        num_gen_imgs = len(eoi_indices)
        has_img_output = num_gen_imgs > 0
        ids_gen_mask = torch.zeros_like(generate_ids, dtype=torch.bool)
        if has_img_output:
            img_gen_feats = []
            for eoi_idx in eoi_indices:
                if eoi_idx >= num_img_gen_tokens:
                    img_gen_feats.append(last_hidden_states[eoi_idx - num_img_gen_tokens:eoi_idx])
                    generate_ids[eoi_idx - num_img_gen_tokens:eoi_idx] = image_gen_ids
                    ids_gen_mask[eoi_idx - num_img_gen_tokens:eoi_idx] = True

            img_gen_feats = torch.stack(img_gen_feats).to(dtype=self.dtype())
            img_gen_feat = self.output_resampler(img_gen_feats).contiguous()
        else:
            img_gen_feat = None

        generate_text = tokenizer.decode(generate_ids, skip_special_tokens=True)

        return {
            'text': generate_text,
            'output_ids': generate_ids,
            'img_gen_feat': img_gen_feat,
            'num_gen_imgs': num_gen_imgs,
            'ids_gen_mask': ids_gen_mask
        }

    @classmethod
    def from_pretrained(cls, llm, input_resampler, output_resampler, pretrained_model_path=None, **kwargs):
        model = cls(llm=llm, input_resampler=input_resampler, output_resampler=output_resampler, **kwargs)
        if pretrained_model_path is not None:
            ckpt = torch.load(pretrained_model_path, map_location='cpu')
            missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
            # print(f"agent model missing keys: {missing_keys}, unexpected keys: {unexpected_keys}")
        return model

    def dtype(self):
        return next(self.input_resampler.parameters()).dtype