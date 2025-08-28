# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        bbox=None,
        dialog_bbox=None,
        aspect_ratio=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class MaskedIPAttnProcessor2_0(nn.Module):
    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_ip_tokens=4, num_dummy_tokens=4):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_ip_tokens = num_ip_tokens
        self.num_dummy_tokens = num_dummy_tokens

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def prepare_attention_mask_ip(self, bbox, hidden_states, head_size, aspect_ratio):
        """
        Args:
            bbox: Torch.Tensor. Relative bbox of each character. Shape (batch_size, max_num_ips, 4)
                each bbox contains [x1, y1, x2, y2]. The coordinates are relative to the height and width,
                ranging from 0 to 1.
            aspect_ratio: float. Height / width of the generating image.
                Used to reshape the 1D sequence_length into 2D image shapes.
        Returns:
            Torch.Tensor. The prepared attention mask that masks the area outside bbox.
                Shape (batch_size, heads, sequence_length, (self.num_dummy_masks + self.num_ip_tokens))
        """
        batch_size, sequence_length, _ = hidden_states.shape
        max_num_ips = bbox.shape[1]

        # Calculate height and width from aspect_ratio and sequence_length
        width = int((sequence_length / aspect_ratio) ** 0.5)
        height = sequence_length // width
        # Adjust width and height to ensure the product equals sequence_length
        while width * height != sequence_length:
            if width * height < sequence_length:
                width += 1
            else:
                width -= 1
            height = sequence_length // width
        
        # Initialize the attention masks
        attention_mask = torch.full((batch_size, head_size, sequence_length, max_num_ips), -10000.0, device=hidden_states.device, dtype=hidden_states.dtype)
        dummy_mask = torch.full((batch_size, head_size, sequence_length, 1), 0.0, device=hidden_states.device, dtype=hidden_states.dtype)

        # Create a grid of coordinates corresponding to each sequence position
        x_coords = torch.linspace(0, 1, steps=width, device=hidden_states.device)
        y_coords = torch.linspace(0, 1, steps=height, device=hidden_states.device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords)
        
        # Flatten the grid to match sequence length
        x_grid = x_grid.flatten()  # (sequence_length,)
        y_grid = y_grid.flatten()  # (sequence_length,)
        
        # Iterate over the batch and max_num_ips dimensions
        for batch_idx in range(batch_size):
            for ip_idx in range(max_num_ips):
                x1, y1, x2, y2 = bbox[batch_idx, ip_idx]

                inside_bbox_mask = (x_grid >= x1) & (x_grid <= x2) & (y_grid >= y1) & (y_grid <= y2)
                # print(batch_idx, ip_idx, sum(inside_bbox_mask), inside_bbox_mask.shape)

                attention_mask[batch_idx, :, inside_bbox_mask, ip_idx] = 0.0
                dummy_mask[batch_idx, :, inside_bbox_mask, 0] = -10000.0

        attention_mask = attention_mask.repeat_interleave(self.num_ip_tokens // max_num_ips, dim=-1)
        dummy_mask = dummy_mask.repeat_interleave(self.num_dummy_tokens, dim=-1)
        attention_mask = torch.cat([dummy_mask, attention_mask], dim=-1)

        return attention_mask

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        bbox=None,
        aspect_ratio=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - (self.num_ip_tokens + self.num_dummy_tokens)
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            attention_mask, _ = (
                attention_mask[:, :, :, :end_pos],
                attention_mask[:, :, :, end_pos:],
            ) if attention_mask is not None else (None, None)
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        new_query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            new_query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Encode IP
        attention_mask_ip = self.prepare_attention_mask_ip(bbox, hidden_states, attn.heads, aspect_ratio)

        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)
        new_query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        ip_hidden_states = F.scaled_dot_product_attention(
            new_query, ip_key, ip_value, attn_mask=attention_mask_ip, dropout_p=0.0, is_causal=False
        )

        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        ip_hidden_states = ip_hidden_states.to(query.dtype)

        hidden_states = hidden_states + self.scale * ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


