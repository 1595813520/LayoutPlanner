python -m torch.distributed.launch \
       --nproc_per_node 4 \
       scripts/train.py \
       --config_file ./layout-generator/configs/planner.yaml

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 \
       DiffSensei-main/layout-generator/train.py \
       --config DiffSensei-main//layout-generator/configs/planner.yaml \
       --save_dir DiffSensei-main//layout-generator/checkpoints --epochs 100 --batch_size 16 


CUDA_VISIBLE_DEVICES=4 nohup python DiffSensei-main/layout-generator/train.py \
       --config DiffSensei-main//layout-generator/configs/planner.yaml \
       --save_dir DiffSensei-main//layout-generator/checkpoints/0817 --epochs 300 --batch_size 64 \
       > DiffSensei-main/layout-generator/outputs/0817.log 2>&1 &