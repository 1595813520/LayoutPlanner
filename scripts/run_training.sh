CUDA_VISIBLE_DEVICES=1 accelerate launch DiffSensei-main/layout-generator/train.py \
       --config DiffSensei-main//layout-generator/configs/train.yaml \
       --save_dir DiffSensei-main//layout-generator/checkpoints/0817 \
       --epochs 201 --batch_size 32

CUDA_VISIBLE_DEVICES=1 nohup python DiffSensei-main/layout-generator/train.py \
       --config DiffSensei-main//layout-generator/configs/train.yaml \
       --save_dir DiffSensei-main//layout-generator/checkpoints/0827 --epochs 201 --batch_size 64 \
       > DiffSensei-main/layout-generator/outputs/0827.log 2>&1 &