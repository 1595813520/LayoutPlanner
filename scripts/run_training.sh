python -m torch.distributed.launch \
       --nproc_per_node 4 \
       scripts/train.py \
       --config_file ./layout-generator/configs/planner.yaml


python DiffSensei-main/layout-generator/train.py \
       --config DiffSensei-main//layout-generator/configs/planner.yaml \
       --save_dir DiffSensei-main//layout-generator/checkpoints --epochs 20 --batch_size 4 