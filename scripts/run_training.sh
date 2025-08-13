python -m torch.distributed.launch \
       --nproc_per_node 4 \
       scripts/train.py \
       --config_file ./layout-generator/configs/planner.yaml