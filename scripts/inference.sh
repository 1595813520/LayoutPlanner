CUDA_VISIBLE_DEVICES=7 python DiffSensei-main/layout-generator/infer.py \
    --config DiffSensei-main/layout-generator/configs/infer.yaml \
    --checkpoint /data/DiffSensei-main/layout-generator/checkpoints/0903_/planner_best.pt \
    --test_json /data/DiffSensei-main/layout-generator/checkpoints/test.json \
    --image_dir DiffSensei-main/checkpoints/mangazero/images \
    --output_dir DiffSensei-main/layout-generator/outputs/0905


python infer_new.py --config configs/infer.yaml \
    --checkpoint /data/checkpoints/planner_step4000.pt \
    --test_json test.json \
    --image_dir ./images \
    --output_dir ./preds