python DiffSensei-main/layout-generator/infer.py \
    --config DiffSensei-main/layout-generator/configs/infer.yaml \
    --checkpoint checkpoints/planner_step10000.pt \
    --test_json DiffSensei-main/new/util/test.json \
    --image_dir DiffSensei-main/checkpoints/mangazero/images \
    --output_dir DiffSensei-main/layout-generator/outputs/0901


python infer_new.py --config configs/infer.yaml \
    --checkpoint /data/checkpoints/planner_step4000.pt \
    --test_json test.json \
    --image_dir ./images \
    --output_dir ./preds