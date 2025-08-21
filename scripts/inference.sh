python DiffSensei-main/layout-generator/infer.py \
    --config DiffSensei-main/layout-generator/configs/planner.yaml \
    --checkpoint DiffSensei-main/layout-generator/checkpoints/planner_best.pt \
    --test_json DiffSensei-main/layout-generator/configs/test.json \
    --image_dir DiffSensei-main/checkpoints/mangazero/images \
    --output_dir DiffSensei-main/layout-generator/outputs
