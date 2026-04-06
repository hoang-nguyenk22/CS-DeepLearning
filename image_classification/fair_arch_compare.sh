#!/bin/bash
# fair_arch_compare.sh
# Phase F: ResNet50 @ LR=5e-5, 20 epochs, tag=fairlr  (~12 min)
# Phase L: ResNet50 + ViT-B/16 @ LR=5e-5, 50 epochs, tag=longtrain  (~3.5h)

set -e
source venv/bin/activate
mkdir -p logs checkpoints

run() {
    MODEL=$1; TAG=$2; LR=$3; EPOCHS=$4; SEED=$5; PATIENCE=$6; AUG=${7:-light}
    echo "============================================================"
    echo " $MODEL | tag=$TAG | lr=$LR | ep=$EPOCHS | seed=$SEED | aug=$AUG"
    echo "============================================================"
    python train.py \
        --model "$MODEL" --epochs "$EPOCHS" --batch_size 32 \
        --augment_level "$AUG" --lr "$LR" --seed "$SEED" \
        --exp_tag "$TAG" --patience "$PATIENCE" \
        --data_dir ./data --log_dir logs --output_dir checkpoints
    STATUS=$?
    if [ $STATUS -eq 0 ]; then echo "Done."; else echo "Failed (code $STATUS)."; fi
    echo ""
}

echo "Phase F — Fair Architecture Comparison (20 epochs)"
for SEED in 42 43 44; do
    run "resnet50" "fairlr" "5e-5" 20 $SEED 5 "light"
done

echo "Phase L — Long-Train Comparison (50 epochs)"
for SEED in 42 43 44; do
    # patience=10: avoid early-stop before ViT crossover appears
    run "resnet50"  "longtrain" "5e-5" 50 $SEED 10 "light"
    run "vit_b_16"  "longtrain" "5e-5" 50 $SEED 10 "light"
done

echo "All runs complete. Use plot_results.py for visualization."
