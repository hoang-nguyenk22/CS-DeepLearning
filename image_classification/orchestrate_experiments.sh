#!/bin/bash

# Strictly Controlled Orchestrator for FGVC Aircraft (Grid Search)
# 8 Configurations (Apples-to-Apples Comparisons)

source venv/bin/activate
mkdir -p logs
mkdir -p checkpoints

# Helper function
run_experiment() {
    MODEL=$1
    STRATEGY=$2     # "probing", "finetune", or "llrd"
    AUGMENT=$3      # "light" or "strong"
    SEED=$4
    EPOCHS=20       # Fixed for controlled comparison
    BATCH_SIZE=32   # Fixed for controlled comparison

    echo "----------------------------------------------------------"
    echo " RUNNING: Model=$MODEL | Strategy=$STRATEGY | Aug=$AUGMENT | Seed=$SEED "
    echo "----------------------------------------------------------"

    EXTRA_ARGS=""
    if [ "$STRATEGY" == "probing" ]; then
        EXTRA_ARGS="--freeze_backbone"
        LR=1e-3
    elif [ "$STRATEGY" == "llrd" ]; then
        EXTRA_ARGS="--llrd"
        if [ "$MODEL" == "vit_b_16" ]; then LR=5e-5; else LR=1e-4; fi
    else
        # Full Fine-Tune
        if [ "$MODEL" == "vit_b_16" ]; then LR=5e-5; else LR=1e-4; fi
    fi

    python train.py \
        --model $MODEL \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --augment_level $AUGMENT \
        --lr $LR \
        --seed $SEED \
        $EXTRA_ARGS \
        --data_dir ./data \
        --log_dir logs \
        --output_dir checkpoints

    STATUS=$?
    if [ $STATUS -eq 0 ]; then echo "Experiment Succeeded."
    else echo "Experiment Failed or OOMed (Code $STATUS). Skipping..."; fi
    echo ""
}

# CHUỖI 30 kịch bản (10 Grid x 3 Seeds)
# Grid: Probing + FineTune + LLRD  x  Light + Strong  x  ResNet + ViT

for SEED in 42 43 44; do
    echo "----------------------------------------------------------"
    echo " SEED: $SEED"
    echo "----------------------------------------------------------"

    # ResNet50: Probing (light only — no point in strong augment when backbone frozen)
    run_experiment "resnet50" "probing"  "light"  $SEED
    # ResNet50: Full Fine-Tune
    run_experiment "resnet50" "finetune" "light"  $SEED
    run_experiment "resnet50" "finetune" "strong" $SEED
    # ResNet50: LLRD
    run_experiment "resnet50" "llrd"     "light"  $SEED
    run_experiment "resnet50" "llrd"     "strong" $SEED

    # ViT-B/16: Probing
    run_experiment "vit_b_16" "probing"  "light"  $SEED
    # ViT-B/16: Full Fine-Tune
    run_experiment "vit_b_16" "finetune" "light"  $SEED
    run_experiment "vit_b_16" "finetune" "strong" $SEED
    # ViT-B/16: LLRD (most impactful here)
    run_experiment "vit_b_16" "llrd"     "light"  $SEED
    run_experiment "vit_b_16" "llrd"     "strong" $SEED
done

echo "Experiments complete. Use plot_results.py for visualization."
