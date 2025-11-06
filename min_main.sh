#/bin/bash

# CIL CONFIG

MODE="er_selection_freqbalanced" # er_selection_frequency adaptive_freeze_selection
DATASET="COCO_70_10" # VOC_10_10 BDD_domain SHIFT_domain MILITARY_SYNTHETIC_domain_1 MILITARY_SYNTHETIC_domain_2 MILITARY_SYNTHETIC_domain_3
# DATASET="VOC_15_5"
SIGMA=10
REPEAT=1
INIT_CLS=100
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"
MOSAIC=0.0
MIXUP=0.0
SEEDS="1"

SELECTION_METHOD="loss"
PRIORITY_SELECTION="prob"
NOTE=${DATASET}_${MODE}_${SELECTION_METHOD}_${PRIORITY_SELECTION} # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)


if [ "$DATASET" == "VOC_10_10" ]; then
    MEM_SIZE=1000 ONLINE_ITER=1
    MODEL_NAME="damo" EVAL_PERIOD=500
    BATCHSIZE=16; LR=2e-4 OPT_NAME="SGD" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
elif [ "$DATASET" == "VOC_15_5" ]; then
    MEM_SIZE=1000 ONLINE_ITER=1 #1000
    MODEL_NAME="damo" EVAL_PERIOD=500
    BATCHSIZE=16; LR=1e-4 OPT_NAME="SGD" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
elif [ "$DATASET" == "COCO_70_10" || "$DATASET" == "COCO_60_20" ]; then
    MEM_SIZE=5000 ONLINE_ITER=0.5
    MODEL_NAME="damo" EVAL_PERIOD=1000
    BATCHSIZE=16; LR=2e-5 OPT_NAME="SGD" SCHED_NAME="default"
elif [ "$DATASET" == "BDD_domain" ]; then
    MEM_SIZE=10 ONLINE_ITER=1
    MODEL_NAME="yolov9-s" EVAL_PERIOD=1000
    BATCHSIZE=16; LR=3e-4 OPT_NAME="SGD" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
elif [ "$DATASET" == "SHIFT_domain_small" ]; then
    MEM_SIZE=500 ONLINE_ITER=1
    MODEL_NAME="damo5" EVAL_PERIOD=1000
    BATCHSIZE=16; LR=1e-6 OPT_NAME="SGD" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
elif [ "$DATASET" == "SHIFT_domain_small2" ]; then
    MEM_SIZE=500 ONLINE_ITER=1
    MODEL_NAME="damo5" EVAL_PERIOD=1000
    BATCHSIZE=16; LR=1e-6 OPT_NAME="SGD" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
elif [[ "$DATASET" == "MILITARY_SYNTHETIC_domain_1" || \
        "$DATASET" == "MILITARY_SYNTHETIC_domain_2" || \
        "$DATASET" == "MILITARY_SYNTHETIC_domain_3" ]]; then
    MEM_SIZE=500 ONLINE_ITER=1
    MODEL_NAME="yolov9-s" EVAL_PERIOD=4000
    BATCHSIZE=16; LR=3e-4 OPT_NAME="SGD" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

else
    echo "Undefined setting"
    exit 1
fi

for RND_SEED in $SEEDS
do
    CUDA_VISIBLE_DEVICES=$1 python main.py --mode $MODE --n_worker 8 \
    --dataset $DATASET \
    --sigma $SIGMA --repeat $REPEAT \
    --rnd_seed $RND_SEED \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE \
    --memory_size $MEM_SIZE --online_iter $ONLINE_ITER \
    --priority_selection $PRIORITY_SELECTION --selection_method $SELECTION_METHOD \
    --note $NOTE --eval_period $EVAL_PERIOD $USE_AMP #> ${NOTE}.out 2>&1 
done