#/bin/bash

# CIL CONFIG
NOTE="er_freq_balanced" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="er_freq_balanced"
DATASET="HS_TOD_domain" # VOC_10_10 BDD_domain SHIFT_domain MILITARY_SYNTHETIC_domain_1 MILITARY_SYNTHETIC_domain_2 MILITARY_SYNTHETIC_domain_3
# DATASET="SHIFT_domain_small" # VOC_10_10 BDD_domain SHIFT_domain MILITARY_SYNTHETIC_domain_1 MILITARY_SYNTHETIC_domain_2 MILITARY_SYNTHETIC_domain_3
# DATASET="VOC_15_5"
SIGMA=10
REPEAT=1
USE_AMP="--use_amp"
SEEDS="1"

if [ "$DATASET" == "VOC_10_10" ]; then
    MEM_SIZE=1000 ONLINE_ITER=1
    MODEL_NAME="damo" EVAL_PERIOD=500
    BATCHSIZE=16; LR=1e-4 OPT_NAME="SGD" SCHED_NAME="default"
elif [ "$DATASET" == "VOC_15_5" ]; then
    MEM_SIZE=1000 ONLINE_ITER=1
    MODEL_NAME="damo" EVAL_PERIOD=500
    BATCHSIZE=16; LR=1e-4 OPT_NAME="SGD" SCHED_NAME="default"
elif [ "$DATASET" = "COCO_70_10" ] || [ "$DATASET" = "COCO_60_20" ]; then
    MEM_SIZE=5000 ONLINE_ITER=0.5
    MODEL_NAME="damo" EVAL_PERIOD=1000
    BATCHSIZE=16; LR=2e-5 OPT_NAME="SGD" SCHED_NAME="default"
elif [ "$DATASET" == "HS_TOD_domain" ]; then
    MEM_SIZE=1000 ONLINE_ITER=1
    MODEL_NAME="damo" EVAL_PERIOD=1000
    BATCHSIZE=16; LR=1e-4 OPT_NAME="SGD" SCHED_NAME="default"
elif [ "$DATASET" == "HS_TOD_class" ]; then
    MEM_SIZE=1000 ONLINE_ITER=1
    MODEL_NAME="damo" EVAL_PERIOD=1000
    BATCHSIZE=16; LR=1e-4 OPT_NAME="SGD" SCHED_NAME="default"
elif [ "$DATASET" == "VisDrone_3_4" ]; then
    MEM_SIZE=1000 ONLINE_ITER=1
    MODEL_NAME="damo" EVAL_PERIOD=500
    BATCHSIZE=16; LR=2e-4 OPT_NAME="SGD" SCHED_NAME="default"
elif [ "$DATASET" == "BDD_domain" ]; then
    MEM_SIZE=500 ONLINE_ITER=1
    MODEL_NAME="damo" EVAL_PERIOD=1000
    BATCHSIZE=16; LR=1e-5 OPT_NAME="SGD" SCHED_NAME="default"
elif [ "$DATASET" == "SHIFT_domain_small" ]; then
    MEM_SIZE=500 ONLINE_ITER=1
    MODEL_NAME="damo" EVAL_PERIOD=1000
    BATCHSIZE=16; LR=1e-5 OPT_NAME="SGD" SCHED_NAME="default"
elif [ "$DATASET" == "SHIFT_domain_small2" ]; then
    MEM_SIZE=500 ONLINE_ITER=1
    MODEL_NAME="damo" EVAL_PERIOD=1000
    BATCHSIZE=16; LR=2e-6 OPT_NAME="SGD" SCHED_NAME="default"
elif [[ "$DATASET" == *"SHIFT_hanhwa"* ]]; then
    MEM_SIZE=500 ONLINE_ITER=1
    MODEL_NAME="damo" EVAL_PERIOD=1000
    BATCHSIZE=16; LR=2e-6 OPT_NAME="SGD" SCHED_NAME="default"
elif [[ "$DATASET" == "MILITARY_SYNTHETIC_domain_1" || \
        "$DATASET" == "MILITARY_SYNTHETIC_domain_2" || \
        "$DATASET" == "MILITARY_SYNTHETIC_domain_3" ]]; then
    MEM_SIZE=500 ONLINE_ITER=1
    MODEL_NAME="damo" EVAL_PERIOD=4000
    BATCHSIZE=16; LR=1e-5 OPT_NAME="SGD" SCHED_NAME="default"

else
    echo "Undefined setting"
    exit 1
fi

for RND_SEED in $SEEDS
do
    CUDA_VISIBLE_DEVICES=$1 python main.py --mode $MODE --n_worker 8 \
    --dataset $DATASET --sigma $SIGMA --repeat $REPEAT \
    --rnd_seed $RND_SEED \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE \
    --memory_size $MEM_SIZE  --online_iter $ONLINE_ITER \
    --note $NOTE --eval_period $EVAL_PERIOD $USE_AMP # > new_${MODE}_${DATASET}_mem${MEM_SIZE}_lr${LR}_online_iter${ONLINE_ITER}_seed${RND_SEED}_mosaic${MOSAIC}_mixup${MIXUP}.out 2>&1 &
done