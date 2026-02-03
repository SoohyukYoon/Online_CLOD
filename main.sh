#/bin/bash

# CIL CONFIG
NOTE="er_pseudo" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="er_pseudo"
DATASET="SHIFT_domain_small2" # VOC_10_10 BDD_domain SHIFT_domain MILITARY_SYNTHETIC_domain_1 MILITARY_SYNTHETIC_domain_2 MILITARY_SYNTHETIC_domain_3
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
    BATCHSIZE=16; LR=1e-6 OPT_NAME="SGD" SCHED_NAME="default"
    SCORE_THRESHOLD=0.45 TEMP_BATCHSIZE=4
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
    --score_threshold $SCORE_THRESHOLD  --temp_batchsize $TEMP_BATCHSIZE \
    --note $NOTE --eval_period $EVAL_PERIOD $USE_AMP > ${DATASET}_${TEMP_BATCHSIZE}_${SCORE_THRESHOLD}_${1}_direct_path.out 2>&1 &
done