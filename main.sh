#/bin/bash

# CIL CONFIG
NOTE="er" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="er"
DATASET="VOC_10_10" # VOC_10_10 BDD_domain SHIFT_domain MILITARY_SYNTHETIC_domain_1 MILITARY_SYNTHETIC_domain_2 MILITARY_SYNTHETIC_domain_3
SIGMA=10
REPEAT=1
INIT_CLS=100
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"
MOSAIC=1.0
MIXUP=0.0
SEEDS="1"

if [ "$DATASET" == "VOC_10_10" ]; then
    MEM_SIZE=500 ONLINE_ITER=1
    MODEL_NAME="damo" EVAL_PERIOD=500
    BATCHSIZE=16; LR=3e-4 OPT_NAME="SGD" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
elif [ "$DATASET" == "BDD_domain" ]; then
    MEM_SIZE=10 ONLINE_ITER=1
    MODEL_NAME="yolov9-s" EVAL_PERIOD=1000
    BATCHSIZE=16; LR=3e-4 OPT_NAME="SGD" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "SHIFT_domain" ]; then
    MEM_SIZE=500 ONLINE_ITER=0.5
    MODEL_NAME="yolov9-s" EVAL_PERIOD=4000
    BATCHSIZE=16; LR=1e-4 OPT_NAME="SGD" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

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
    --sigma $SIGMA --repeat $REPEAT --init_cls $INIT_CLS\
    --rnd_seed $RND_SEED \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER \
    --mosaic_prob $MOSAIC --mixup_prob $MIXUP \
    --note $NOTE --eval_period $EVAL_PERIOD --imp_update_period $IMP_UPDATE_PERIOD $USE_AMP #> new_${MODE}_${DATASET}_mem${MEM_SIZE}_lr${LR}_online_iter${ONLINE_ITER}_seed${RND_SEED}_mosaic${MOSAIC}_mixup${MIXUP}.out 2>&1 &
done