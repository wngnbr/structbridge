export PYTHONPATH=$PYTHONPATH:./

DATASET_NAME=imagenet_inpaint_center
TRAIN_MODE=ddbm

source scripts/args.sh $DATASET_NAME

FREQ_SAVE_ITER=5000
EXP=${DATASET_NAME}-${TRAIN_MODE}

CKPT=workdir/imagenet_inpaint_center-ddbm-all-07-10-16:06:07/model_180000.pt

# For cluster
# export ADDR=$1
# run_args="--nproc_per_node 8 \
#           --master_addr $ADDR \
#           --node_rank $RANK \
#           --master_port $MASTER_PORT \
#           --nnodes $WORLD_SIZE"
# For local
export CUDA_VISIBLE_DEVICES=1
run_args="--nproc_per_node 1 \
          --master_port 29514"

MICRO_BS=4

torchrun $run_args train.py --exp=$EXP \
 --class_cond $CLASS_COND  \
 --dropout $DROPOUT  --microbatch $MICRO_BS \
 --image_size $IMG_SIZE  --num_channels $NUM_CH  \
 --num_res_blocks $NUM_RES_BLOCKS  --condition_mode=$COND  \
 --noise_schedule=$PRED    \
 --use_new_attention_order $ATTN_TYPE  \
 ${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"} ${BETA_MAX:+ --beta_max="${BETA_MAX}"}  \
 --data_dir=$DATA_DIR --dataset=$DATASET  \
 --sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN  \
 --save_interval_for_preemption=$FREQ_SAVE_ITER --save_interval=$SAVE_ITER --debug=False \
 ${CKPT:+ --resume_checkpoint="${CKPT}"} 