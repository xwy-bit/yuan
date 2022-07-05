#! /bin/bash

TASKNAME=multi_node_i

NNODES=4
GPUS_PER_NODE=1
MASTER_PORT=12308
MASTER_ADDR=10.1.13.59
NODE_RANK=$1

echo I am icarus$1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
#LOAD_CHECKPOINT_PATH=./checkpoints/gpt3_case11_300B/
#SAVE_CHECKPOINT_PATH=./checkpoints/gpt3_case11_300B/
TENSORBOARD_PATH=./tensorboard/${TASKNAME}/${DATETIME}
LOGFILE=./Log/${TASKNAME}/${DATETIME}
VOCAB_FILE=vocab.txt
DATA_PATH=/home/nfs/ella/hpc/YLLM/yuan_database/012complete
python -u -m torch.distributed.launch $DISTRIBUTED_ARGS \
        ./pretrain_gpt.py \
        --tokenizer-type EncDecTokenizer \
        --vocab-file $VOCAB_FILE \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 4 \
        --num-layers 40 \
        --hidden-size 3072 \
        --num-attention-heads 24 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --micro-batch-size 4 \
        --global-batch-size 2800 \
        --train-samples 488282 \
	--rampup-batch-size 40 2760 20000 \
        --lr-decay-samples 439453 \
        --lr-warmup-samples 20000 \
        --lr 6.0e-04 \
        --min-lr 6.0e-05 \
        --lr-decay-style cosine \
        --log-interval 1 \
        --eval-iters -1 \
        --data-path ${DATA_PATH} \
        --save-interval 2000 \
        --split 100,0,0 \
        --clip-grad 1.0 \
        --weight-decay 0.01 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.002 \
        --fp16 \
	--DDP-impl local \
        --checkpoint-activations \
        --checkpoint-num-layers 1 \
        --log-num-zeros-in-grad \
        --log-params-norm \
        --distributed-backend nccl \
        --tensorboard-dir $TENSORBOARD_PATH \
        --tensorboard-log-interval 1 \
        2>&1 | tee ${LOGFILE}	
