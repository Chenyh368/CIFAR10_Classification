#!/bin/bash
trap "exit" INT

LOG_DIR="$HOME/log/NonParaHW"
DATA_DIR="$HOME/DATA"
NUM_WORKERS=2

if [[ $1 = 'fine_tuning' ]]; then
  shift
  python ./main.py \
    --run_name "classification-cifar10-$1" --exp_name "resnet50" \
    --data_dir $DATA_DIR --log_dir $LOG_DIR --num_workers $NUM_WORKERS \
    --batch 64 --optimizer 'sgd' --lr 1e-3 --pretrained --epoch 5 \
    "$@"
elif [[ $1 = 'from_scratch' ]]; then
  shift
  python ./main.py \
    --run_name "classification-cifar10-$1" --exp_name "resnet50" \
    --data_dir $DATA_DIR --log_dir $LOG_DIR --num_workers $NUM_WORKERS \
    --batch 64 --optimizer 'adam' --lr 1e-3 --epoch 100 \
    "$@"
else
    echo "invalid args, check"
    exit 1
fi

trap - INT
echo "END."