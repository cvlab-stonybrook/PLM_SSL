#!/usr/bin/env bash

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
WORK_DIR=$4
MASTER_ADDR='127.0.0.1'

mkdir -p ${WORK_DIR}

PYTHONPATH=".":$PYTHONPATH \
nohup srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:4 \
    --ntasks=4 \
    --ntasks-per-node=64 \
    --cpus-per-task=16 \
    --kill-on-bad-exit=1 \
    -N 1  -t 7:59:49\
    python -u -W ignore tools/train.py ${CONFIG} \
        --work_dir ${WORK_DIR} --seed 0 --launcher "slurm" \
    &> ${WORK_DIR}/nohup_output.out &