#!/usr/bin/env bash

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
WORK_DIR=$4
MASTER_ADDR='127.0.0.1'

mkdir -p ${WORK_DIR}

srcdir="/gpfs/scratch/jingwezhang/data/NCT-CRC-HE/original_data/NCT-CRC-HE-100K-NONORM"
dstdir="/tmp/jingwezhang/NCT-CRC-HE-100K-NONORM"
if [ ! -d ${dstdir} ]; then
  mkdir -p $dstdir
  echo "Copy NCT data."
  cp -r $srcdir/* $dstdir
  echo "Copying NCT data finished"
fi

PYTHONPATH=".":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:4 \
    --ntasks=4 \
    --ntasks-per-node=64 \
    --cpus-per-task=16 \
    --kill-on-bad-exit=1 \
    -N 1  -t 7:59:49\
    python -u -W ignore tools/train.py ${CONFIG} \
        --work_dir ${WORK_DIR} --seed 0 --launcher "slurm" \
    &> ${WORK_DIR}/nohup_output.out

rm -r ${dstdir}
