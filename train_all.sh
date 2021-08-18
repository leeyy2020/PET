#!/bin/bash

#export pretrained_model="ernie1p0"
export name="pet"
# export pretrained_model="ernie1p0"
export pretrained_model="ernie1p0"

task_name=$1
gpus=$2

indexs=(0)
# indexs=(0 1 2)
# indexs=(3 4 few_all)
#indexs=(3)
#indexs=(4)
# indexs=(few_all)
#indexs=(0 few_all)


for index in ${indexs[@]}; do
	bash train_r.sh ${task_name} ${gpus} ${index}
	# bash commands/train.sh ${task_name} ${gpus} ${index} ${t} >> log_t/${name}_train_${pretrained_model}_${task_name}_index${index}_turn${t}.log 2>&1
done