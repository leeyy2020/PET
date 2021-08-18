 #!/bin/bash

#source "${HOME}/share/env.sh"

PYTHON_BIN="/usr/bin/python"
task_name=$1
gpus=$2
index=$3
turn=$4

#task_data_dir="/home/tianxin04/develop/FewCLUE/datasets/"
# local_log_path="/home/tianxin04/global_data/paddlenlp/examples/few_shot/pet_erni1p0/"
local_log_path="/home/Code/PET/Rdrop/log_pet_${pretrained_model}_rdrop"
submit_dir="/home/Code/PET/Rdrop/log_pet_${pretrained_model}_rdrop/submit"
#task_name="iflytek"
#task_name="tnews"
#task_name="eprstmt"
#task_name="bustm"
#task_name="ocnli"
#task_name="csl"
#task_name="csldcp"
#task_name="cluewsc"
#task_name="chid"

#gpus=7
alpha=(0.1 0.25 0.5 0.75 1)

language_model="ernie-1.0"

if [[ ${task_name} == "iflytek" || ${task_name} == "csl" || ${task_name} == "csldcp" ]]; then
	batch_size=(8 16)
else
	batch_size=(8 16 32)
fi

if [[ ${task_name} == "chid" ]]; then
	pattern_id=(0)
else
	pattern_id=(0 1 2 3)
fi
# batch_size=(8)
# pattern_id=(0)
learning_rate=(5E-4 1E-4 5E-5 1E-5)
# learning_rate=(1E-4 5E-5 1E-5)
# learning_rate=(1E-4)
epoch=10
max_seq_len=512

function train() {
	local task_name=$1
	local lr=$2
	local bs=$3
    local pattern_id=$4
    local lm=$5
	local al=$6

	strategy="bs${bs}_lr${lr}_patternId${id}_al${al}"
	save_checkpoint_dir="${local_log_path}/checkpoints/${strategy}/${task_name}"
	output_dir="${local_log_path}/output/${strategy}/${task_name}"
	log_dir="${local_log_path}/log/${strategy}/${task_name}"
	log_file="${log_dir}/index${index}_log"

	mkdir -p ${save_checkpoint_dir}
	mkdir -p ${log_dir}
	mkdir -p ${output_dir}

	train_script="pet.py"

	cmd="${PYTHON_BIN} -u -m paddle.distributed.launch --gpus "${gpus}" --log_dir launch_log/${strategy}/${task_name} \
		${train_script} \
		--task_name ${task_name} \
		--device gpu \
        --pattern_id ${pattern_id} \
		--save_dir ${save_checkpoint_dir} \
		--index ${index} \
		--output_dir ${output_dir} \
		--batch_size ${bs} \
		--learning_rate ${lr} \
		--epochs ${epoch} \
		--max_seq_length ${max_seq_len} \
		--language_model ${lm} \
        --alpha ${al}  \
		> ${log_file} 2>/dev/null"

	echo $cmd
	eval $cmd
}


function train_wrapper() {
	for lr in ${learning_rate[@]}; do
	for bs in ${batch_size[@]}; do
	for al in ${alpha[@]};do
    for id in ${pattern_id[@]}; do
		echo "[strat training] ${task_name} ${lr} ${bs} ${id} ${language_model} ${al}"
		train ${task_name} ${lr} ${bs} ${id} ${language_model} ${al}
    done
	done
	done
	done
}

function get_max_result() {

	:> "${local_log_path}/output/index${index}_${task_name}_result"
	:> "${local_log_path}/output/index${index}_${task_name}_result_all"
	for lr in ${learning_rate[@]}; do
	for bs in ${batch_size[@]}; do
	for al in ${alpha[@]}; do
    for id in ${pattern_id[@]}; do
		strategy="bs${bs}_lr${lr}_patternId${id}_al${al}"

		output_dir="${local_log_path}/output/${strategy}/${task_name}"

		log_dir="${local_log_path}/log/${strategy}/${task_name}"
		log_file="${log_dir}/index${index}_log"
		
		grep "test_accuracy" ${log_file} > ${output_dir}/index${index}_test_acc
		# grep "test_accuracy" ${log_file} > ${output_dir}/test_acc
		cat ${output_dir}/index${index}_test_acc | ${PYTHON_BIN} get_max.py ${strategy} 1>> "${local_log_path}/output/index${index}_${task_name}_result" 2>> "${local_log_path}/output/index${index}_${task_name}_result_all"

    done
    done
	done
	done
	
	${PYTHON_BIN} get_final.py ${local_log_path}/output/index${index}_${task_name}_result \
	"${local_log_path}/output" \
	${task_name} \
	${submit_dir} \
	> ${local_log_path}/output/index${index}_${task_name}_result_final
}

train_wrapper
echo "Train ends"
get_max_result
