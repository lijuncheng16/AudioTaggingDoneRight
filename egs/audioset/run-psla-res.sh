#!/bin/bash

export TORCH_HOME=../../pretrained_models
hostname=$(hostname | cut -d'.' -f1)
echo "Current Machine is ${hostname}"
slurm_id=$(squeue -u billyli | grep $hostname | awk '{ print $1; }')
echo "Slurm ID is __${slurm_id}__"
python prep_AudioSet.py --slurm-id ${slurm_id}
slurm_folder=/local/slurm-${slurm_id}/local/audio/data/datafiles
model=resnet
dataset=audioset
att_head=4
eff_b=2
# full or balanced for audioset
set=full
imagenetpretrain=True
if [ $set == balanced ]
then
  bal=none
  lr=5e-5
  epoch=25
  tr_data=${slurm_folder}/audioset_bal_train_data.json
else
  bal=bal
  lr=1e-4
  epoch=10
  tr_data=${slurm_folder}/audioset_bal_unbal_train_data.json
fi
te_data=${slurm_folder}/audioset_eval_data.json
freqm=48
n_mels=128
timem=192
mixup=0.5

batch_size=128
mean=-3.3458831
std=4.1563106
suffix=psla_dev
exp_dir=./exp/${model}-${dataset}-${set}-p$imagenetpretrain-b$batch_size-lr${lr}-fm${freqm}-tm${timem}-mix${mixup}-m${mean}-std${std}-epoch${epoch}-${suffix}
expid=${model}-${set}-pre${imagenetpretrain}-b${batch_size}-lr${lr}-mix${mixup}-freqm${freqm}-timem${timem}-m${mean}-std${std}
logger=${exp_dir}/${expid}.txt
if [ -d $exp_dir ]; then
  echo 'exp exist'
  exit
fi
mkdir -p $exp_dir
echo ${model}
echo "CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} --n_mels ${n_mels}\
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir --mean ${mean} --std ${std} \
--label-csv ./data/class_labels_indices.csv --n_class 527 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--eff_b $eff_b --att_head ${att_head} --imagenet_pretrain $imagenetpretrain" >> $logger
CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} --n_mels ${n_mels} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir --mean ${mean} --std ${std} \
--label-csv ./data/class_labels_indices.csv --n_class 527 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--eff_b $eff_b --att_head ${att_head} --imagenet_pretrain $imagenetpretrain
