#!/bin/bash

export TORCH_HOME=../../pretrained_models
hostname=$(hostname | cut -d'.' -f1)
echo "Current Machine is ${hostname}"
slurm_id=$(squeue -u billyli | grep $hostname | awk '{ print $1; }' )
echo "Slurm ID is $slurm_id"
python prep_AudioSet.py --slurm-id ${slurm_id}
slurm_folder=/local/slurm-${slurm_id}/local/audio/data/datafiles
model=ast
dataset=audioset_s
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
  lr=1e-5
  epoch=5
  tr_data=${slurm_folder}/audioset_bal_unbal_train_data.json
fi
te_data=${slurm_folder}/audioset_eval_data.json
freqm=24
n_mels=64
timem=75
mixup=0.5
# corresponding to overlap of 6 for 16*16 patches
fstride=10
tstride=10
batch_size=448
exp_dir=./exp/${dataset}-${set}-f$fstride-t$tstride-p$imagenetpretrain-b$batch_size-lr${lr}-ast_debug
expid=${model}-${set}-f${fstride}-t${tstride}-pre${imagenetpretrain}-b${batch_size}-lr${lr}-mix${mixup}-freqm${freqm}-timem${timem}
logger=${exp_dir}/${expid}.txt
if [ -d $exp_dir ]; then
  echo 'exp exist'
  exit
fi
mkdir -p $exp_dir
echo ${model}
echo "CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} --n_mels ${n_mels}\
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ./data/class_labels_indices.csv --n_class 527 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain" >> $logger
CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} --n_mels ${n_mels} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ./data/class_labels_indices.csv --n_class 527 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain
