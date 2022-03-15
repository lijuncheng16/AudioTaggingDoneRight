#!/bin/bash

export TORCH_HOME=../../pretrained_models
hostname=$(hostname | cut -d'.' -f1)
echo "Current Machine is ${hostname}"
slurm_id=$(squeue -u billyli | grep $hostname | awk '{ print $1; }')
echo "Slurm ID is __${slurm_id}__"
python prep_AudioSet.py --slurm-id ${slurm_id}
slurm_folder=/local/slurm-${slurm_id}/local/audio/data/datafiles
model=TAL
dataset=audioset
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
  lr=4e-4
  epoch=10
  tr_data=${slurm_folder}/audioset_bal_unbal_train_data.json
fi
te_data=${slurm_folder}/audioset_eval_data.json
freqm=36
n_mels=128
timem=192
mixup=0.3
#TAL-trans
embedding_size=2048
n_conv_layers=10
n_pool_layers=5
n_trans_layers=2
kernel_size=3
bn=True
dropout=0
pooling=att
addpos=True
transformer_dropout=0.25

batch_size=100
mean=-3.6788461
std=4.3739743
suffix=big_feat_embed
exp_dir=./exp/${model}-${dataset}-${set}-p$imagenetpretrain-b$batch_size-lr${lr}-fm${freqm}-tm${timem}-mix${mixup}-m${mean}-std${std}-epoch${epoch}-conv${n_conv_layers}-pool${n_pool_layers}-trans${n_trans_layers}-${suffix}
expid=${model}-${set}-pre${imagenetpretrain}-b${batch_size}-lr${lr}-mix${mixup}-freqm${freqm}-timem${timem}-m${mean}-std${std}-pos${addpos}
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
--batch_norm $bn --dropout $dropout --pooling $pooling --addpos $addpos --transformer_dropout $transformer_dropout --n_trans_layers $n_trans_layers \
--embedding_size $embedding_size --n_conv_layers $n_conv_layers --n_pool_layers $n_pool_layers --imagenet_pretrain $imagenetpretrain" >> $logger
CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} --n_mels ${n_mels} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir --mean ${mean} --std ${std} \
--label-csv ./data/class_labels_indices.csv --n_class 527 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--batch_norm $bn --dropout $dropout --pooling $pooling --addpos $addpos --transformer_dropout $transformer_dropout --n_trans_layers $n_trans_layers \
--embedding_size $embedding_size --n_conv_layers $n_conv_layers --n_pool_layers $n_pool_layers --imagenet_pretrain $imagenetpretrain
