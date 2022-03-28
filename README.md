
# AudioTagging Done Right: 2nd comparison of deep learning methods for environmental sound classification
 - [Introduction](#Introduction)
 - [Citing](#Citing)  
 - [AudioSet Recipe](#Audioset-Recipe)
 - [Contact](#Contact)
## Introduction  
This repository is the implementation of our submission to InterSpeech 2022: https://arxiv.org/abs/2203.13448 with the same title of the repository.
#### This implementation's is a new iteration of our previous effort: https://github.com/lijuncheng16/AudioSetDoneRight
We still maintain our previous repo for testing/probing the model. This repository is mainly for training models.
Thanks to the other opensourced project AST (https://github.com/YuanGongND/ast), where we used their data loading pipeline, and DeiT adaptation. 
The pre-trained DeiT model is loaded from the timm libraray: https://github.com/rwightman/pytorch-image-models (timm version: 0.4.5) newer version of timm changed the API for base-384 of deit, but should still be one liner to load the equivalent or better model.
Without these opensourced efforts, it would be way less efficient for researchers like us to make things happen.

## Getting Started 

**Parameters:**\
`label_dim` : The number of classes (default:`527`).\
`fstride`:  The stride of patch spliting on the frequency dimension, for 16\*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6 (used in the paper). (default:`10`)\
`tstride`:  The stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6 (used in the paper). (default:`10`)\
`input_fdim`: The number of frequency bins of the input spectrogram. (default:`128`)\
`input_tdim`: The number of time frames of the input spectrogram. (default:`1024`, i.e., 10.24s)\
`imagenet_pretrain`: If `True`, use ImageNet pretrained model. (default: `True`, we recommend to set it as `True` for all tasks.)\
`audioset_pretrain`: If`True`,  use full AudioSet And ImageNet pretrained model. Currently only support `base384` model with `fstride=tstride=10`. (default: `False`, we recommend to set it as `True` for all tasks except AudioSet.)\
`model_size`: The model size of AST, should be in `[tiny224, small224, base224, base384]` (default: `base384`).

**Input:** Tensor in shape `[batch_size, temporal_frame_num, frequency_bin_num]`. Note: the input spectrogram should be normalized with dataset mean and std, see [here](https://github.com/YuanGongND/ast/blob/102f0477099f83e04f6f2b30a498464b78bbaf46/src/dataloader.py#L191). \
**Output:** Tensor of raw logits (i.e., without Sigmoid) in shape `[batch_size, label_dim]`.

## Audioset Recipe  
Audioset is a little bit more complex, you will need to prepare your data json files (i.e., `train_data.json` and `eval_data.json`) by your self.
The reason is that the raw wavefiles of Audioset is not released and you need to download them by yourself. We have put a sample json file in `ast/egs/audioset/data/datafiles`, please generate files in the same format (You can also refer to `ast/egs/esc50/prep_esc50.py` and `ast/egs/speechcommands/prep_sc.py`.). Please keep the label code consistent with `ast/egs/audioset/data/class_labels_indices.csv`.

Once you have the json files, you will need to generate the sampling weight file of your training data.
```
cd ast/egs/audioset
python gen_weight_file.py ./data/datafiles/train_data.json
```

Then you just need to change the `tr_data` and `te_data` in `/ast/egs/audioset/run.sh` and then 
cd ast/egs/audioset
(slurm user) sbatch run.sh
(local user) ./run.sh


8. [Speechcommands V2-35, 10 tstride, 10 fstride, without Weight Averaging, Model (98.12% accuracy on evaluation set)](https://www.dropbox.com/s/q0tbqpwv44pquwy/speechcommands_10_10_0.9812.pth?dl=1)

If you want to use our training pipeline, you would need to modify below for your new dataset.
1. You need to create a json file, and a label index for your dataset, see ``ast/egs/audioset/data/`` for an example.
2. In ``/your_dataset/run.sh``, you need to specify the data json file path, the SpecAug parameters (``freqm`` and ``timem``, we recommend to mask 48 frequency bins out of 128, and 20% of your time frames), the mixup rate (i.e., how many samples are mixup samples), batch size, initial learning rate, etc. Please see ``ast/egs/[audioset,esc50,speechcommands]/run.sh]`` for samples.
3. In ``ast/src/run.py``, line 60-65, you need to add the normalization stats, the input frame length, and if noise augmentation is needed for your dataset. Also take a look at line 101-127 if you have a seperate validation set. For normalization stats, you need to compute the mean and std of your dataset (check ``ast/src/get_norm_stats.py``) or you can try using our AudioSet normalization ``input_spec = (input_spec + 4.26) / (4.57 * 2)``.
4. In ``ast/src/traintest.`` line 55-82, you need to specify the learning rate scheduler, metrics, warmup setting and the optimizer for your task.

To summarize, to use our training pipeline, you need to creat data files and modify the above three python scripts. You can refer to our ESC-50 and Speechcommands recipes.

Also, please note that we use `16kHz` audios for the pretrained model, so if you want to use the pretrained model, please prepare your data in `16kHz`.

## Citing
Please cite our paper(s) if you find this repository useful.
```  
@article{li2022ATbaseline,
  doi = {10.48550/ARXIV.2203.13448},
  url = {https://arxiv.org/abs/2203.13448},
  author = {Li, Juncheng B and Qu, Shuhui and Huang, Po-Yao and Metze, Florian},
  title = {AudioTagging Done Right: 2nd comparison of deep learning methods for environmental sound classification},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```  