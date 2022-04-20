
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
`mean`: The mean value of the features, note this has to be recomputed for different data augmentation.
`std`: Standard deviation of the features.
`freqm`, `timem`: time and frequency mask length for SpecAugmentation.
*For TALtrans Models*
`embedding_size`: embedding size 
`n_conv_layers`: number of conv layers before the bottleneck, default set to 10
`n_pool_layers`: number of pooling layers before the bottleneck, default set to 5
`n_trans_layers`: number of transformer block layers
`kernel_size`: conv kernel size

**Input:** Tensor in shape `[batch_size, temporal_frame_num, frequency_bin_num]`. Note: the input spectrogram should be normalized with dataset mean and std.
**Output:** Tensor of raw logits (i.e., without Sigmoid) in shape `[batch_size, label_dim]`.

## Audioset Recipe  
Audioset is the largest general Audio dataset you can find as of 2022, you will need to prepare your data json files (i.e., `train_data.json` and `eval_data.json`) using the scripts provided here.
The raw wavefiles of Audioset is not released by Google due to Copyright constraints, and you need to download them by yourself. 
We are considering releasing a version here just for research purpose (yet to be done).
Once you have downloaded the .wav files of audioset.(balanced train, unbalanced train, and eval).
You can use our prep_AudioSet.py to generate a sample json file in `~/egs/audioset/data/datafiles`, please keep the label code consistent with `~/egs/audioset/data/class_labels_indices.csv`. You can see the prep_audioset.ipynb for the brief logic.

Once you have the json files, you will need to generate the sampling weight file of your training data.
```
cd ~/egs/audioset
python gen_weight_file.py ./data/datafiles/train_data.json
```

Then you just need to change the `tr_data` and `te_data` in `~/egs/audioset/run.sh` and then 
cd ~/egs/audioset
./run-{modelrecipe_name}.sh


3. In ``~/src/run.py``, line 60-65, you need to add the normalization stats, the input frame length, and if noise augmentation is needed for your dataset. Also take a look at line 101-127 if you have a seperate validation set. For normalization stats, you need to compute the mean and std of your dataset (check ``ast/src/get_norm_stats.py``) or you can try using our AudioSet normalization ``input_spec = (input_spec + mean) / (std * 2)``.

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

## Contact
Feel free to shoot me an email at junchenl@cs.cmu.edu in case you have long questions.
For short questions, Twitter DM @JunchengLi
Or just submit an issue here.