# py-MDNet

by [Hyeonseob Nam](https://kr.linkedin.com/in/hyeonseob-nam/) and [Bohyung Han](http://cvlab.postech.ac.kr/~bhhan/) at POSTECH

### Pretraining (where I am making changes)
 - Download [VGG-M](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m.mat) (matconvnet model) and save as "models/imagenet-vgg-m.mat" (This is what the code is currently designed to use)
 - Download [VOT](http://www.votchallenge.net/) datasets into "dataset/vot201x"
 - [VOT2013](http://www.votchallenge.net/vot2013/dataset.html)
 - [VOT2014](http://www.votchallenge.net/vot2014/dataset.html)
 - [VOT2016](http://www.votchallenge.net/vot2016/dataset.html), 2015 omitted on purpose
``` bash
 cd pretrain
 python prepro_data.py
 python train_mdnet.py
```
 - prepro_data.py should already be run, results stored in pretrain/data/vot-otb.pkl

## Introduction
Python (PyTorch) implementation of MDNet tracker, which is ~2x faster than the original matlab implementation. 
#### [[Project]](http://cvlab.postech.ac.kr/research/mdnet/) [[Paper]](https://arxiv.org/abs/1510.07945) [[Matlab code]](https://github.com/HyeonseobNam/MDNet)

If you're using this code for your research, please cite:

	@InProceedings{nam2016mdnet,
	author = {Nam, Hyeonseob and Han, Bohyung},
	title = {Learning Multi-Domain Convolutional Neural Networks for Visual Tracking},
	booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	month = {June},
	year = {2016}
	}
 
## Prerequisites
- python 2.7
- [PyTorch](http://pytorch.org/) and its dependencies 

## Usage

### Tracking
```bash
 cd tracking
 python run_tracker.py -s DragonBaby [-d (display fig)] [-f (save fig)]
```
 - You can provide a sequence configuration in two ways (see tracking/gen_config.py):
   - ```python run_tracker.py -s [seq name]```
   - ```python run_tracker.py -j [json path]```
