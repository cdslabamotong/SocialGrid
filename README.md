# SocialGrid
A novel framework for forecasting thread dynamics in online discussion forums

This is a implementation of the framework SocialGrid, as described in our paper:  
Chen Ling, Ruiqi Wang, and Guangmo Tong, [SocialGrid: A TCN-enhanced Method for Online Discussion Forecasting](https://arxiv.org/abs/2003.07189)

## Requirements
This code is written in Python. To use it you will need:
- Numpy > 1.16
- Tensorflow > 2.0
- pandas

## Usage
### Run the demo
```
python train_reply.py
python train_main.py
```

### Data
The data used in the paper can be access with this [link](https://drive.google.com/drive/folders/1uZudmS2y9npqG0sbfLy6AlduwFG32Kbg?usp=sharing). Please put the data folder in the folder of SocialGrid to correctly read it.

In order to use your own data, you'll have to provide:
- An N size array recording the arrival time of N main threads
- An N by D ndarray recording the arrival time of D replies of each main threads. Note that the number of D for each replies list can be different. 

[Pushshift](https://github.com/pushshift/api) is an efficient way of searching threads and associated replies on Reddit, which is the data source of our paper.

### Model
The pre_trained models for reply and main event streams can be found in `\pre_trained` folder

## Cite
```
@article{ling2020socialgrid,
  title={SocialGrid: A TCN-enhanced Method for Online Discussion Forecasting},
  author={Ling, Chen and Wang, Ruiqi and Tong, Guangmo},
  journal={arXiv preprint arXiv:2003.07189},
  year={2020}
}
```
