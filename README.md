# R2Plus1D-mxnet

R2Plus1D MXNet Implementation

R2Plus1D: [A Closer Look at Spatiotemporal Convolutions for Action Recognition (CVPR 2018)](https://arxiv.org/pdf/1711.11248.pdf)

Caffe2 Implementation: https://github.com/facebookresearch/R2Plus1D

## Dataset
[UCF101](http://crcv.ucf.edu/data/UCF101.php)


## Result

Achieved 92.6% Accuracy(Clip@1, prediction using only 1 clip) on UCF101 Dataset, which is **higher than the original Caffe2 model**(Accuracy 91.3%).

## Usage

#### Requirements
 
 * MXNet with GPU support
 * opencv

### Data Preparation

 * Download and extract [UCF101](http://crcv.ucf.edu/data/UCF101.php) dataset to ~/UCF101
 * Download pre-trained model from [Caffe2 Pre-trained model](https://github.com/facebookresearch/R2Plus1D/blob/master/tutorials/models.md) to ~/r2.5d_d34_l32.pkl
  
 
#### Training 
 ```
$  python train.py --gpus 0,1,2,3,4,5,6,7 --pretrained ~/r2.5d_d34_l32.pkl --output ~/r2plus1d_output --batch_per_device 4 --lr 1e-4 
--model_depth 34 --wd 0.005 --num_class 101 --num_epoch 80 
```

#### Testing

Assume the training output directory is ~/r2plus1d_output and the epoch number we want to test is 80.

```
$ python validation.py --gpus 0 --output ~/r2plus1d_output --eval_epoch 80 --batch_per_device 48 --model_prefix test 
```




