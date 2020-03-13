# Deep-Light-Field-Driven-Saliency-Detection-from-A-Single-View
## Introduction
Accepted paper in IJCAI2019, 'Deep Light-Field-Driven Saliency Detection from A Single View', [Yongri Piao](http://ice.dlut.edu.cn/yrpiao/), Zhengkun Rong, Miao Zhang, Xiao Li and [Huchuan Lu](http://ice.dlut.edu.cn/lu/publications.html).

## Dataset: DUTLF
* This dataset consists of DUTLF-MV, DUTLF-FS and DUTLF-Depth.
* The dataset will be expanded to 3000 or so real scenes.
* We are working on it and will make it publicly available soon.

## Dataset: DUTLF-MV
* DUTLF-MV is part of DUTLF, which consists of 1580 real scenes.
* Each scene of this dataset consists of an all-focus image, multi-view images and a corresponding ground truth.
* Dataset can be downloaded from [here](https://pan.baidu.com/s/1hvrTL4PQp-PZ6QZEl5fH7Q). Code: 9c7k
* Training set: 1100 samples before data augmentation
* Testing set: 480 samples

## Usage Instructions
Requirements
* Windows 10
* Tensorflow 1.10.0
* CUDA 9.0
* Cudnn 9.0
* Python 3.6.5
* Numpy 1.14.3

Training
* Download pretrained vgg-19.npy from [here](https://pan.baidu.com/s/1U6J9XenDOnUvkEzj0ZBmxg). Code: yiov
* Hyperparameter: is_training=1
* Modify your path of training dataset
* Run Main_model
* cd 'your path'/logs, tensorboard --logdir=train

Testing
* Download pretrained model from [here](https://pan.baidu.com/s/1cm5nkdKVHU2vCIqgzlmLMw). Code: eu72
* Hyperparameter: is_training=0
* Modify your path of testing dataset
* Run Main_model to generate saliency maps, synthesized mutli-view images and depth maps
## Saliency map
Saliency maps of this paper can be downloaded [BaiduYun](https://pan.baidu.com/s/1KXG7xRv7WOcSj_NUmbz8cA). Code: 2jl0
## Contact and Questions
Contact: Zhengkun Rong. Email: 18642840242@163.com or rzk911113@mail.dlut.edu.cn
