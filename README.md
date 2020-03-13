# Deep-Light-Field-Driven-Saliency-Detection-from-A-Single-View
## Introduction
Accepted paper in CVPR2020, 'A2dele: Adaptive and Attentive Depth Distiller for Efficient RGB-D Salient Object Detection', [Yongri Piao](http://ice.dlut.edu.cn/yrpiao/), Zhengkun Rong, Miao Zhang, Weisong Ren and [Huchuan Lu](http://ice.dlut.edu.cn/lu/publications.html).

## Usage Instructions
Requirements
* Windows 10
* PyTorch 0.4.1
* CUDA 9.0
* Cudnn 7.6.0
* Python 3.6.5
* Numpy 1.16.4

## Training and Testing Datasets
Training dataset
* [Download Link](https://pan.baidu.com/s/1cm5nkdKVHU2vCIqgzlmLMw). Code: eu72

Testing dataset
* [Download Link](https://pan.baidu.com/s/1cm5nkdKVHU2vCIqgzlmLMw). Code: eu72

## Depth Stream
Training
* Modify your path of training dataset in train_depth
* Run train_depth

Testing
* Download pretrained depth model from [here](https://pan.baidu.com/s/1cm5nkdKVHU2vCIqgzlmLMw). Code: eu72
* Modify your path of testing dataset in test_depth
* Run test_depth to inference saliency maps
* Saliency maps generated from the depth stream can be downnloaded from [here](https://pan.baidu.com/s/1cm5nkdKVHU2vCIqgzlmLMw). Code: eu72

## RGB Stream
Training
* Modify your path of training dataset in train_RGB
* Modify the pretrained depth model path
* Run train_RGB

Testing
* Download pretrained RGB model from [here](https://pan.baidu.com/s/1cm5nkdKVHU2vCIqgzlmLMw). Code: eu72
* Modify your path of testing dataset in test_depth
* Run test_RGB to inference saliency maps
* Saliency maps generated from the RGB stream can be downnloaded from [here](https://pan.baidu.com/s/1cm5nkdKVHU2vCIqgzlmLMw). Code: eu72

## Contact and Questions
Contact: Zhengkun Rong. Email: 18642840242@163.com or rzk911113@mail.dlut.edu.cn
