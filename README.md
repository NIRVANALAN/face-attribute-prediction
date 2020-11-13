# face-attribute-prediction
Face Attribute Prediction on [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) benchmark with PyTorch Implemantation.

## Dependencies

* Anaconda3 (Python 3.6+, with Numpy etc.)
* PyTorch 0.4+
* tensorboard, tensorboardX

## Dataset

[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset is a large-scale face dataset with attribute-based annotations. Cropped and aligned face regions are utilized as the training source. For the pre-processed data and specific split, please feel free to contact me: <d-li14@mails.tsighua.edu.cn>
[lfwA](http://vis-www.cs.umass.edu/lfw/) dataset is the private test dataset.

## Features

* Both ResNet and MobileNet as the backbone for scalability
* Each of the 40 annotated attributes predicted with multi-head networks
* Achieve ~92% average accuracy, comparative to state-of-the-art
* Fast convergence (5~10 epochs) through finetuning the ImageNet pre-trained models

## Result

| Method           | CelebA Acc | LFWA ACC |
| ---------------- | ---------- | -------- |
| Baseline         | 91.3       | 72.2     |
| Cosine decay     | 91.3       | 72.36    |
| Balance Sampling |
| Sqrt PosWeight   | 91.2       | 72.2     |
| Focal            |


1. input - mean
2. celebA liangdianduiqi lfw EYE   
3. test 70 112 108 112
<!-- 4. Homorgraphy ?  -->
5. decoupling. balanced 2-stage.
<!-- 6. translate DA      -->
7. weights on attributes. 