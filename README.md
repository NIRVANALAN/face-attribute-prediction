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

* BCE loss for attributes recognition
* Good capacity as well as generalization ability.Achieve 92%+ average accuracy on CelebA Val as well as >73% on LFWA+.
* ResNet-50 as backbone
* Focal Loss
* Class balanced sampler 
* fast convergence: 91% acc on CelebA Val after 1 epoch.

## Result

| Method                 | CelebA Acc | LFWA ACC |
| ---------------------- | ---------- | -------- |
| BCE Baseline           | 91.3       | 72.2     |
| Cos Decay + US + focal | 92.14      | 73.43    |