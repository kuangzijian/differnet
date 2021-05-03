# Computer-Vision-and-Normalizing-Flow-Based-Defect-Detection

Source code for the following paper([arXiv link](https://arxiv.org/pdf/2012.06737v1.pdf)):

        Computer Vision and Normalizing Flow Based Defect Detection
        Zijian Kuang, Xinran Tie

This is the repository to the paper "Computer Vision and Normalizing Flow Based Defect Detection" by Zijian Kuang and Xinran Tie.
![0](https://github.com/kuangzijian/Computer-Vision-and-Normalizing-Flow-Based-Defect-Detection/blob/master/Model_Architecture_Overview.png)


# DifferNet
This project is used for experiment to train and test models on various datasets. The core function has been packaged as "[differnet-zerobox](https://github.com/zerobox-ai/pydiffernet)". Pleaser refer to the [readme](https://github.com/zerobox-ai/pydiffernet/blob/master/README.md) for how to use the package.

If you need more information about DifferNet, please reference to the official repository. 

**Differnet Officical repository**
The [official repository](https://github.com/marco-rudolph/differnet) to the WACV 2021 paper "[Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows](
https://arxiv.org/abs/2008.12577)" by Marco Rudolph, Bastian Wandt and Bodo Rosenhahn.

## Getting Started

The project has been upgraded to python 3.9. Please setup python 3.9 virtual environment then do the following.

### Install torch and torch vision
In order to have proper torch and torch vision to use either GPU or CPU please follow [pytorch.org](https://pytorch.org/get-started/locally/) to install torch and torch vision

## Install rest packages with:

```
$ pip install -r requirements.txt
```

## Configure and Run

All configuration has default values from package differnet(from package differnet-zerobox).
The project can have dict based configuration to overwrite any default value.


Common settings
```
#conf/settings.conf
"differnet_work_dir": "./work", #work folder
"device": "cuda",  # cuda or cpu
"device_id": 0,  # the device you want to use. depends on how many GPU or CPU you have. 
"verbose": True, # Set to true, when you do experiments.
"meta_epochs": 10,  # traing loop
"sub_epochs": 8,  # sub-loop of traing
"test_anormaly_target": 10, # threshold when run testing model to identify if a given image is good or bad

```

Traing

```
python training.py
```

Run test cases
```
python  -m pytest -s
```

## Prepare new dataset
The data structure under work folder looks like this. The model folder will save trained model.
For experiment purpose, you would like to give test and validate folder with proper labled data. While, for zerobox 
it only requires train folder and data. The minimum images is 16 based on the differnet paper.

```
pink1/
├── model
├── test
│   ├── defect
│   └── good
├─── validate
│    ├── defect
│    └── good
└── train
    └── good
        ├── 01.jpg
        ├── 02.jpg
        ├── 03.jpg
        ├── 04.jpg
        ├── 05.jpg
        ├── 06.jpg
        ├── 07.jpg
        ├── 08.jpg
        ├── 09.jpg
        ├── 10.jpg
        ├── 11.jpg
        ├── 12.jpg
        ├── 13.jpg
        ├── 14.jpg
        ├── 15.jpg
        └── 16.jpg
```
 
