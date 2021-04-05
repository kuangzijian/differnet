# DifferNet
This project is used for experiment to train and test models on variouse datasets. The core function has been packaged as "[differnet-zerobox](https://github.com/zerobox-ai/pydiffernet)". Pleaser refer to the [readme](https://github.com/zerobox-ai/pydiffernet/blob/master/README.md) for how to use the package.

If you need more information please reference to the official repository. 

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
python run_training.py
```

Validate model
```
python run_test.py
```

## Data
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
### How to use Data extraction tool to extract data from video clips:
 1. Create folder structure like the example shows in the picture below.
 
  ![1](https://github.com/zerobox-ai/differnet/blob/zijian/dataset/data-generation/annotations/structure1.png)
  
 2. Dump the videos and annotations (rename them use 1.xml, 1.avi as one pair annotation and video) into the folders under data-generation folder.
 
  ![2](https://github.com/zerobox-ai/differnet/blob/zijian/dataset/data-generation/annotations/structure2.png)
  
 3. Modify the annotation files: Since the annotation uses label "defect" to indicate the defect area, while, both good and defective bottles are labeled as "bottle" which is confusing. To indicate which "bottle" is defective, we need to find the frames that labeled with defect, and then manully update the group's label from "bottle" to "defective" for the groups that falling in to those frames. 
  ![3](https://github.com/zerobox-ai/differnet/blob/zijian/dataset/data-generation/annotations/structure3.png)
  
 - For example: in the example image above, the frame 15 and 16 are labeled as "defect" which indicates those 2 frames has defect areas on the bottles. So we need to find the group that contains frame 15 and 16, and then manully update the label from "bottle" to "defective". and then delete the whole \<track\> group that labeled as "defect" (since we don't care about the defect area in data extraction).
 
 4. Modify the config.py, fill in appropriate value for num_videos, save_cropped_image_to and save_original_image_to
 
 5. run the data extraction: python data_extraction.py


The given dummy dataset shows how the implementation expects the construction of a dataset. Coincidentally, the [MVTec AD dataset](https://www.mvtec.com/de/unternehmen/forschung/datasets/mvtec-ad/) is constructed in this way.

Set the variables _dataset_path_ and _class_name_ in _config.py_ to run experiments on a dataset of your choice. The expected structure of the data is as follows:

``` 
train data:

        dataset_path/class_name/train/good/any_filename.png
        dataset_path/class_name/train/good/another_filename.tif
        dataset_path/class_name/train/good/xyz.png
        [...]

test data:

    'normal data' = non-anomalies

        dataset_path/class_name/test/good/name_the_file_as_you_like_as_long_as_there_is_an_image_extension.webp
        dataset_path/class_name/test/good/did_you_know_the_image_extension_webp?.png
        dataset_path/class_name/test/good/did_you_know_that_filenames_may_contain_question_marks????.png
        dataset_path/class_name/test/good/dont_know_how_it_is_with_windows.png
        dataset_path/class_name/test/good/just_dont_use_windows_for_this.png
        [...]

    anomalies - assume there are anomaly classes 'crack' and 'curved'

        dataset_path/class_name/test/crack/dat_crack_damn.png
        dataset_path/class_name/test/crack/let_it_crack.png
        dataset_path/class_name/test/crack/writing_docs_is_fun.png
        [...]

        dataset_path/class_name/test/curved/wont_make_a_difference_if_you_put_all_anomalies_in_one_class.png
        dataset_path/class_name/test/curved/but_this_code_is_practicable_for_the_mvtec_dataset.png
        [...]
``` 
## License

This project is licensed under the MIT License.

 
