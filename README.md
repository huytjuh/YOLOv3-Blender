# **YOLOv3 Blender**
Person-detection model based on YOLOv3 using 3D synthetic images from Blender. 

A Keras implementation of YOLOv3 (Tensorflow backend) inspired by [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3).

## Introduction
Following up on Philips’ recent development in Remote Patient Monitoring and synthetic data, this algorithm is a continuation of the existing vision-based patient monitoring framework, aiming to improve the current object detection model using Deep Learning and synthetic data. That is, integrating patient monitoring with the state-of-the-art YOLOv3 object detection model is enormously promising as it provides a comprehensive knowledge base that accelerates the clinical response-ability, effectively allowing us to respond to emerging events or fatal incidents. Additionally, by embedding Domain Adaptation in the pipeline of Deep Learning, the objective is to exploit synthetic data allowing us to address and circumvent the strict privacy constraints and lack of available data in the healthcare industry. 

Hence, in the virtue of confidentiality restrictions and ethical claims, there is an increasing incentive to use less privacy-intrusive methods and opt for data that do not directly refer to identifiable persons. Therefore, we constructed a privacy-preserving object detection model using YOLOv3 based on synthetic data. Especially for AI and Machine Learning practices that require enormous amount of data to be effective, innovative solutions to preserve privacy are necessary to alleviate the scarcity of health care data available. The disintermediation of what was previously considered as a long-standing goal for many hospitals, is now slowly becoming more of a realization with the introduction of AI-powered patient monitoring, allowing healthcare providers to continue monitor their patients remotely in real-time without excessively relying on confidential data; all contributing to the paradigm shift of healthcare.

---

## Quick Start
---
### 1.1 BLENDER 2.92 CUSTOM-BUILT 
Folder: *Huy/Blender Custom/*

- Enabled rendering of viewer nodes and previews in background mode.
(*source:* https://blender.stackexchange.com/questions/69230/python-render-script-different-outcome-when-run-in-background/81240#81240)
https://wiki.blender.org/wiki/Building_Blender/Windows
- For more information see the Blender documentation in the Appendix attached.

### 1.2 PYTHON IN BLENDER 
Folder: *Huy/Data/*

- 3D images are constructed, assembled, and rendered using the Blender.py script. Comments and documentations are included.
- Adjust and change certain parameters and inputs in the Blender.py to create the synthetic data accordingly.
- For more information on the custom Python API *BPY* module see: https://docs.blender.org/api/current/index.html

### 1.3 HPC WINDOWS CLUSTER 
Folder: *Huy/*

- Open Microsoft HPC Job Manager > New Job from XML file > /Huy/HPC_cluster_1.xml > Resource Selection > Select one of the available nodes.
- In this case, we used four nodes to parallelize the rendering of 3D synthetic images. Each HPC_cluster_x.xml for x in {1,2,3,4} represents a batch.
- For more information see the HPC Windows Cluster documentation by Jimenez Jose:
https://confluence.atlas.philips.com/display/CMON/How-to+run+applications+on+the+Windows+HPC+cluster


## 2. EXTRACTING IMG FROM GOOGLE OPENIMAGES DATABASE V6
---
### 2.1 FILTER, DOWNLOAD, AND CREATE ANNOTATION FILES USING LINUX CLUSTER 
Folder: *Huy/Data/OpenImages/*

- Non-person IMG are filtered and downloaded using the openimages.py script (incl. documentations and comments).
-- Configure settings and initialize paths in [cfg.txt]()
-- Run [run_openimages.sh]() shell command in Linux to execute the required Python environment and python scripts.
-- Respective annotations are in the [<imgfile>.txt]() file with format `imagefile_path x_min,y_min,x_max,y_max,class_id` 
*(source: https://github.com/qqwweee/keras-yolo3)*
- For more information on Google OpenImages Database see: https://storage.googleapis.com/openimages/web/download.html


## 3. CREATING ANNOTATIONS FILE FROM DATA REQUIRED FOR YOLOV3
---
### 3.1 SELECT & COMBINE ANNOTATION FILES INTO A SINGLE TXT FILE FOR YOLOV3
Folder: *Huy/Data/Annotations/*

- Each folder contains a [Annotation_<model>.py]() script that navigates to the IMG folder and combine the selected sample annotations
-- Configure settings and initialize paths in [cfg.txt]()
-- Run [run_annot.sh]() shell command in Linux to execute the required Python environment and python scripts.
-- The resulting combined annotations txt file can be found in the same folder under the name [Full_Train_Annotations.txt]() or [Model>_Annotations.txt]()

## 4. RUN YOLOV3 PERSON-DETECTION MODEL
---
### 4.1 INSTALLING LIBRARIES & DEPENDENCIES
Folder: *Huy/YOLOv3/keras-yolo3-master*
- All required Python environment modules for each Python script are specified in the [<script>.sh]() command. For simplicity, copy & paste the virtual environment to your respective folder.
- Adjust the model framework in the master folder, if needed.
-- ***Note.*** The original YOLOv3 version is based on *Tensorflow 1.x*, however Philips' Linux GPU not compatible with this version.
-- Hence, the current YOLOv3 master folder is updated accordingly to the latest *Tensorflow 2.x* and *Keras 2.x* version.
- The YOLOv3 environment consists of standard libraries, if you want to install it again from scratch:
-- First, conda install tensorflow-gpu using the Philips `cadenv` environment: this automatically includes the compatible *CUDA* and *cuDNN* installations. For more information see the Linux Servers & GPU Cluster documentation in the Appendix attached.
-- Then use `Pip` module to install the remaining libraries by trial & error and update the libraries accordingly.

### 4.2 TRAINING YOLOv3 MODELS
Folder: *Huy/YOLOv3/Main/Model_<name>*

- Specify the following three folders in your *Main/Model_<name>* folder required to train YOLOv3 model
-- *Model_<name>/Model* : [synthetic_classes.txt]() (default classses), and [yolo_anchors.txt]() (default anchors)
-- *Model_<name>/Train* : [DarkNet53.h5]() (default initial weight), [<model>_Annotations.txt]() (Final annotation file compiled in Section 3.1)
-- *Model_<name>/linux_logs* : Output [train.txt]() logfile includes training process and errors if there are any
- Each *Model_<name>* folder contains a [train.py]() script that performs the YOLOv3 training process 
-- Specify learning parameters and epochs in [train.py](). Default are: 
Initial Stage (Freeze first 50 layers): `Adam(lr=1e-2), Batch_size=8, Epochs=10`
Main Process (Unfreeze all layers): `Adam(lr=1e-3), Batch_size=8, Epochs=100`
-- Configure settings and initialize paths in [cfg.txt]()
-- Run [run_train.sh]() shell command in Linux to execute the required Python environment and python scripts.
-- After every 3 epochs (default) the models are saved in *Model_<name>/Train/logs/001* (default). When training multiple models, make sure that you change the `logs_path` in the [cfg.txt]() file.
- Recalculate anchor boxes using [kmeans.py]() script (OPTIONAL)
-- Configure settings and initialize paths in [cfg.txt]()
-- Run [run_kmeans.sh]() shell command in Linux to execute the required Python environment and python scripts.
-- Resulting anchor boxes can be found in *Model_<name>/Model* named as [yolo_anchors.txt]() (default)
- ***Note.*** Different Python environment are used for training (Section 4.2) and testing (Section 4.3). The compatible virtual environment for training can be found in [run_train.sh]() shell command.

### 4.3 TESTING YOLOV3 MODELS ON PHILIPS' LAB DATA
Folder: *Huy/YOLOv3/Main/Model_<name>*

- Specify the following three folders in your *Main/Model_<name>* folder required to test YOLOv3 model
-- *Model_<name>/Model* : [<model_name>.h5](), [synthetic_classes.txt]() (default classses), and [yolo_anchors.txt]() (default anchors).
*(If you haven't specified it already during Training Section 4.2)*
-- *Model_<name>/Output* : The resulting test images with bounding boxes will be output here, incl. [confidence_scores.csv]().
-- *Model_<name>/Evaluation* : The resulting performance score will be output here in [Model_<name>_evaluation.csv]().
- Each *Model_<name>* folder contains a [yolo_video.py]() script that performs the YOLOv3 bounding box predictions 
-- Configure settings and initialize paths in [cfg.txt]()
Specify trained model (from Section 4.2) here.
***Note.*** Correctly name `model_name` as it will be used later on to make graphs and overall benchmark evaluation. 
-- Run [run_yolov3.sh]() shell command in Linux to execute the required Python environment and python scripts.
-- The bounding box predictions are saved in *Model_<name>/Output* and performance score in *Model_<name>/Evaluation*.
- Additionally [run_yolov3.sh]() automatically execute the [evaluation.py]() script to evaluate the bounding boxes based on various metrics.
-- Change or add performance metrics in the script, default is `overlap_threshold=0.5` (IOU). 
-- Comments and documentations are included. 
- ***Note.*** Different Python environment are used for training (Section 4.2) and testing (Section 4.3). The compatible virtual environment for both testing and evaluation can be found in [run_yolov3.sh]() shell command.

### 4.4 FINE-TUNE YOLOv3 MODELS ON PHILIPS' TRAIN LAB DATA (OPTIONAL)

Folder: *Huy/YOLOv3/Main/Model_<name>_Lab*

- Specify the following three folders in your *Main/Model_<name>_Lab* folder required to fine-tune YOLOv3 model
-- *Model_<name>_Lab/Model* : [synthetic_classes.txt]() (default classses), and [yolo_anchors.txt]() (default anchors)
-- *Model_<name>_Lab/Train* : [<model_name>.h5](), [<model>_Annotations.txt]() (Final annotation file compiled in Section 3.1)
-- *Model_<name>_Lab/linux_logs* : Output [train.txt]() logfile includes training process and errors if there are any.
- Each *Model_<name>_Lab* folder contains a [train.py]() script that performs the YOLOv3 training process 
-- Specify learning parameters and epochs in [train.py](). Default are: 
Initial Stage (Freeze first 50 layers): `Adam(lr=1e-4), Batch_size=16, Epochs=10`
Main Process (Unfreeze all layers): `Adam(lr=1e-5), Batch_size=16, Epochs=50`
-- Configure settings and initialize paths in [cfg.txt]()
-- Run [run_train.sh]() shell command in Linux to execute the required Python environment and python scripts.
-- After every 3 epochs (default) the models are saved in *Model_<name>_Lab/Train/logs/001* (default). When training multiple models, make sure that you change the `logs_path` in the [cfg.txt]() file.
- ***Note.*** Different Python environment are used for training (Section 4.2) and testing (Section 4.3). The compatible virtual environment for training can be found in [run_train.sh]() shell command.

## 5. RESULTS YOLOv3 MODELS BASED ON F1 AND AP-SCORES
---
### 5.1 OBTAIN PRECISION-RECALL (PR) CURVE AND HIGHEST F1-SCORES 
Folder: *Huy/YOLOv3/Main*

- Iterate through all  *Main/Model_<name>/Evaluation* folders and combine and calculate all performance score in [scores_all.py]() script.
-- Outputs the [PR_all.xlsx]() in *Main/Visualization* required to calculate and plot the PR-curves.
-- ***Note.*** To omit certain models from the PR-curve or overall evaluation, change the folder name accordingly, i.e. *Main/Model_<name>/Evaluation_DEACTIVATE*.
(The scripts loops through all folders containing *Main/Model_{}/Evaluation* with folder name `/Evaluation` hard coded)
- Run [run_scores.sh]() shell command in Linux to execute the required Python environment and python scripts.
-- It automatically executes the [Main/Visualization/create_graphs.py]() script ti plot the PR-curves and other graphs with `IOU=0.5` (default).
-- Documentation and comments follows from the shell and python script.

### 5.2 CASE-BY-CASE AP-SCORE EVALUATION (OPTIONAL)
Folder: *Huy/YOLOv3/Main/Results_IMGLabels*

- Case-by-case evaluations are executed by calculating the AP-scores for each case in Philips' Lab test set in [scores_IMGLabels.py]() script.
-- In [Main/ArgusImages_test_IMGLabels.csv]() we find the features for each test image, incl. *RGB/IR/Occl./Male/Female* etc.
-- Include [scores_IMGLabels.py]() at the end of [Main/run_scores.sh]() shell command in Linux to execute the required Python environment and python scripts (default).
-- The resulting case-by-case evaluation scores can be found in [Evaluation_IMGlabels_case.xlsx]() with each tab corresponds to the feature being kept fixed.

## 6. RUN CYCLEGANS ON SYNTHETIC IMAGES
---
### 6.1 RUN CYCLEGAN USING PYTORCH
Folder: *Huy/Domain_Adaptation/CycleGAN*

- Train CycleGANs model using the [train.py]() script. Documentation and comments are included.
-- Create your datasets map by using the [<name_A>2<name_B>.py]() script, which creates *trainA, trainB, testA, testB* folders from the annotation inputs specified in [cfg.txt](). You can run [run_dataset.sh]() shell command in Linux to execute the required Python environment and python scripts.
-- Specify your parameters in [base_option.py]() and [train_option.py]() in the folder *CycleGAN/options*
-- Run [train_cyclegan.sh]() shell command in Linux to execute the required Python environment and python scripts.
(Input parameters are included and follows from the shell script.)
-- The models can be found in *CycleGAN/results/<name>* folder, including logfiles.
- Run CycleGANs model using the [test.py]() script. 
-- Use the [latest_net_G_A.pth]() model as input.
-- Specify your parameters in [base_option.py]() and [test_option.py]() in the folder *CycleGAN/options*
-- Run [run_cyclegan.sh]() shell command in Linux to execute the required Python environment and python scripts.
Additional parameters are included and follows from the shell script.

> Source: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

### 6.2 RUN BAYESIAN CYCLEGANS
Folder: *Huy/Domain_Adaptation/CycleGAN_Bayesian*

- Train Bayesian CycleGANs model using the [train_bayes.py]() script. Documentation and comments are included.
-- Create your datasets map by using the [<name_A>2<name_B>.py]() script, which creates *trainA, trainB, testA, testB* folders from the annotation inputs specified in [cfg.txt](). You can run [run_dataset.sh]() shell command in Linux to execute the required Python environment and python scripts.
-- Specify your parameters in [base_option.py]() and [train_option.py]() in the folder *CycleGAN_Bayesian/options*
-- Run [train_cyclegan_bayes.sh]() shell command in Linux to execute the required Python environment and python scripts.
Input parameters are included and follows from the shell script.
-- The models can be found in *CycleGAN_Bayesian/results/<name>* folder, including logfiles..
- Run CycleGANs model using the [test.py]() script. 
-- Use the [latest_net_G_A.pth]() model as input.
-- Run [run_cyclegan_bayes.sh]() shell command in Linux to execute the required Python environment and python scripts.
Additional parameters are included and follows from the shell script.

> Source: https://github.com/ranery/Bayesian-CycleGAN
