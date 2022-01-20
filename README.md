# **YOLOv3 Blender**
Person-detection model based on YOLOv3 using 3D synthetic images from Blender. 

A Keras implementation of YOLOv3 (Tensorflow backend) inspired by [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3).

![](https://learn.alwaysai.co/hubfs/Screen%20Shot%202020-01-23%20at%202.28.16%20PM.png)

---

## Introduction
Following up on Philips’ recent development in Remote Patient Monitoring and synthetic data, this algorithm is a continuation of the existing vision-based patient monitoring framework, aiming to improve the current object detection model using Deep Learning and synthetic data. That is, integrating patient monitoring with the state-of-the-art YOLOv3 object detection model is enormously promising as it provides a comprehensive knowledge base that accelerates the clinical response-ability, effectively allowing us to respond to emerging events or fatal incidents. Additionally, by embedding Domain Adaptation in the pipeline of Deep Learning, the objective is to exploit synthetic data allowing us to address and circumvent the strict privacy constraints and lack of available data in the healthcare industry. 

Hence, in the virtue of confidentiality restrictions and ethical claims, there is an increasing incentive to use less privacy-intrusive methods and opt for data that do not directly refer to identifiable persons. Therefore, we constructed a privacy-preserving object detection model using YOLOv3 based on synthetic data. Especially for AI and Machine Learning practices that require enormous amount of data to be effective, innovative solutions to preserve privacy are necessary to alleviate the scarcity of health care data available. The disintermediation of what was previously considered as a long-standing goal for many hospitals, is now slowly becoming more of a realization with the introduction of AI-powered patient monitoring, allowing healthcare providers to continue monitor their patients remotely in real-time without excessively relying on confidential data; all contributing to the paradigm shift of healthcare.

## Prerequisites

* python 3.8
* tensorflow 2.4
* Keras 2.1.4
* NVIDIA GPU + CUDA CuDNN
* Blender 2.92 (custom-built)

## Getting Started

### Installation Blender 2.92 Custom-Built
* Enable rendering of viewer nodes in background mode to properly update the pixels within Blender background.
* Open `source/blender/compositor/operations/COM_ViewerOperation.h` and change lines:
```
bool isOutputOperation(bool /*rendering*/) const { 
if (G.background) return false; return isActiveViewerOutput();
```
into:
```
bool isOutputOperation(bool /*rendering*/) const {return isActiveViewerOutput(); }
```
* Open `source/blender/compositor/operations/COM_PreviewOperation.h` and change line:
```
bool isOutputOperation(bool /*rendering*/) const { return !G.background; }
```
into:
```
bool isOutputOperation(bool /*rendering*/) const { return true; }
```

### Create Synthetic Images in Blender + Annotations
* Render 3D person images.
```
#!./scripts/run_blender.sh
"Blender Custom/blender.exe" --background --python "Data/Blender.py" -- 1
```
* Annotation files are saved in the respective `.txt` file with the same name and has the following format:
```
image_file_path min_x,min_y,max_x,max_y,class_id min_x,min_y,max_x,max_y,class_id ...
```

### Run YOLOv3 Blender synthetic model
* Run Trained Blender Synthetic Model.
```
#!./scripts/run_yolov3.sh
python3 scripts/yolo_video.py --image
python3 scripts/evaluation.py
```
* The bounding box predictions are saved in folder `output`.
* Performance scores and evaluation metrics are saved in `Evaluation` (Default is `overlap_threshold=0.5`).

## Custom Datasets for YOLOv3 Blender Training

### YOLOv3 Blender Training 
* Select & combine annotation files into a single `.txt` file as input for YOLOv3 training. Edit `Annotations/cfg.txt` accordingly.
```
!./scripts/run_annotations.sh
python3 Annotations/Annotation_synthetic2.py
```
* Specify the following three folders in your `Main/Model_<name>` folder required to train YOLOv3 model:
  * `Model_<name>/Model`: `synthetic_classes.txt` (class_id file) and `yolo_anchors.txt` (default anchors).
  * `Model_<name>/Train`: `DarkNet53.h5` (default .h5 weight) and `Model_Annotations.txt` (final annotation `.txt` file).
  * `Model_<name>/linux_logs`: Saves a `train.txt` logfile and includes training process and errors if there are any.
* Specify learning parameters and number of epochs in `train.py`. Defaults are:
  * Initial Stage (Freeze first 50 layers): `Adam(lr=1e-2)`, `Batch_size=8`, `Epochs=10`
  * Main Process (Unfreeze all layers): `Adam(lr=1e-3)`, `Batch_size=8`, `Epochs=100`
* Recompile anchor boxes using `kmeans.py` script (OPTIONAL)
* Configure settings and initialize paths in `Model_<name>/cfg.txt`
* Train YOLOv3 model.
```
!./scripts/run_scores.sh
python3 scores_all.py
python3 Visualizations/create_graphs.py
python3 Results_IMGLabels/scores_IMGLabels.py
```

### Benchmark & Evaluate All YOLOv3 Trained Models
* Obtain Precision-Recall (PR) curve and highest F1-scores by iterating through all `Main/Model_<name>/Evaluation` folders and calculate & combine all performance scores.
```
!./scripts/run_train.sh
python3 train.py >Main/Model_Synth_Lab/linux_logs/train.log
```
* Case-by-case AP-score Evaluation using `Main/scores_IMGLabels.py` (OPTIONAL)
  * Resulting case-by-case evaluation score can be found in `Main/Evaluation_IMGlabels-case.xlsx` with each tab corresponding to a feature kept fixed.

## Extracting RGB Images from Google OpenImages Database v6
Google’s OpenImages Database v6 dataset is used to collect negative non-person samples by extracting pre-annotated images that includes all kinds of objects and environments but without containing instances of persons.
* Non-person images are filtered and downloaded.
```
!./scripts/run_openimages.sh
python3 OpenImages.py > OpenImages/openimages.log
```
* Configure settings and initialize paths in `OpenImages/cfg.txt`.
* Annotation files are saved in the respective `.txt` file with the same name and has the following format:
```
image_file_path min_x,min_y,max_x,max_y,class_id min_x,min_y,max_x,max_y,class_id ...
```

> Source: https://storage.googleapis.com/openimages/web/download.html

## Acknowledgements

Code is inspired by [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3).
