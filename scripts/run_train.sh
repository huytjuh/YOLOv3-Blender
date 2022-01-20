#!/usr/bin/env bash
cd /home/2014-0353_generaleye/Huy/YOLOv3
source activate venv/conda_env_test
cd Main/Model_COCO_Lab
python3 train.py >/home/2014-0353_generaleye/Huy/YOLOv3/Main/Model_COCO_Lab/linux_logs/train_lre-4.log