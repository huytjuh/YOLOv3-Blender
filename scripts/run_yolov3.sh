#!/usr/bin/env bash
cd /home/2014-0353_generaleye/Huy/YOLOv3
source venv/my_env/bin/activate
cd Main/Model_COCO_Lab
python3 yolo_video.py --image
python3 evaluation.py