# INITIALIZE BASE PATH (= MODEL FOLDER)
base_path = '/home/2014-0353_generaleye/Huy/YOLOv3/keras-yolo3-master/'
#base_path = '/home/2014-0353_generaleye/Huy/YOLOv3/test/keras-yolo3/'

# INITIALIZE TRAINING & TEST SAMPLE
img_path = '/home/2014-0353_generaleye/Huy/Data/Output_IMG/'
true_path = '/home/2014-0353_generaleye/Huy/Data/ArgusImages_test/'

# INITIALIZE CLASSES & ANCHORS
classes_path = 'Model/synthetic_classes.txt'
anchors_path = 'Model/yolo_anchors.txt'

# TRAINING MODEL
annot_path = 'Train/Lab_Pos_Annotations.txt'
logs_path = 'Train/logs/001_lre-4/'
input_weights = 'Train/logs/001_lre-4_2/ep093-loss13.874-val_loss13.859.h5'

# PERFORM YOLOV3 MODEL
model_name = 'Model_Synth_DA_Lab'
model_path = 'Train/logs/001_lre-4/ep051-loss13.417-val_loss13.565.h5'
output_path = 'Output/'