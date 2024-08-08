import os
import numpy as np
import cv2

# Data Preprocessing
def load_data(data_dir):
    data = []
    labels = []
    for category in ['PNEUMONIA', 'NORMAL']:
        path = os.path.join(data_dir, category)
        class_num = 0 if category == 'NORMAL' else 1
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_arr = cv2.imread(img_path)
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)  # Convert to RGB
            resized_arr = cv2.resize(img_arr, (150, 150))
            data.append(resized_arr)
            labels.append(class_num)
    return np.array(data), np.array(labels)

def normalize_data(data):
    return data / 255.0
