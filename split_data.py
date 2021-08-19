# split data to train and dev
import os, shutil
import random
# 90%訓練 10%驗證
train_ratio = 0.9
train_num = int(round(853 * train_ratio, 0)) #總資料數目以2
# prepare train folder
images_list = []
for img in os.listdir("./training_data/to_yolo_format"):
    if img.endswith(".jpg"):
        images_list.append(img.split(".")[0])
        
random.shuffle(images_list)
yolo_format_folder = os.path.join("./training_data/to_yolo_format")
train_folder = os.path.join("./train")
dev_folder = os.path.join("./dev")
if not os.path.isdir("./train"):
    os.mkdir(train_folder)    
if not os.path.isdir("./dev"):
    os.mkdir(dev_folder)
# train data 
for train_data in images_list[:train_num]:
    shutil.copyfile(os.path.join(yolo_format_folder, "{}.jpg".format(train_data)),  
                    os.path.join(train_folder, "{}.jpg".format(train_data)))
    shutil.copyfile(os.path.join(yolo_format_folder, "{}.txt".format(train_data)),  
                    os.path.join(train_folder, "{}.txt".format(train_data)))
   
# dev data
for test_data in images_list[train_num+1:]:
    shutil.copyfile(os.path.join(yolo_format_folder, "{}.jpg".format(test_data)),  
                    os.path.join(dev_folder, "{}.jpg".format(test_data)))
    shutil.copyfile(os.path.join(yolo_format_folder, "{}.txt".format(test_data)),  
                    os.path.join(dev_folder, "{}.txt".format(test_data)))
# show total data 
print("="*35)
print("number of training set :", len(os.listdir(train_folder)))
print("number of dev set :", len(os.listdir(dev_folder)))
print("="*35)