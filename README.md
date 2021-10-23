# Jetson-Nano-YoloV4-tiny
## 前言

![](https://i.imgur.com/LS5Hz44.png)


這個項目是使用 Nvidia Jetson Nano 開發板進行YoloV4 tiny 模型的訓練以及辨識。
本項目以口罩辨識為範例，若要訓練自己的資料請自行改變參數。

本項目所使用的資源:

1. XML 格式轉換成 TXT 格式
    https://github.com/Isabek/XmlToTxt.git
2. darknet 框架 
    https://github.com/AlexeyAB/darknet.git
3.  kaggle 口罩資料集
    https://www.kaggle.com/andrewmvd/face-mask-detection
 ---
## 編譯draknet

使用git clone 指令將本項目克隆到Jetson Nano 中:
```bash=
git clone https://github.com/wei900508/Jetson-Nano-YoloV4-tiny.git
```
:::info
若要使用自己的圖片
將圖片放進 training_data/images
將標記XML檔放進 training_data/annotations
:::

我已經將 Makefile 修改成適合 Jetson 平台運行的參數了
可以直接使用 `make` 指令編譯

```bash=
cd Jetson-Nano-YoloV4-tiny/darknet/
make
```

以下是說明我修改了那些內容:


```java=
GPU=1
CUDNN=1
CUDNN_HALF=1
OPENCV=1
AVX=0
OPENMP=1
LIBSO=1
ZED_CAMERA=0
ZED_CAMERA_v2_8=0

......

USE_CPP=0
DEBUG=0

ARCH= -gencode arch=compute_53,code=[sm_53,compute_53]

......

NVCC=/usr/local/cuda/bin/nvcc

```
:::info
ARCH的部分，需要將其修改成compute_53
:::

:::warning
nvcc 可能會抓不到，所以修改成絕對路徑
:::

## 訓練資料設定

1. **更改圖片格式** 

若圖片是PNG檔，可以使用以下程式
將圖片從PNG轉換成JPG

```bash=
python3 png2jpg.py
```

要更改路徑請使用下面的程式並填入路徑
```python=
from PIL import Image
import os
# 轉 jpg
for img in os.listdir("./training_data/images/"):
    if img.endswith(".png"):
        im = Image.open("./training_data/images/{}".format(img))
        rgb_im = im.convert("RGB")
        img_name = img.split(".")[0]
        rgb_im.save("./training_data/images/{}.jpg".format(img_name))
        
# 刪除原始 png 資料
for img in os.listdir("./training_data/images/"):
    if img.endswith(".png"):
        os.remove("./training_data/images/{}".format(img))
```
    

2. **複製一份影像資料**

把影像和標記檔放複製後放在同一個資料夾裡 (training_data/to_yolo_format)
指令如下：


```bash=
cd ./training_data/ ; cp ./annotations/*.xml ./to_yolo_format
cd ./training_data/ ; cp ./images/*.jpg ./to_yolo_format
```
在 to_yolo_format 資料夾裡添加一個 classes.txt 裡面放要辨識的標籤

```=
with_mask
without_mask
mask_weared_incorrect
```

利用他人撰寫好的轉換程式來轉換 XML -> YOLO 。
下完指令後資料夾會多出 .txt 檔的 YOLO 標註格式

```bash=
cd ./XmlToTxt
```

```bash=
pip3 install -r requirements.txt
```

```bash=
cd ./XmlToTxt
python3 xmltotxt.py -c ../training_data/to_yolo_format/classes.txt -xml ../training_data/to_yolo_format -out ../training_data/to_yolo_format
```


3. **準備訓練以及驗證資料集**

可以直接運行以下指令

```bash=
python3 split_data.py
```
若想使用自己的路徑則須修改Python檔
路徑要依據你實際存放的路徑做修改
```python=
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

```


到這邊應該會有 train 以及 dev 資料夾，裡面分別有對應的 jpg 圖片檔案以及 YOLO 影像標註檔案(.txt)，在訓練前我們需要把資料輸出成一個 file list (圖片的絕對路徑) 這樣跑訓練時才會讓 YOLO 知道你的檔案在哪裡，待會會用到這兩個檔案。此步驟操作如下：

```bash=
# 輸出 train.txt
cd ./train ; ls -d "$PWD"/*.jpg > train.txt
# 輸出 dev.txt
cd ./dev ; ls -d "$PWD"/*.jpg > dev.txt
```
---

## 訓練 YOLOv4 前準備

這邊需要修改4個檔案，分別是：
* names file
* data file
* cfg file

我已經將以上檔案針對Jetson Nano環境修改好並放在config資料夾內
以下是說明我修改了哪些地方

### 手動修改參數
1. names file
    存放要辨識的物件名稱，到 config中建立一個 mask.names，裡面資料為要辨識的物件類別
    
    ```
    with_mask
    without_mask
    mask_weared_incorrect
    ```
2. data file
    
    存放一些參數，物件類別數量、路徑 (前面步驟的 train.txt & dev.txt)，一樣到 config 建立一個 mask.data ，要記得修改成你的資料路徑!
    
    :::info
    train= (上一步驟的 train.txt 路徑)
    valid= (上一步驟的 dev.txt 路徑)
    names= (.names 檔案路徑)
    backup= (權重存放路徑)
    :::
    
    ```bash
    classes=3
    train=../train/train.txt
    valid=../dev/dev.txt
    names=../data/mask.names
    backup=backup/
    ```
    
3. cfg file
存放 yolo 的結構以及各種參數，先到 darknet/cfg 資料夾裡面複製一份 yolov4-tiny-custom.cfg 並改名為 yolov4-tiny-mask.cfg，複製好後裡面有幾個地方需要變更
* 變更 batch、subdivisions
    若已經是變更後的資料則不用變更
    ```c
    batch = 64
    subdivisions = 16
    ```
* 變更 max_batches 
    max_batches = clsss * 2000 (最少6000)
    我們有三個類別的標籤所以這邊 max_batches = 6000
    ```c
    max_batches = 6000
    ```
* 變更 steps
    steps = max_batches * 0.8, 0.9
    
    ```c
    steps = 4800, 5400
    ```
    
* 變更 width、height（需是32倍數）
    因Jetson Nano 性能，設定為160
    ```c
    width = 160
    height = 160
    ```
* 變更兩個 [yolo] 區塊的 classes 改成需辨識的類別
    ```c
    classes = 3
    ```
    
* 變更兩個 [convolution] 區塊的 filter 
    :::info
    yolov4 偵測的濾鏡(filter) 大小為 (C+5)*B
    -B 是每個Feature Map可以偵測的bndBox數量，這裡設定為3
    -5 是bndBox輸出的5個預測值: x,y,w,h 以及 Confidence
    -C 是類別數量
    filters=(classes + 5)*3  # 因為是三個類別，所以filters更改為 24
    :::
    ```c
    filters=24
    ```
* 修改預設 anchors 值，
    可以使用以下指令 (記得更改參數 cfg/face.data, num_of_clusters, width, height)
    是由 Darknet 官方寫好可以自動算出 anchors 值
    
    ```bash
    cd /darknet
    ./darknet detector calc_anchors ../config/mask.data -num_of_clusters 6 -width 160 -height 160 -showpause
    ```
    
    輸出:
    ```c=
    CUDA-version: 10020 (10020), cuDNN: 8.0.0, CUDNN_HALF=1, GPU count: 1
    CUDNN_HALF=1
    OpenCV version: 4.1.1

    num_of_clusters = 6, width = 160, height = 160
    read labels from 768 images
    loaded          image: 768      box: 3545
    all loaded.

    calculating k-means++ ...

    iterations = 54


    counters_per_class = 2795, 638, 112

    avg IoU = 73.14 %

    Saving anchors to the file: anchors.txt
    anchors =   3,  6,   6, 11,  10, 17,  16, 27,  28, 40,  54, 58
    ```
    
    將最下面的6組數字填入yolov4-tiny-mask.cfg
    
### 準備權重

我事先將 yolov4-tiny Darknet 官方事先訓練好的`yolov4-tiny.conv.29` 放入 config/weights 中可以直接訓練

若使用其他 cfg 檔可以從 darknet 官方 github 下載相對應的 weight
尋找其它權重 ➔ [🔎](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects)

---    
## 開始訓練&辨識

### 訓練指令
:::danger
注意路徑是否正確
:::

```bash=
cd /darknet
./darknet detector train ../config/mask.data ../config/yolov4-tiny-mask.cfg ../config/weights/yolov4-tiny.conv.29
```

:::info
什麼時候可以結束訓練？
darknet 文件提到訓練時 avg_loss 介於 0.05(小模型、簡易資料集) ~ 3.0(大模型、複雜資料集) 即可中止訓練，它會將權重存到 darknet/backup 資料夾 (每 1000 iteration 會自動存一次，中斷訓練前也會存最後一次)
:::

### 辨識指令
後方參數夾帶要辨識的圖片路徑即可
這邊我使用dev資料夾中的一張圖片示範
```bash=
cd darknet/ 
./darknet detector test ../config/mask.data ../config/yolov4-tiny-mask.cfg ./backup/yolov4-tiny-mask_last.weights ../dev/maksssksksss7.jpg
```

辨識結果

![](https://i.imgur.com/2nYxVJh.png)

## 使用openCV進行即時辨識
在config 資料夾中新增一個opencv.data
內容如下
```c=
classes=3
train=../train/train.txt
valid=../dev/dev.txt
names=./config/mask.names
backup=backup/
```
使用python 指令直接運行 `opencv.py` 即可啟動相機進行辨識
:::info
確認接上鏡頭後輸入`ls /dev/video*`
若有出現/dev/video0 則可以直接運`opencv.py`
:::

:::warning
輸入`ls /dev/video*`
若是出現其它數字請編輯`opencv.py`
將27行改成對應數字再運行
:::

開始辨識，按下q可以關閉相機
```bash=
python3 opencv.py
```

若有使用自己的路徑請修改對應的地方
```python=
import cv2
import darknet.darknet as darknet
import time


win_title = 'YOLOv4 CUSTOM DETECTOR'
cfg_file = 'config/yolov4-tiny-mask.cfg'
data_file = 'config/opencv.data'
weight_file = 'darknet/backup/yolov4-tiny-mask_last.weights'
thre = 0.25
show_coordinates = True

network, class_names, class_colors = darknet.load_network(
        cfg_file,
        data_file,
        weight_file,
        batch_size=1
    )

width = 720
height = 480

#width = darknet.network_width(network)
#height = darknet.network_height(network)


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    t_prev = time.time()
    
    frame_rgb = cv2.cvtColor( frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize( frame_rgb, (width, height))
    
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes()) 
    
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thre)
    darknet.print_detections(detections, show_coordinates)
    darknet.free_image(darknet_image)
    
    image = darknet.draw_boxes(detections, frame_resized, class_colors)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    fps = int(1/(time.time()-t_prev))
    cv2.rectangle(image, (5, 5), (75, 25), (0,0,0), -1)
    cv2.putText(image, f'FPS {fps}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow(win_title, image)
    
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()

```
    
 








---
## 參考教學

1. 手把手 Train 你的 YOLOv4 (上) — YOLOv4 起手式
https://mrhandbyhand.medium.com/hand-by-hand-train-your-yolov4-1-5f6a70618500

2. 手把手 Train 你的 YOLOv4 (下) — 從訓練到使用 — 實戰 kaggle 口罩資料集
https://mrhandbyhand.medium.com/hand-by-hand-train-your-yolov4-kaggle-dataset-ac1456e06604

3. YOLOv4 訓練教學
https://medium.com/ching-i/yolo-c49f70241aa7

4. NVIDIA Jetson Nano使用Tensor RT加速YOLOv4神經網路推論
https://www.rs-online.com/designspark/nvidia-jetson-nanotensor-rtyolov4-cn

5. NVIDAI Jetson Nano深度學習應用-使用OpenCV處理YOLOv4即時影像辨識
https://www.rs-online.com/designspark/nvidai-jetson-nano-opencvyolov4-cn
