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

## 設定

1. **clone** 

使用git clone 指令將本項目克隆到Jetson Nano 中:
```bash=
git clone wei900508/Jetson-Nano-YoloV4-tiny
```
:::info
若要使用自己的圖片
將圖片放進 training_data/images
將標記XML檔放進 training_data/annotations
:::




2. **更改圖片格式** 

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
for img in os.listdir("/content/yolo_train/data/images/"):
    if img.endswith(".png"):
        im = Image.open("/content/yolo_train/data/images/{}".format(img))
        rgb_im = im.convert("RGB")
        img_name = img.split(".")[0]
        rgb_im.save("/content/yolo_train/data/images/{}.jpg".format(img_name))

# 刪除原始 png 資料
for img in os.listdir("/content/yolo_train/data/images/"):
    if img.endswith(".png"):
        os.remove("/content/yolo_train/data/images/{}".format(img))
```
    

3. **複製一份影像資料**

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


準備訓練以及驗證資料集
路徑要依據你實際存放的路徑做修改，否則可能會噴錯

```
pytohn3 split_data.py
```

到這邊應該會有 train 以及 dev 資料夾，裡面分別有對應的 jpg 圖片檔案以及 YOLO 影像標註檔案(.txt)，在訓練前我們需要把資料輸出成一個 file list (圖片的絕對路徑) 這樣跑訓練時才會讓 YOLO 知道你的檔案在哪裡，待會會用到這兩個檔案。此步驟操作如下：

```
# 輸出 train.txt
cd ./train ; ls -d "$PWD"/*.jpg > train.txt
# 輸出 dev.txt
cd ./dev ; ls -d "$PWD"/*.jpg > dev.txt
```



參考教學
https://mrhandbyhand.medium.com/hand-by-hand-train-your-yolov4-1-5f6a70618500

https://mrhandbyhand.medium.com/hand-by-hand-train-your-yolov4-kaggle-dataset-ac1456e06604

https://medium.com/ching-i/yolo-c49f70241aa7

https://www.rs-online.com/designspark/nvidia-jetson-nanotensor-rtyolov4-cn