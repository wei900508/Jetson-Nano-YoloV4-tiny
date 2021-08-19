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