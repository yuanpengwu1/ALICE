#!/user/bin/env python
# -*- coding:utf-8 -*-
from PIL import Image

img = Image.open("Koala.jpg")   # 读取图片
img = img.convert("L")   # 转化为黑白图片
img.save("444.jpg")   # 存储图片
