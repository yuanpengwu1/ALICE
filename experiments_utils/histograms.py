import cv2
import numpy as np
import matplotlib.pyplot as plt


def save_histogram_image(image_path, output_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or cannot be opened.")

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算直方图
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # 创建画布
    fig, ax = plt.subplots(facecolor='white')

    # 填充灰色直方图
    ax.fill_between(range(256), hist.ravel(), color='gray')

    # 设置背景颜色为白色
    ax.set_facecolor('white')

    # 去除坐标轴
    ax.set_xticks([])
    ax.set_yticks([])

    # 去除边框
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 保存直方图图像
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=False)
    plt.close()


# 使用函数生成并保存直方图图像
image_paths = [
    '/home/tdx/YPW/LCM/Real/Real/Test/R1440/Mono/0019.png',
]

output_paths = [
    '/home/tdx/YPW/monohistogram1.png',
]

for image_path, output_path in zip(image_paths, output_paths):
    save_histogram_image(image_path, output_path)

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def save_histogram_image(image_path, output_path):
#     # 读取图像
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError("Image not found or cannot be opened.")
#
#     # 转换为灰度图像
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # 计算直方图
#     hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
#
#     # 创建画布
#     fig, ax = plt.subplots(facecolor='white')
#
#     # 用红色填充
#     ax.fill_between(range(256), hist.flatten(), color='red')
#
#     # 设置背景颜色为白色
#     ax.set_facecolor('white')
#
#     # 去除坐标轴
#     ax.set_xticks([])
#     ax.set_yticks([])
#
#     # 去除边框
#     for spine in ax.spines.values():
#         spine.set_visible(False)
#
#     # 保存直方图图像
#     plt.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=False)
#     plt.close()
#
#
# # 使用函数生成并保存直方图图像
# image_paths = [
#     '/home/tdx/YPW/LCM/Real/Real/Test/R1440/GT/0019.png',
#     '/home/tdx/YPW/LCM/Real/Real/Test/R1440/Color/0019.png',
# ]
#
# output_paths = [
#     '/home/tdx/YPW/GThistogram.png',
#     '/home/tdx/YPW/colorhistogram.png',
# ]
#
# for image_path, output_path in zip(image_paths, output_paths):
#     save_histogram_image(image_path, output_path)


