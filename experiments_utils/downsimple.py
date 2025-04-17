import cv2

def downsample_image(input_path, output_path):
    # 读取输入图像
    image = cv2.imread(input_path)

    # 计算缩放后的尺寸（宽度和高度都减半）
    width = int(image.shape[1] / 2)
    height = int(image.shape[0] / 2)
    dim = (width, height)

    # 缩放图像
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # 保存缩放后的图像
    cv2.imwrite(output_path, resized_image)

    print(f"Image downsampled and saved to {output_path}")

if __name__ == "__main__":
    input_path = "../output_image2.jpg"  # 输入图像路径
    output_path = "../output_image3.jpg"  # 输出图像路径

    downsample_image(input_path, output_path)
