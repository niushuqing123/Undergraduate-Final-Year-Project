import os
from PIL import Image
import numpy as np
# 原始图片所在目录
dir_path = r'img_原始/'
# dir_path = r'data_test/'

# 新图片保存目录
save_path = r'img/'
# save_path = r'data_test/'

# 如果保存目录不存在，创建它
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 遍历目录下所有的图片
for filename in os.listdir(dir_path):
    if filename.endswith('.png'):
        # 读入图片
        img_path = os.path.join(dir_path, filename)
        img = Image.open(img_path)

        # 获取图片的宽度和高度
        width, height = img.size

        # 创建一个与原图大小相同的空图
        # new_img = Image.new('RGB', (width, height))

        # # 逐像素拷贝像素值
        # for x in range(width):
        #     for y in range(height):
        #         pixel = img.getpixel((x, y))
        #         new_img.putpixel((x, y), pixel)
                
        # # 使用NumPy数组逐像素拷贝像素值,（加速）
        # pixels = np.array(img)
        # new_pixels = np.zeros_like(pixels)
        # new_pixels[:, :, :] = pixels[:, :, :]
        # new_img = Image.fromarray(new_pixels)


        #再次加速
        # 转换为NumPy数组
        img_array = np.array(img)
        # 创建一个与原图大小相同的空图
        new_img_array = np.zeros((height, width, 3), dtype=np.uint8)
        # 逐像素拷贝像素值
        new_img_array[:] = img_array[:]
        # 转换回PIL的Image对象
        new_img = Image.fromarray(new_img_array)


        # 保存新的图片
        new_img = new_img.resize((1920, 1080))
        save_filename = f"{os.path.splitext(filename)[0]}.png"
        save_pathname = os.path.join(save_path, save_filename)
        
        print(save_pathname)
        new_img.save(save_pathname)