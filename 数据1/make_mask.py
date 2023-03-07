from PIL import Image
import os
import numpy as np


# 定义颜色与标签的映射关系
COLOR_TO_LABEL = {
    # (128,64,128): 1,  # 标签1
    (64,0,128): 255,  # 标签2移动的车辆，必要时作为道路的类别1
    (192,0,192): 255 # 标签2
    #其他都为零，在定义一张空图片的时候初始都是零
    
}
# Background    (0,0,0)
# Building		(128,0,0)
# Road			(128,64,128)
# Tree			(0,128,0)
# Low vegetation(128,128,0)
# Moving car	(64,0,128)
# Static car	(192,0,192)
# Human			(64,64,0)

def main():
    # dir_path = r'mask_target/'
    dir_path = r'mask_target36/'
    save_path= r'masktest/'
    for file_name in os.listdir(dir_path):
        if not file_name.endswith('.png'):continue
        img_path = os.path.join(dir_path, file_name)
        print("img_path",img_path)

        img = Image.open(img_path)

        # 将PIL图片转换为numpy数组
        img_array = np.array(img)

        # 创建空的标签数组
        label_array = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)

        # 对每种颜色进行处理
        for color, label in COLOR_TO_LABEL.items():
            # 找到对应颜色的像素
            # np.all(img_array == color, axis=2)返回的是一个布尔型的数组，表示img_array中每个像素是否满足等于color的条件，数组的形状与img_array的形状相同。其中，axis=2表示按照颜色通道进行判断，对于RGB三个通道的像素值，将其看作是一个3维数组，即(height, width, 3)，axis=2即表示对这个3维数组，以最后一个维度为轴进行判断，得到一个2维布尔数组，即(height, width)。
            # 返回的布尔型数组中，每个True或False表示对应位置的像素是否符合要求，可以使用这个数组进行像素值的选择或筛选。具体地，通过将这个数组作为索引，可以获取满足要求的像素位置坐标
            color_pixels = np.all(img_array == color, axis=0)

            # 将对应像素的标签设为对应值
            label_array[color_pixels] = label

        # 将标签数组保存为图片文件
        label_img = Image.fromarray(label_array)
        label_img=label_img.resize((1920,1080))
        
        #file_name[:-4]这个语法表示截取file_name字符串从第一个字符到倒数第五个字符（不包括第五个字符），即去掉了文件名的后缀名（通常为.png、.jpg等）。然后重新加上后缀：_mask.png
        label_img.save(os.path.join(save_path, f'{file_name[:-4]}.png'))
        print(os.path.join(save_path, f'{file_name[:-4]}.png'))
        
main()