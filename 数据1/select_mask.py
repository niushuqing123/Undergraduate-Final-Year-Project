import os
import shutil

# img和mask文件夹路径
img_dir = "img_原始"
mask_dir = "mask_target"

# img1和mask1文件夹路径
img1_dir = "img1"
mask1_dir = "mask1"

# 获取img1文件夹中的图片名称列表
img1_names = os.listdir(img1_dir)

# 遍历img1文件夹中的图片，将对应的mask文件复制到mask1文件夹中
for img1_name in img1_names:
    # 获取对应的mask文件名
    mask_name = img1_name.replace(".jpg", ".png")  # 根据需要修改后缀名
    mask_path = os.path.join(mask_dir, mask_name)
    mask1_path = os.path.join(mask1_dir, mask_name)
    # 判断mask文件是否存在，存在则复制到mask1文件夹中
    if os.path.exists(mask_path):
        shutil.copy(mask_path, mask1_path)
    else:
        print(f"Mask file not found for image {img1_name}.")