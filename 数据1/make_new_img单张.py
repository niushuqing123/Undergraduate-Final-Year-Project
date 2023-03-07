from PIL import Image

# 读入图片
img = Image.open('000100.png')

# 获取图片的宽度和高度
width, height = img.size

# 创建一个与原图大小相同的空图
new_img = Image.new('RGB', (width, height))

# 逐像素拷贝像素值
for x in range(width):
    for y in range(height):
        pixel = img.getpixel((x, y))
        new_img.putpixel((x, y), pixel)

# 保存新的图片
new_img = new_img.resize((1920, 1080))
new_img.save('000100e_new.png')





