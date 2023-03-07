import torch
import torch.nn.functional as F

import cv2
import numpy as np
import os


# import unet
from net.archs_ import UNet as UNet1
from net.archs_ import NestedUNet as NestedUNet1
from net.net_ import UNet as UNet2
from net.unet_.model import UNet as Unet3
from net.unet_.model import NestedUNet as NestedUNet2


from torchvision import transforms
import albumentations

transform_tensor = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((96,96)),
    # transforms.Resize((576,576)),
    transforms.Resize((384,384)),

    
])

transform = albumentations.Compose([
    albumentations.Normalize(),
    albumentations.Resize(96,96)
])
transform =None

transform = albumentations.Compose([
    albumentations.Resize(96,96)
])
transform =None

transform = albumentations.Compose([
    albumentations.Normalize(),
])

# 加载模型
model = NestedUNet1(num_classes=1).to(torch.device('cpu'))
model.load_state_dict(torch.load('model/model.pth', map_location='cpu'))
model.eval()


# 读取测试图片
data_path = r'data1/data_test'
# img_path = "test.jpg"
data_idx=os.listdir(data_path)
data_idx_len=len(data_idx)
print(data_idx_len)


for i in range(data_idx_len):
    img_path=os.path.join(data_path,data_idx[i])
    print(img_path)

    img = cv2.imread(img_path)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # input_numpy=input.numpy()
    input=img
    
    input=transform(image=input)["image"]
    # input_numpy=input.numpy()
    
    input=transform_tensor(input)

    # 将图片转换成模型输入格式
    # input = torch.tensor(img.transpose((2, 0, 1)), dtype=torch.float32)/255
    print("input.shape",input.shape)
    
    # input = transform(img_BGR2RGB)
    
    input = input.unsqueeze(0)#增加一个维度模拟batchsize
    
    print("input.shape",input.shape)

    
    # 进行模型推理
    with torch.no_grad():
        
        output = model(input)
        # print("output.shape",output.shape)
                
        # output = F.sigmoid(output)


        output = output.squeeze(0).squeeze(0).numpy()
        # output = output.squeeze(0).numpy()
        print("output.shape",output.shape)


        # 将分割结果转换成RGB图像
        # print(output)
        # print("type(output)",type(output))
        #output的取值范围约为-20000~7000，大于零算1，小于零算0
        #sigmoid与output = (output > 0.5)作用一样，sigmoid后值为01，(output > 0.5)后值为bool
        output = (output > 0)
        # output=output[0,:,:]

        output = output.astype(np.uint8) * 255
        
        
        # output = output.astype(np.uint8) * 255
        # output = (output > 0.5).astype(int) * 255
        # output= (output > 0.40).astype(int) * 255


        print("output.shape",output.shape)
        # output=np.squeeze((output > 0.40)[0,:,:].astype(np.uint8))*255
        # print("type(output)",type(output))


        # output = cv2.applyColorMap(output, cv2.COLORMAP_JET)#这个函数是是把图片转换为彩色图
        # output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)#这个函数是是把单通道灰度图转换为彩色图，但是只是格式上为三通道彩色图，显示出来还是黑白的
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        print("output.shape",output.shape)

        # 调整图片大小以便拼接
        
        
        
        img = cv2.resize(img, (512, 512))
        output = cv2.resize(output, (512, 512))
        print("output.shape",output.shape)


    # 将图片和分割结果水平拼接成一张图像
    result = cv2.hconcat([img, output])

    # 显示图片和分割结果
    cv2.imshow("TestImage", result)

    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    k=1
    
    
