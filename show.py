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
    # transforms.Resize((384,384)),
])

transform = albumentations.Compose([
    albumentations.Normalize(),
])
# transform =None

# 加载模型
model = UNet1(num_classes=1).to(torch.device('cpu'))
model.load_state_dict(torch.load('model/model.pth', map_location='cpu'))
model.eval()


# 读取测试图片
data_path = r'data/data_test'
# img_path = "test.jpg"
data_idx=os.listdir(data_path)
data_idx_len=len(data_idx)
print(data_idx_len)


def fuckingtest(output,img,fuck):
    
    output = (output > fuck)
    output = output.astype(np.uint8) * 255
    
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    # print("output.shape",output.shape)

    # 调整图片大小以便拼接
    img = cv2.resize(img, (512, 512))
    output = cv2.resize(output, (512, 512))
    # print("output.shape",output.shape)


    # 将图片和分割结果水平拼接成一张图像
    result = cv2.hconcat([img, output])

    print("fuck:",fuck)
    cv2.imshow("TestImage", result)
    cv2.waitKey(0)
    
    
    #窗口老关来关去的不好
    # win_name = "TestImage: fuck=%d" % fuck
    # # 显示图片和分割结果
    # cv2.imshow(win_name, result)
    # cv2.waitKey(0)
    # cv2.destroyWindow(win_name)
    # cv2.destroyAllWindows()
    
    
def main():
    for i in range(data_idx_len):
        img_path=os.path.join(data_path,data_idx[i])
        print(img_path)

        img = cv2.imread(img_path)    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # input_numpy=input.numpy()
        input=img
        
        if transform!=None:
            input=transform(image=input)["image"]
            
        # input_numpy=input.numpy()
        
        input=transform_tensor(input)

        # 将图片转换成模型输入格式
        # input = torch.tensor(img.transpose((2, 0, 1)), dtype=torch.float32)/255
        print("input.shape",input.shape)
        
        # input = transform(img_BGR2RGB)
        
        input = input.unsqueeze(0)#增加一个维度模拟batchsize
        
        print("input.shape",input.shape)

        
        # 使用模型产生输出
        with torch.no_grad():
            output = model(input)
            output = output.squeeze(0).squeeze(0).numpy()#解batch、去通道数、转numpy
            
        fuckmax=output.max().astype(int)
        fuckmin=output.min().astype(int)
        
        for fuck in range(fuckmin,fuckmax):
            fuckingtest(output,img,fuck)
            #相当于在最后一步交互式调参了
            
            
        #如果是小范围值，用这个测试
        if(0):
            fuck_f=output.min()
            dt=0.02
            fuckmax_f=output.max()

            while(1):
                fuckingtest(output,img,fuck_f)
                fuck_f+=dt
                if(fuck_f>=fuckmax_f):break
            

    
    
main()