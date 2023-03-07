import torch
import torch.utils.data

import os
import cv2

from torchvision import transforms

import numpy as np

np.set_printoptions(threshold=np.inf)

transform_tensor = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize()
    # transforms.Resize(1920,1080),
])

# import numpy 

# torch.utils.data.dataset.IterableDataset
class MYdata(torch.utils.data.Dataset):
    def __init__(self,data_path,img_path=r'img',mask_path=r'mask',transform=None):
        # 继承的写法有待实验
        # super().__init__()

        self.img_dir=os.path.join(data_path,img_path)
        self.mask_dir=os.path.join(data_path,mask_path)
        
        # listdir可替代这句
        # self.len = len([os.path.splitext(os.path.basename(p))[0] for p in img_ids])

        self.img_idx=os.listdir(self.img_dir)
        self.mask_idx=os.listdir(self.mask_dir)
        
        self.len=len(self.img_idx)
        self.transform=transform

    def __len__(self):
        return self.len


    def __getitem__(self, index):
        img=cv2.imread(os.path.join(self.img_dir,self.img_idx[index]))
        mask=cv2.imread(os.path.join(self.mask_dir,self.mask_idx[index]),cv2.IMREAD_GRAYSCALE)
        #此处的mask读取方式为灰度图，也就是单通道图片， 若以后需要多类别多通道mask的读取也需要修改
        # cv2全是BGR，都要换成RGB
        # img = img[..., ::-1]
        # ------------------------------------------------------------------------------------------------------------------------------
        # ？？？？？？？？？？？？？？？？？？？按道理cv2读进来的图片应该BGRtoRGB，但转换后却有问题，不转却更好？？？？？？？？？？？？？？？？
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # ------------------------------------------------------------------------------------------------------------------------------
        # print(mask.shape)
        # mask=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # print(mask.shape)


        # cv2.imshow('IMREAD_UNCHANGED+Color',img)
        # var=cv2.waitKey()

        # print("---------------",type(img))
        # print("---------------",type(mask))


        # print("***********",mask.shape)

        # 因为是用cv2的原因，都需要对维度置换，以符合torch的要求。若用PIL读取则不用
        # 除以255是对颜色值进行归一化，但为什么要归一化尚不明确，可能与mask的形式有关
        # img = img.astype('float32') / 255
        # img = img.transpose(2, 0, 1)
        # mask = mask.astype('float32') / 255
        # mask = mask.transpose(2, 0, 1)
        
        # 当使用了torchvision.transforms后，就不再需要手动转换格式了
        # img = img / 255
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)#这个包比较方便，能把mask也一并做掉
            img = augmented['image']#参考https://github.com/albumentations-team/albumentations
            # mask = augmented['mask']
        
        # print(img)
        # print(mask)

        img=transform_tensor(img)
        mask=transform_tensor(mask)
        
        # mask=np.expand_dims(mask,axis=0)

        
        # img = img.astype('float32') / 255
        # img = img.transpose(2, 0, 1)
        # mask = mask.astype('float32') / 255
        # mask = mask.transpose(2, 0, 1)

        # print("---------------",type(img))
        # print("---------------",type(mask))

        # print("---------------mask.shape):",mask.shape)

        # 用for x，x in data： ,return出去的数据都会变成<class 'torch.Tensor'>
        # 直接调用的或者对data中括号取值调用的return，则还是<class 'numpy.ndarray'>
        
        # print(img)
        # print(mask)
        return img,mask
        






