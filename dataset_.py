import os

import cv2
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self,img_dir, mask_dir,num_classes, mask_ext='.png', img_ext='.png', transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ids=os.listdir(self.img_dir)

        
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        # print("self.img_ids",type(self.img_ids))
        return len(self.img_ids)
    # 这两个函数说明pytorch的dataset再取数据时的操作是：for idx in range（len）：get（idx）...
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = cv2.imread(os.path.join(self.img_dir, img_id))

        # print("++++++++++++++++++++")

        # print(os.path.join(self.img_dir, img_id + self.img_ext))

        mask = []
        # print("num_classes")
        # print(self.num_classes)
        # num_classes=1.i恒等于0代表路径中有‘0’这个文件夹
        # [..., None]升维
        # for i in range(self.num_classes):
            # print("dir:",os.path.join(self.mask_dir, str(i),img_id + self.mask_ext))
            
            # t=cv2.imread(os.path.join(self.mask_dir, str(i),img_id + self.mask_ext), cv2.IMREAD_COLOR)[..., None]
            # print("i.shape:",t.shape)

            # mask.append(cv2.imread(os.path.join(self.mask_dir,
            #             img_id ), cv2.IMREAD_UNCHANGED)[..., None])
            #             # img_id + self.mask_ext), cv2.IMREAD_UNCHANGED))
            # print("np.array(mask).shape:",np.array(mask).shape)
            
        p=os.path.join(self.mask_dir,img_id)
        temp=cv2.imread(p, cv2.IMREAD_UNCHANGED)
        temp=temp[..., None]
        mask.append(temp)
        
        mask = np.dstack(mask)
        # print("mask.shape):",mask.shape)

        # print("++++++++++++++++++++")

        # 如果要使用数据增强，则执行：
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)#这个包比较方便，能把mask也一并做掉
            img = augmented['image']#参考https://github.com/albumentations-team/albumentations
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        
        return img, mask
