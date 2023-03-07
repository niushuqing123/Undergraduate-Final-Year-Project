import os
import torch
import torch.utils.data
import torch.nn
import torch.nn.functional
import torch.backends.cudnn as cudnn
import torchmetrics

import torch.optim
import albumentations


import cv2

import dataset_
import Data
# import unet
from net.archs_ import UNet as UNet1
from net.archs_ import NestedUNet as NestedUNet1
from net.net_ import UNet as UNet2
from net.unet_.model import UNet as Unet3
from net.unet_.model import NestedUNet as NestedUNet2


# from torch.utils.tensorboard import SummaryWriter #tensorboard



train_path=r'data\data_train'
val_path=r'data\data_val'  



def train(data_loder,model,optimizer,loss_func):
    model.train()
    # 从loder中取数据，每份是一个batch，img和mask是一组数据，一组里有多少取决于batchsize
    for img,mask in data_loder:
        #之后将用to（device）的方法替代所有cuda
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # img_numpy=img.numpy()
        o=0
        img=img.cuda()
        mask=mask.cuda()
        
        # print(img)
        # print(mask)
        
        # print("inputs.size()")
        # print(img.size())
        # print(mask.size())

        # 当img在GPU上时，outputs也会在GPU上
        outputs=model(img)
        outputs_numpy=outputs.detach().cpu().numpy()
        # print("outputs.size()")
        # print(outputs.size())
        # print(mask.size())

        loss=loss_func(outputs,mask)

        # 当(outputs,mask)在GPU上时，损失函数的计算也会在GPU上
        # loss_func=torch.nn.functional.binary_cross_entropy_with_logits(outputs,mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def verify(data_loder,model,loss_func):
    model.eval() # 将模型置为评估模式
    loss_avg=0

    with torch.no_grad():
        cnt=0
        loss_sum=0
        for inputs, labels in data_loder:
            # 将数据和模型放到GPU上
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs)

            loss = loss_func(outputs, labels)
            loss_sum+=loss.item() 
            cnt+=1
        loss_avg=loss_sum/cnt

    return loss_avg


class BCEDiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
    #     print("input.size()")
    #     print(input.size())
    #     print(target.size())

        bce = torch.nn.functional.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


# transform = albumentations.Compose([
#     albumentations.Normalize()
# ])

transform = albumentations.Compose([
    # transforms.RandomRotate90(),
    albumentations.RandomRotate90(),
    # transforms.Flip(),
    albumentations.Flip(),
    albumentations.OneOf([
        albumentations.HueSaturationValue(),
        albumentations.RandomBrightness(),
        # transforms.RandomContrast(),
        albumentations.RandomBrightnessContrast(),
    ], p=1),#按照归一化的概率选择执行哪一个
    # transforms.Resize(config['input_h'], config['input_w']),
    albumentations.Resize(96,96),
    # albumentations.Resize(512,512),
    albumentations.Normalize(),
])
transform =None


def main():
    cudnn.benchmark = True
    cudnn.enabled = True

    data_train=Data.MYdata(train_path,transform=transform)

    # data_train=dataset_.Dataset(os.path.join(train_path, r'img'),os.path.join(train_path, r'mask'),1,transform)



    # data_val=Data.MYdata(val_path)
    # 添加验证集，改进路径写法，把细节放进dataset文件
    # 传入的，只需要有一个最高级目录地址即可


    print(data_train[0][0].shape)
    print(data_train[0][1].shape)
    

    # print("type(data):",type(data))
    # print("type(data):",type(data[0]))
    # print("type(data):",type(data[0][1]))
    # print("type(data.__getitem__(0)[0]):",type(data.__getitem__(0)[0]))
    # print("data.__getitem__(0)[0].size():",data.__getitem__(0)[0].size())
    
    # batch_size=6/18/31
    data_loder=torch.utils.data.DataLoader(dataset=data_train,batch_size=8)
    # val_loder=torch.utils.data.DataLoader(dataset=data_val,batch_size=2)

    model=UNet1(num_classes=1).to(torch.device('cuda'))
    # model=NestedUNet1(num_classes=1).to(torch.device('cuda'))
    # model.cuda()#to()等价于这句
    # print("type(model):",type(model))


    loss_func = torch.nn.functional.binary_cross_entropy_with_logits
    # loss_func = BCEDiceLoss().cuda()
    # loss_func = BCEDiceLoss()
    
    
    # params = filter(lambda p: p.requires_grad, model.parameters())

    # optimizer  = torch.optim.Adam(model.parameters())
    # optimizer  = torch.optim.SGD(params, lr=0.001, momentum=0.9)
    optimizer  = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer  = torch.optim.SGD(params, lr=0.001, momentum=0.9,nesterov=False,weight_decay=0.0001)

    #iou
    # jaccard = torchmetrics.JaccardIndex

    e=0
    epoch=66
    while(e<epoch):
        loss_e=train(data_loder,model,optimizer,loss_func)
        # verify(val_loder,model,loss_func)

        print("loss_e:",loss_e)

        torch.cuda.empty_cache()
        e+=1
    
    # 把训练函数和验证函数都调通，然后直接保存模型
    # 调整迭代和数据集，测试简陋的训练手段是否有效果

    # 提高开发速度，否则时间长了代码难以控制，前面的部分会忘记
    
    # 从torchhub上找找

    torch.save(model.state_dict(),'model/model.pth')



#测试损失函数，gpt说Adam可以会过拟合，测试小数据验证
# 测试优化器
#dataset里面/255
#dataset里面调试看详细数据
#找到output数值差异的原因


#当数据集变化时，网络处理的尺寸也要修改
#先测试之前的猜测，细胞分割的每个细节都要检查到，dataset的数据、训练时的数据

#然后先修改已有的unet，再尝试自己搭一下

#在同一个dataset里验证两种读取方式的结果是否完全一致？

#在u++中数据集30，迭代40，可以有正确的结果

#问题定位：batchsize，dataset中mask(已解决：是dataset的问题)
#当数据集和辅助配件均正确时，30张66次训练就可以有很高正确率
#在局部测试中，有两张图效果不好，之后放到完整训练的模型中进行测试

main()
