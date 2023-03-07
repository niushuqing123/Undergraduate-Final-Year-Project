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
transform = albumentations.Compose([
    # albumentations.Resize(576,576),
    # albumentations.Resize(384,384),
    # albumentations.Resize(96,96),
    albumentations.Normalize()
])

# transform =None



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



def train(data_loder,model,optimizer,loss_func,jaccard=None):
    model.train()
    # 从loder中取数据，每份是一个batch，img和mask是一组数据，一组里有多少取决于batchsize
    loss_sum=0
    iou_sum=0
    i=0
    for img,mask in data_loder:
        i+=1
        #for循环从data_loder里取数据，每次取出的img,mask都是按照batchsize打包的一组图片
        #batchsize小，就要取很多次，大的话几次就把data_loder用完了
        #之后将用to（device）的方法替代所有cuda
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # img_numpy=img.numpy()
        img=img.cuda()
        mask=mask.cuda()
        
        # print(img)
        # print(mask)
        
        # print("inputs.size()")
        # print(img.size())
        # print(mask.size())

        # 当img在GPU上时，outputs也会在GPU上
        outputs=model(img)
        # outputs_numpy=outputs.detach().cpu().numpy()
        # print("outputs.size()")
        # print(outputs.size())
        # print(mask.size())

        loss=loss_func(outputs,mask)
        loss_sum+=loss.item()
        #(outputs>0)又来了，根本就不能确定到底取那个值才是最优解
        
        outputsforiou=(outputs>0)
        maskforiou=mask.type(torch.uint8)
        iou_sum+=jaccard(outputsforiou,maskforiou).item()

        # 当(outputs,mask)在GPU上时，损失函数的计算也会在GPU上
        # loss_func=torch.nn.functional.binary_cross_entropy_with_logits(outputs,mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_avg=loss_sum/i
    iou_avg=iou_sum/i
    # return loss.item()
    return loss_avg,iou_avg


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


def main():
    cudnn.benchmark = True
    cudnn.enabled = True

    data_train=Data.MYdata(train_path,mask_path=r'mask30_01',transform=transform)

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
    # model.cuda()
    # print("type(model):",type(model))

    #对于损失函数的调用，一种是函数式调用，一种是类调用，暂时选择类调用，也许类调用更好
    # loss_func = torch.nn.functional.binary_cross_entropy_with_logits
    loss_func = torch.nn.BCEWithLogitsLoss().cuda()
    # loss_func = torch.nn.CrossEntropyLoss()
    # loss_func = BCEDiceLoss().cuda()
    # loss_func = BCEDiceLoss()
    
    
    # params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer  = torch.optim.Adam(model.parameters())
    
    # optimizer  = torch.optim.SGD(params, lr=0.001, momentum=0.9)
    # optimizer  = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer  = torch.optim.SGD(params, lr=0.001, momentum=0.9,nesterov=False,weight_decay=0.0001)

    #iou
    jaccard = torchmetrics.JaccardIndex(task="binary").cuda()

    e=0
    epoch=66
    while(e<epoch):
        # loss_e=train(data_loder,model,optimizer,loss_func)
        loss_e,iou_e=train(data_loder,model,optimizer,loss_func,jaccard=jaccard)
        # verify(val_loder,model,loss_func)

        # print("loss_e: ",e,loss_e)
        print("loss_e: ",e,loss_e,iou_e)

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
#unet2可用，但是会出错

#航空图像data36，resize576，batchsize4，显存极限
#航空图像的mask分类，区分文件夹
#抽空做一个最大的航空图像数据集，测试大数据集高迭代能出什么效果
#细胞任务的iou要实现一下
#从细胞任务着手，尝试嵌套uent
#从细胞任务着手，尝试自己改写unet
#unet的输入尺寸调整

#学习率：过小收敛慢，过大不收

#高分辨率可以overlap

# https://www.bilibili.com/video/BV1rq4y1w7xM/?p=1&share_medium=android&share_plat=android&share_source=COPY&share_tag=s_i&timestamp=1641867183&unique_k=PCJJmqN&vd_source=135341a8f58b630cefa83526404a27a5

# 测试torch和albumentations的Normalize有什么区别
# 回到细胞任务测试01mask到底有什么问题
# 在进行细胞任务备份的norm问题时想到：mask01这个问题也许是因为img和mask的数值不在一个大小范围内导致的
# 也就是说，img的像素值都是各通道0-255，如果这时候mask为01的话可能会导致一些问题：（loss计算时对数值不敏感，对参数更新力度不足，梯度太小，总之0255和01区间上差了二百多倍很可能是有问题的）（同时norm或同时不norm的结果是一样的，都是对的）
# 所以：如果要是用01的mask，可以尝试将img做norm，而mask不做，因为01mask本身就相当于norm了

# 对街道的分割，似乎由于街道的形状比较长条，整体跨度范围范围大，而导致需要更大的感受野。并且街道和路的颜色经常与建筑物或者水面相似。
# 如何改善对道路的分割效果，大致方向是，降低模型对颜色敏感度，提高对形状和纹理的敏感度。（可能的相关内容：提高感受野的空洞卷积，加大卷积核，或者加大卷积或者特征图的尺寸）
# 因为未来考虑三分类，还要把car这个类别分割出来，然而car这个任务的需求跟街道正好相反，所以初步设想能否在网络内部对街道和车辆做出区分，比如处理对不同通道作不同处理，或者在某一步分两条线进行


#3.7，目前最新进度是已经成功分割了航拍道路，在unet和unet++均成功（正确率可接受），已经制作了汽车mask，尚未实施但原理与道路类似。三分类尚未开始，训练输入输出值问题仍不明确但不影响

main()


# #测试torch给的iou指标
# def xixi():
        
#     import torch
#     import torchmetrics

#     # create tensor with binary classification
#     preds = torch.tensor([1, 1, 1, True]).cuda()
#     target = torch.tensor([1, 1, 0, 0]).cuda()


#     # initialize IoU metric
#     # jaccard = torchmetrics.IoU(num_classes=2)
#     jaccard = torchmetrics.JaccardIndex(task="binary").cuda()

#     # calculate IoU score
#     iou_score = jaccard(preds, target)

#     print(f"IoU score: {iou_score:.2f}")
    
# xixi()