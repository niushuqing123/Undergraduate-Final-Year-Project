import torch
import torch.utils.data
import torch._utils

import Data

import cv2



# from torch.utils.tensorboard import SummaryWriter #tensorboard

# img_dir=None
img_dir=r'data0\img'
mask_idr=r'data0\mask'




train_path=r'data\data_train'
val_path=r'data\data_train'  

def __main__():

    data=Data.MYdata(train_path)

    '''
    get=data.__getitem__(0)
    print(type(get))
    print((get))
    print(type(get[0]))
    print((get[0]))

    cv2.imshow('IMREAD',(get[0]))
    var=cv2.waitKey()
    '''
    

    print(data[0][0].shape)
    print(data[0][1].shape)
    
    print("type(data):",type(data))
    print("type(data):",type(data[0]))
    print("type(data):",type(data[0][1]))
    print("type(data.__getitem__(0)):",type(data.__getitem__(0)[0]))


    get=data.__getitem__(3)

    print("type(get)):",type(get[0]))
    print("(get)):",get[0].shape)

    img_=get[0].numpy()

    img_t = img_.transpose(1, 2,0)

    print(img_t.shape)

    cv2.imshow('IMREAD',img_t)
    var=cv2.waitKey()



    data_loder=torch.utils.data.DataLoader(dataset=data,batch_size=2)

    print(data_loder)

    i=0
    # 从loder中取数据，每份是一个batch，img和mask是一组数据，一组里有多少取决于batchsize
    for img,mask in data_loder:
        i+=1
        print("i:",i)

        # print(img)
        # print(mask)


        print(type(img))
        print(type(mask))
        

        img_=img.numpy()
        print(type(img_))
        print(type(img_[0]))
        print(img_[0].shape)
        
        # 由于dataset类中get函数为了迎合torch的格式，在输出是对维度进行了置换，所以取出来还想显示图片的话必须把维度调整回去
        # 然后对于像素值/255的操作cv2是可以接受的，但显示图片时似乎慢了一些
        img_t = img_[0].transpose(1, 2,0)
        print(img_t.shape)
        cv2.imshow('IMREAD',img_t)
        var=cv2.waitKey()


    print(i)





    # cv2.imshow('IMREAD_UNCHANGED+Color',get)
    # var=cv2.waitKey()




__main__()
