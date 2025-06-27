from torchvision import transforms
from torchvision.datasets import FashionMNIST
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt

train_data=FashionMNIST(root='D:/Lenet/data',
                        train=True,
                        transform=transforms.Compose([transforms.Resize(size=224),transforms.ToTensor()]),
                        download=True)
train_loader=Data.DataLoader(dataset=train_data,
                             batch_size=64,
                             shuffle=True,
                             num_workers=0)
#获得一个batch的数据
for step,(b_x,b_y) in enumerate(train_loader):  #获取一个batch的数据
    if step>0:
        break
batch_x=b_x.squeeze().numpy()  #将四维张量移除第一维，并转换为Numpy数据
batch_y=b_y.numpy() #将张量转为numpy格式
class_label=train_data.classes #训练集的标签
#可视化一个Batch的数据
plt.figure(figsize=(12,5))
for ii in np.arange(len(batch_y)):
    plt.subplot(4,16,ii+1)
    plt.imshow(batch_x[ii,:,:],cmap=plt.cm.gray)
    plt.title(class_label[batch_y[ii]],size=10)
    plt.axis("off")
    plt.subplots_adjust(wspace=0.05)
plt.show()
