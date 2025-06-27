from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt
from model import LeNet
import torch
import torch.nn as nn
import time
import copy
import pandas as pd

# 获取训练和验证集的 DataLoader
def train_val_data_process():
    train_data = FashionMNIST(root='D:/Lenet/data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(28), transforms.ToTensor()]),
                              download=True)
    train_len = round(len(train_data) * 0.8)
    val_len = len(train_data) - train_len
    train_data, val_data = Data.random_split(train_data, [train_len, val_len])

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True, num_workers=2)
    return train_dataloader, val_dataloader

train_dataloader, val_dataloader = train_val_data_process()

# 训练模型
def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_all, train_acc_all, val_loss_all, val_acc_all = [], [], [], []
    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-" * 10)

        train_loss, train_corrects, train_num = 0, 0, 0
        model.train()
        for step, (b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_x.size(0)
            train_corrects += torch.sum(pre_lab == b_y.data)
            train_num += b_x.size(0)

        val_loss, val_corrects, val_num = 0, 0 ,0
        model.eval()
        for step, (b_x, b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)
            val_loss += loss.item() * b_x.size(0)
            val_corrects += torch.sum(pre_lab == b_y.data)
            val_num += b_x.size(0)

        train_loss_all.append(train_loss / train_num)
        # 将GPU上的张量移动到CPU上再转换为NumPy数组
        train_acc_all.append(train_corrects.double().cpu().numpy() / train_num)
        val_loss_all.append(val_loss / val_num)
        # 将GPU上的张量移动到CPU上再转换为NumPy数组
        val_acc_all.append(val_corrects.double().cpu().numpy() / val_num)

        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        time_use = time.time() - since
        print("训练和验证耗费的时间 {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))

    model.load_state_dict(best_model_wts)
    import os
    save_dir = 'Lenet'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(best_model_wts, os.path.join(save_dir, 'model.pth'))

    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all})

    return train_process

# 绘制准确率和损失曲线
def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, 'ro-', label='Train Loss')
    plt.plot(train_process["epoch"], train_process.val_loss_all, 'bs-', label='Val Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, 'ro-', label='Train Acc')
    plt.plot(train_process["epoch"], train_process.val_acc_all, 'bs-', label='Val Acc')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.show()

if __name__ == '__main__':
    LeNet_model = LeNet()
    train_dataloader, val_dataloader = train_val_data_process()
    train_process = train_model_process(LeNet_model, train_dataloader, val_dataloader, num_epochs=20)
    matplot_acc_loss(train_process)