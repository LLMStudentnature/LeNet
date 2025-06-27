import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import LeNet

def test_data_process():
    test_data = FashionMNIST(root='D:/Lenet/data',
                              train=False,
                              transform=transforms.Compose([transforms.Resize(28), transforms.ToTensor()]),
                              download=True)
    test_dataloader=Data.DataLoader(dataset=test_data,
                                    batch_size=1,
                                    shuffle=True,
                                    num_workers=2)
    return test_dataloader

def test_model_process(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    #初始化参数
    test_corrects=0.0
    #
    test_num=0
    #只进行前向传播，而不计算梯度，从而节省内存，加快运行速度
    with torch.no_grad():
        for test_data_x,test_data_y in test_dataloader:
            test_data_x=test_data_x.to(device)
            test_data_y=test_data_y.to(device)
            model.eval()
            output = model(test_data_x)
            pre_label = torch.argmax(output, dim=1)
            #查找正确率
            test_corrects += torch.sum(pre_label == test_data_y.data)
            #查找次数
            test_num += test_data_x.size(0)

        test_acc=test_corrects/test_num
        print(f'Test Accuracy: {test_acc}')


if __name__ == '__main__':
    #加载模型
    model = LeNet()
    model.load_state_dict(torch.load('model.pth'))
    #加载测试数据
    test_dataloader=test_data_process()
    #加载模型测试
    test_model_process(model, test_dataloader)

    #设定测试所用到的设备
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    classes=['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
    with torch.no_grad():
        for b_x,b_y in test_dataloader:
            b_x=b_x.to(device)
            b_y=b_y.to(device)

            model.eval()
            output = model(b_x)
            pre_lab=torch.argmax(output, dim=1)
            result=pre_lab.item()
            label=b_y.item()
            print("预测值：",classes[result],"------","真实值：",classes[label])

