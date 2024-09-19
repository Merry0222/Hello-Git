#引入库
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
#选择使用gpu进行训练
device = torch.device("cuda")
# 数据预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载训练集
train_dataset = torchvision.datasets.CIFAR10(root='./data',  # 数据集保存路径
                                             train=True,      # 是否为训练集
                                             transform=transform,  # 数据预处理
                                             download=True)   # 是否下载

# 加载测试集
test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                            train=False,
                                            transform=transform,
                                            download=True)

# 一次选择的训练量
batch_size = 64

# 加载训练数据
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)  # 是否打乱

# 加载测试数据
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)  # 是否打乱
#定义名为LeNet5的神经网络模型类，并初始化它的基本结构
class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        # 卷积层 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),	# 卷积
            nn.BatchNorm2d(6),		# 批归一化
            nn.ReLU(),)
        # 下采样
        self.subsampel1 = nn.MaxPool2d(kernel_size = 2, stride = 2)		# 最大池化(pooling)
        # 卷积层 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),)
        # 下采样
        self.subsampel2 = nn.MaxPool2d(kernel_size = 2, stride = 2) #(pooling)
        # 全连接
        self.L1 = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.L2 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.L3 = nn.Linear(84, num_classes)
    # 前向传播(前面是对卷积，池化层和全连接层的编写，接下来是调用他们的时候了)
    def forward(self, x):
        out = self.layer1(x)
        out = self.subsampel1(out)
        out = self.layer2(out)
        out = self.subsampel2(out)
        # 将上一步输出的像素展平成一维向量，以便下一步全连接（全连接只能输入一维数据，故需要展平操作）
        out = out.reshape(out.size(0), -1)
        # 全连接
        out = self.L1(out)
        out = self.relu(out)
        out = self.L2(out)
        out = self.relu1(out)
        out = self.L3(out)
        return out
#因为CIFAR10一共有10种数据类型,所以我们定义一个输出的类型数,赋值10
num_classes = 10
#创建LeNet5模型
model = LeNet5(num_classes).to(device)
#计算损失的函数
cost = nn.CrossEntropyLoss()
learning_rate = 0.001#能用就行，所以固定学习率
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#优化器
total_step = len(train_loader)
# 设置一共训练几轮（epoch）
num_epochs = 10
# 外部循环用于遍历轮次
for epoch in range(num_epochs):
    # 内部循环用于遍历每轮中的所有批次
    for i, (images, labels) in enumerate(train_loader):  
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)   # 通过模型进行前向传播，得到模型的预测结果 outputs
        loss = cost(outputs, labels)	# 计算模型预测与真实标签之间的损失

        # 反向传播和优化
        optimizer.zero_grad()	# 清零梯度，以便在下一次反向传播中不累积之前的梯度
        loss.backward()		# 进行反向传播，计算梯度
        optimizer.step()	# 根据梯度更新（优化）模型参数

        # 定期输出训练信息
        # 在每经过一定数量的批次后，输出当前训练轮次、总周轮数、当前批次、总批次数和损失值
        if (i+1) % 400 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
        		           .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
with torch.no_grad():	# 指示 PyTorch 在接下来的代码块中不要计算梯度
    # 初始化计数器
    correct = 0		# 正确分类的样本数
    total = 0		# 总样本数

    # 遍历测试数据集的每个批次
    for images, labels in test_loader:
        # 将加载的图像和标签移动到设备（通常是 GPU）上
        images = images.to(device)
        labels = labels.to(device)

        # 模型预测
        outputs = model(images)

        # 计算准确率
        # 从模型输出中获取每个样本预测的类别
        _, predicted = torch.max(outputs.data, 1)
        # 累积总样本数
        total += labels.size(0)
        # 累积正确分类的样本数
        correct += (predicted == labels).sum().item()

    # 输出准确率，正确的 / 总的
    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

