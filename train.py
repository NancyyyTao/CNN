import pandas as pd
import torcheval
from torch import nn
from torch.utils.data import DataLoader
import ALexmodel绘图
import torch
import torch.optim as optim
from torcheval.metrics.functional import r2_score
from 数据集预处理 import CustomAGB, CustomPNA, CustomNNI, CustomAND
from torchvision.transforms import v2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)   # 打印当前使用的设备（GPU(cuda:0)还是CPU）

# seed = torch.initial_seed()
# print('Used seed : {}'.format(seed))

NUM_epochs = 50
BATCH_SIZE = 70
path='E:/17-20/2.2/CNN/AND_Alexnet50.pth'


# data loader
full_dataset=CustomAND('E:/17-20/paths.csv')

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
train_dataset2=train_dataset

augment1=v2.RandomHorizontalFlip(p=1)
augmented_data1=augment1(train_dataset)

augment2=v2.RandomVerticalFlip(p=1)
augmented_data2=augment1(train_dataset2)

full_dataset = torch.utils.data.ConcatDataset([train_dataset,augmented_data1,augmented_data2])
train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=test_size,shuffle=True)


"""
# model
model = AlexNet.alexnet(pretrained=True)  # 选择模型
for param in model.parameters():  # 将所有的参数层进行冻结
    param.requires_grad = False
# 修改第一层
model.features[0] = nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)
# 修改全连接层
num_fc_ftr = model.classifier[6].in_features # 获取第7层的结果，索引是从 0 开始计数
new_fc_layer = nn.Linear(num_fc_ftr,1) # 如果是分类问题使用len()返回变量中元素的个数或长度
model.classifier[6] = new_fc_layer  # 替换
# 解冻第一层和最后一层的权重
for param in model.features[0].parameters():
    param.requires_grad = True
for param in model.classifier[6].parameters():
    param.requires_grad = True
model.to(device) # 转到GPU/CPU上
print(model) # 最后再打印一下新的模型
"""
model = ALexmodel绘图.AlexNet(num_classes=1)
model.to(device)

# 损失函数
loss_function = nn.MSELoss()
loss_function.to(device)

# optimizer
optimizer = optim.Adam(params=model.parameters(), lr=0.0001)# Adm法 学习率为0.0001
# 每30个epoch降低一次学习率 论文中，alexnet将错误率（应该指的是验证集）作为指标，
# 当错误率一旦不再下降的时候降低学习率。alexnet训练了大约90个epoch，学习率下降3次
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
print('LR Scheduler created')

best_r2, best_RMSE = 0, 0

# 在训练的过程中会根据验证集的最佳准确率保存模型
try:
    # ready to go
    for epoch in range(NUM_epochs):
        loss_sum = 0.00
        r2_train_sum = 0.00
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = data.cuda(), target.cuda()
            target=torch.tensor(target, dtype=torch.float32).reshape(-1, 1)
            # target = target.unsqueeze(dim=1) #0维tensor转1维
            # target = target.to(torch.float32)

            # 计算损失
            optimizer.zero_grad()
            output = model(data) # 输出格式为1维tensor
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step() # 如果效果变成将该句代码转到lr_scheduler.step()前

            r2_train = torcheval.metrics.functional.r2_score(output, target)
            loss_sum += loss.data
            r2_train_sum += r2_train.data
            # 打印训练集

        RMSE_train=torch.sqrt(loss_sum)/ len(train_loader)
        R2_TRAIN = r2_train_sum/ len(train_loader)
        print('Train Epoch: {} [{}/{}] RMSE: {:.2f} R2: {:.2f} lr: {:.2e}'.format(
                 epoch, batch_idx * len(data), len(train_loader.dataset),
                 RMSE_train, R2_TRAIN, optimizer.param_groups[0]['lr']))


        # 打印测试集 Epoch Loss Acc
        test_loss = 0.00
        r2_test_sum = 0.00
        model.eval()
        for data, target in test_loader:

            data, target = data.cuda(), target.cuda()
            target = target.unsqueeze(dim=1)  # 0维tensor转1维
            target = target.to(torch.float32)

            output = model(data)

            test_loss += loss_function(output, target).data
            r2_test = torcheval.metrics.functional.r2_score(output, target)
            r2_test_sum += r2_test.data

        r2_test_sum = r2_test_sum/ len(test_loader)
        RMSE_test = torch.sqrt(test_loss)/ len(test_loader)
        print('Test Epoch: {} [{}/{}] RMSE: {:.2f} R2: {:.2f} lr: {:.2e}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            RMSE_test, r2_test_sum, optimizer.param_groups[0]['lr']))

        lr_scheduler.step()  # 学习率降梯度

        if r2_test_sum > best_r2:
            torch.save(model.state_dict(), path)  # 保存模型参数
            best_r2 = r2_test_sum
            best_RMSE = RMSE_test
            output = output.cpu()
            target = target.cpu()
            output_numpy = output.detach().numpy()
            target_numpy = target.detach().numpy()
            dff = pd.DataFrame({'Y_test': target_numpy.flatten(), 'Prediction': output_numpy.flatten()})

except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    model.eval()
    train_loader = DataLoader(train_dataset, batch_size=train_size, shuffle=True)
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        target = target.unsqueeze(dim=1)  # 0维tensor转1维
        target = target.to(torch.float32)
        output = model(data)
    output = output.cpu()
    target = target.cpu()
    output_numpy = output.detach().numpy()
    target_numpy = target.detach().numpy()
    dff_train = pd.DataFrame({'Y_test': target_numpy.flatten(), 'Prediction': output_numpy.flatten()})

    print("Best r2: {:.3f}, Best RMSE: {:.3f}".format( best_r2, best_RMSE))
    dff_train.to_csv('E:/17-20/2.2/AND50训练.csv', index=False)
    dff.to_csv('E:/17-20/2.2/AND50.csv', index=False)
