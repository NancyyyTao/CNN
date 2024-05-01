import torch.nn as nn
import matplotlib.pyplot as plt
plt.rcdefaults()

class AlexNet(nn.Module):

    def __init__(self, num_classes=1):
        super(AlexNet, self).__init__()
        self.CONV = nn.Sequential(
            nn.Conv2d(4, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.Regressor = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.init_weights() #初始化权重

    def init_weights(self):
        for layer in self.CONV:
            # 先一致初始化
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.normal_(layer.weight, mean=0, std=0.01) # 论文权重初始化策略
                nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 1)
            # 单独对论文网络中的2、4、5卷积层的偏置进行初始化
            nn.init.constant_(self.CONV[4].bias, 1)
            nn.init.constant_(self.CONV[10].bias, 1)
            nn.init.constant_(self.CONV[12].bias, 1)

    def forward(self, x):
        x = self.CONV(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.Regressor(x)  # classifier只是自定义的函数名称
        return x
