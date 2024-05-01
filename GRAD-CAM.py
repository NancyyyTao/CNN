import tifffile
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
import ALexmodel绘图
import torch
from 数据集预处理 import CustomAGB, CustomPNA, CustomNNI, CustomAND
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from torchvision import transforms


dataset=CustomPNA('E:/17-20/paths抽穗后.csv')
img,value=dataset[17] #第49张
path='E:/17-20/2.2/CNN/全/AND_Alexnet50.pth'
model = ALexmodel绘图.AlexNet()
model.load_state_dict(torch.load(path))
model.eval()

input_tensor=img.unsqueeze(0)
targets = [BinaryClassifierOutputTarget(0)]  #如果为 1，它将显示是什么将模型输出值拉向更高，如果为 0，它将显示是什么将其拉向较低
cam = GradCAM(model=model, target_layers=[model.CONV[12]])
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
grayscale_cam = grayscale_cam[0, :]

"""
img= tifffile.imread('E:/17-20ROI/170725/08.tif')
# 图像变换
height, width = img.shape[:2]
while width < 227 or height < 227:
    width *= 2
    height *= 2
img = torch.from_numpy(img) # 转张量
img = torch.permute(img, (2, 0, 1)) # C,H,W转H,W,C
resize = transforms.Resize((height, width),InterpolationMode.NEAREST) # 定义resize变化
img = resize(img)
crop = transforms.CenterCrop(227)
img_as_tensor = crop(img)
img = img_as_tensor.detach().numpy()
tifffile.imwrite('E:/17-20/output.tif', img)
"""

plt.imshow(grayscale_cam, cmap='jet', interpolation='nearest')
plt.colorbar()
plt.show()