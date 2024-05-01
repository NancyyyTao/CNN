import torch
import pandas as pd
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np
import tifffile
from torchvision.transforms import InterpolationMode


class CustomAGB(Dataset):
    """
    __init__()函数是初始逻辑发生的地方，例如读取 csv、分配转换、过滤数据等
    """

    def __init__(self,csv_path, transform=None):
        self.data_info = pd.read_csv(csv_path)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        # Calculate len
        self.data_len = len(self.data_info.index)
        self.transformations = transform
    """
    __len__()返回您拥有的样本数
    """
    def __len__(self):
        return self.data_len
    """
    __getitem__()函数返回数据和标签。该函数是从数据加载器中调用的，如下所示：
    img, label = MyCustomDataset.__getitem__(99)  # For 99th item
     __getitem__()返回单个数据点的特定类型（如张量、numpy 数组等）
    """
    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img= tifffile.imread(single_image_name)
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

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return img_as_tensor, single_image_label
