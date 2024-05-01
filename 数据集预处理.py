from torch.utils.data import DataLoader
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
        normalize=transforms.Normalize((0.059,0.025,0.207,0.419),(0.017,0.008,0.044,0.105))
        img_as_tensor = normalize(img_as_tensor)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return img_as_tensor, single_image_label

class CustomPNA(Dataset):
    """
    __init__()函数是初始逻辑发生的地方，例如读取 csv、分配转换、过滤数据等
    """

    def __init__(self,csv_path, transform=None):
        self.data_info = pd.read_csv(csv_path)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 2])
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
        normalize = transforms.Normalize((0.059, 0.025, 0.207, 0.419), (0.017, 0.008, 0.044, 0.105))
        img_as_tensor = normalize(img_as_tensor)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return img_as_tensor, single_image_label

class CustomNNI(Dataset):
    """
    __init__()函数是初始逻辑发生的地方，例如读取 csv、分配转换、过滤数据等
    """

    def __init__(self,csv_path, transform=None):
        self.data_info = pd.read_csv(csv_path)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 3])
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
        normalize = transforms.Normalize((0.059, 0.025, 0.207, 0.419), (0.017, 0.008, 0.044, 0.105))
        img_as_tensor = normalize(img_as_tensor)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return img_as_tensor, single_image_label

class CustomAND(Dataset):
    """
    __init__()函数是初始逻辑发生的地方，例如读取 csv、分配转换、过滤数据等
    """

    def __init__(self,csv_path, transform=None):
        self.data_info = pd.read_csv(csv_path)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 4])
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
        normalize = transforms.Normalize((0.059, 0.025, 0.207, 0.419), (0.017, 0.008, 0.044, 0.105))
        img_as_tensor = normalize(img_as_tensor)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return img_as_tensor, single_image_label

if __name__ == '__main__':
    """
    Example
    """
    full_dataset=CustomAGB('E:/17-20/paths.csv')
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset,batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

    for batch_idx, (data, target) in enumerate(train_loader):
        print(batch_idx)
    # mn_dataset_loader = DataLoader(dataset=AGB,shuffle=False)

