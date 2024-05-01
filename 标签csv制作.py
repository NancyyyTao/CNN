import os
import csv


def get_image_paths(root_folder):
    # 获取根文件夹下所有子文件夹的路径
    subfolders = [os.path.join(root_folder, folder) for folder in os.listdir(root_folder) if
                  os.path.isdir(os.path.join(root_folder, folder))]

    image_paths = []
    # 遍历每个子文件夹，获取其中所有的图片路径
    for folder in subfolders:
        image_files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.tif')]
        image_paths.extend(image_files)

    return image_paths


def save_paths_to_csv(image_paths, csv_file):
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['路径名称'])
        for path in image_paths:
            writer.writerow([path])


# 设置根文件夹路径
root_folder = 'E:/17-20ROI/'
# 获取图片路径
image_paths = get_image_paths(root_folder)
# 保存到 CSV 文件中
save_paths_to_csv(image_paths, 'E:/17-20/paths.csv')
