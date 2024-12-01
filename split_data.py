import os
import shutil
import random

# 设置随机种子，保证每次拆分一致
random.seed(42)

# 数据集路径
dataset_dir = './datasets/'

# 训练和验证集的比例
train_ratio = 0.8
val_ratio = 0.2

# 创建 train 和 val 文件夹
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'val')

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(val_dir):
    os.makedirs(val_dir)

# 定义图片格式
image_extensions = ('.jpeg', '.jpg', '.png', '.bmp', '.tiff', '.gif', '.webp')

# 遍历每个类别文件夹
for class_folder in os.listdir(os.path.join(dataset_dir)):
    class_folder_path = os.path.join(dataset_dir, class_folder)
    
    # 只处理文件夹
    if os.path.isdir(class_folder_path):
        # 创建训练集和验证集文件夹
        class_train_dir = os.path.join(train_dir, class_folder)
        class_val_dir = os.path.join(val_dir, class_folder)

        if not os.path.exists(class_train_dir):
            os.makedirs(class_train_dir)
        if not os.path.exists(class_val_dir):
            os.makedirs(class_val_dir)

        # 获取该类别下所有图片的路径，支持多种格式
        image_paths = [os.path.join(class_folder_path, filename) for filename in os.listdir(class_folder_path)
                       if filename.lower().endswith(image_extensions)]

        # 打乱图片路径列表
        random.shuffle(image_paths)

        # 计算训练集和验证集的数量
        num_images = len(image_paths)
        num_train = int(num_images * train_ratio)
        num_val = num_images - num_train

        # 拆分为训练集和验证集
        train_images = image_paths[:num_train]
        val_images = image_paths[num_train:]

        # 移动文件到对应文件夹
        for train_image in train_images:
            shutil.copy(train_image, os.path.join(class_train_dir, os.path.basename(train_image)))
        
        for val_image in val_images:
            shutil.copy(val_image, os.path.join(class_val_dir, os.path.basename(val_image)))

        print(f"类别 '{class_folder}' 拆分完成: {num_train} 训练集，{num_val} 验证集")

print("数据集拆分完成！")
