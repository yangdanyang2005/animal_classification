import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import os
from PIL import Image
import copy
from tqdm import tqdm
import logging  # 导入 logging 用于记录训练日志
from datetime import datetime  # 导入 datetime 用于获取当前日期和时间

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epoch_num = 10  # 训练的总轮数
save_interval = 10  # 每隔多少轮保存一次模型

# 创建日志文件夹
if not os.path.exists('log'):
    os.makedirs('log')

# 获取当前日期和时间
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
# 设置日志文件路径，包含当前日期和时间
log_filename = f'log/train_log_{current_time}.txt'
# 配置日志记录
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info("训练开始")

# 数据预处理和增强
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 自定义数据集类，支持任意格式的图片
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list of str): 图片的路径列表。
            labels (list of int): 图片的标签。
            transform (callable, optional): 用于图片预处理的转换。
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        # 添加 classes 属性
        self.classes = sorted(list(set(self.labels)))  # 假设 self.labels 包含所有标签

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 加载图片
        image = Image.open(img_path).convert('RGB')  # 转换为RGB格式

        if self.transform:
            image = self.transform(image)
        
        return image, label

# 加载训练集和验证集图片路径和标签
def load_data(image_folder):
    image_paths = []  # 图片路径
    labels = []  # 图片标签
    for label, class_name in enumerate(os.listdir(image_folder)):
        class_folder = os.path.join(image_folder, class_name)
        if os.path.isdir(class_folder):
            for img_file in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_file)
                if img_path.endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # 支持的图片格式
                    image_paths.append(img_path)
                    labels.append(label)
    return image_paths, labels

# 加载训练集和验证集
train_image_paths, train_labels = load_data('./datasets/train')
val_image_paths, val_labels = load_data('./datasets/val')

# 创建数据集和数据加载器
train_dataset = CustomImageDataset(train_image_paths, train_labels, transform=transform_train)
val_dataset = CustomImageDataset(val_image_paths, val_labels, transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 检查类别标签
class_names = train_dataset.classes
print("类别标签：", class_names)

# 加载预训练的模型（例如 ResNet18）
model = models.resnet18(pretrained=True)

# 替换最后一层为适合你的类别数
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

# 将模型移到 GPU（如果可用）
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, epoch_num, save_interval):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epoch_num):
        print(f"Epoch {epoch + 1}/{epoch_num}")
        print('-' * 10)

        # 每个 epoch 包括训练和验证
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 训练模式
                dataloader = train_loader
            else:
                model.eval()   # 验证模式
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # 使用 tqdm 包装 dataloader，显示进度条
            for inputs, labels in tqdm(dataloader, desc=f"{phase} Epoch {epoch+1}", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)

                # 清零梯度
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            # 记录日志
            logging.info(f"{phase} Epoch {epoch + 1}/{epoch_num}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # 记录最好的模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                # 保存最佳模型
                best_model_path = os.path.join('model', f'best_model_epoch_{epoch+1}_best.pth')
                torch.save(model.state_dict(), best_model_path)
                print(f"最佳模型已保存到 {best_model_path}")

        # 每隔 `save_interval` 轮保存一次模型
        if (epoch + 1) % save_interval == 0:
            save_path = os.path.join('model', f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"模型已保存到 {save_path}")

        print()

    print('Best val Acc: {:4f}'.format(best_acc))

    # 加载最好的模型权重
    model.load_state_dict(best_model_wts)
    return model

# 训练模型
trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, epoch_num, save_interval)

# 保存最终训练后的模型
final_model_path = os.path.join('model', f'final_model_epoch_{epoch_num}.pth')
torch.save(trained_model.state_dict(), final_model_path)
logging.info(f"最终模型已保存到 {final_model_path}")
