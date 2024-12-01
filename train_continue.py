import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import copy
from tqdm import tqdm  # 导入 tqdm
from PIL import Image
from datetime import datetime
import logging

# 设置全局训练配置
start_epoch = 20  # 假设你已经训练了 20 轮
continue_epoch = 10  # 继续训练的轮数（可以根据需要调整）
num_epochs = start_epoch + continue_epoch  # 总训练轮数

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.classes = list(set(labels))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# 加载数据集
def load_data(image_folder):
    image_paths = []
    labels = []
    for label, class_name in enumerate(os.listdir(image_folder)):
        class_folder = os.path.join(image_folder, class_name)
        if os.path.isdir(class_folder):
            for img_file in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_file)
                if img_path.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_paths.append(img_path)
                    labels.append(label)
    return image_paths, labels

train_image_paths, train_labels = load_data('./datasets/train')
val_image_paths, val_labels = load_data('./datasets/val')

train_dataset = CustomImageDataset(train_image_paths, train_labels, transform=transform_train)
val_dataset = CustomImageDataset(val_image_paths, val_labels, transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 加载预训练模型
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(set(train_labels)))
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 检查是否有已保存的模型，如果有则加载
checkpoint_path = f'model/model_epoch_{start_epoch}.pth'

if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))

# 动态生成日志文件名
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f'log/train_log_{current_time}.txt'

# 配置日志记录
if not os.path.exists('log'):
    os.makedirs('log')

logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info("训练开始")

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, start_epoch):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
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

            # 使用 tqdm 包裹 dataloader
            with tqdm(dataloader, desc=f"{phase} Progress", leave=False) as pbar:
                for inputs, labels in pbar:
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

                    # 更新进度条后缀
                    pbar.set_postfix({
                        "Loss": f"{running_loss / len(dataloader.dataset):.4f}",
                        "Acc": f"{running_corrects.double() / len(dataloader.dataset):.4f}"
                    })

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # 记录日志
            logging.info(f"{phase} Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

            # 记录最好的模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # 保存最佳模型
                best_model_path = f'model/model_best_epoch_{epoch + 1}_{current_time}.pth'
                torch.save(model.state_dict(), best_model_path)
                print(f"最佳模型已保存到 {best_model_path}")
        
        # 每训练10轮保存一次模型
        if (epoch + 1) % 10 == 0:
            model_save_path = f'model/model_epoch_{epoch + 1}_{current_time}.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f"模型已保存到 {model_save_path}")

    print('Best val Acc: {:4f}'.format(best_acc))

    # 加载最好的模型权重
    model.load_state_dict(best_model_wts)
    return model

# 继续训练模型
trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, start_epoch)

# 保存训练后的最终模型
final_model_path = f'model/model_epoch_final_{current_time}.pth'
torch.save(trained_model.state_dict(), final_model_path)
logging.info(f"最终模型已保存到 {final_model_path}")
