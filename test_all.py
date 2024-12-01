import torch
from torchvision import transforms, models
from PIL import Image
import os

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载类别标签
class_names = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']  # 根据你数据集的类别名称调整

# 加载训练好的模型
model = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(class_names))  # 调整最后一层输出
model.load_state_dict(torch.load('./model/model_epoch_20.pth', weights_only=True))
# 推荐将 weights_only=True 来只加载模型的权重，而不加载模型结构。这样可以避免潜在的安全问题
model = model.to(device)
model.eval()  # 设置为评估模式

# 预测函数
def predict(model, img_path):
    transform_predict = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    try:
        img = Image.open(img_path).convert("RGB")  # 确保图像格式一致
        img = transform_predict(img).unsqueeze(0).to(device)  # 增加batch维度并转移到设备

        with torch.no_grad():
            outputs = model(img)
            _, preds = torch.max(outputs, 1)
        
        return class_names[preds.item()]
    except Exception as e:
        print(f"无法预测图片 {img_path}: {e}")
        return None

# 批量预测函数
def predict_folder(model, folder_path):
    predictions = {}
    supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")  # 支持的图片格式
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(supported_formats):
                img_path = os.path.join(root, file)
                pred = predict(model, img_path)
                if pred is not None:
                    predictions[img_path] = pred
    return predictions

# 使用文件夹路径进行批量预测
folder_path = 'datasets/test/cats_and_dogs_filtered/train/cat/'  # 修改为你要预测的文件夹路径
results = predict_folder(model, folder_path)

# 打印预测结果
for img_path, pred_class in results.items():
    print(f"图片: {img_path} -> 预测类别: {pred_class}")
