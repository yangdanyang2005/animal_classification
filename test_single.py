# test.py
import torch
from torchvision import transforms, models
from PIL import Image

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载类别标签
class_names = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']  # 根据你数据集的类别名称调整

# 加载训练好的模型
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(class_names))  # 调整最后一层输出
model.load_state_dict(torch.load('best_model.pth'))
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
    
    img = Image.open(img_path)
    img = transform_predict(img).unsqueeze(0).to(device)  # 增加batch维度并转移到设备

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
    
    return class_names[preds.item()]

# 进行预测
img_path = 'datasets/test/cow/cow_1.jpeg'  # 你可以修改这个路径为你想预测的图片
predicted_class = predict(model, img_path)
print(f"预测的类别是: {predicted_class}")
