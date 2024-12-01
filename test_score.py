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

# 计算类别正确率
def calculate_accuracy(predictions, true_labels):
    correct = 0
    total = 0
    
    for img_path, pred_class in predictions.items():
        # 获取文件夹名称作为实际标签
        true_class = os.path.basename(os.path.dirname(img_path))
        
        if pred_class == true_class:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy

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

# 批量处理测试集文件夹，计算总的准确率
def evaluate_model_on_test_set(model, test_folder):
    all_predictions = {}
    all_true_labels = {}
    total_correct = 0
    total_images = 0
    
    for class_folder in os.listdir(test_folder):
        class_folder_path = os.path.join(test_folder, class_folder)
        if os.path.isdir(class_folder_path):  # 只处理文件夹
            print(f"正在预测类别 '{class_folder}' 中的图片...")
            # 对每个类别的图片进行预测
            predictions = predict_folder(model, class_folder_path)
            
            # 计算每个类别的正确率
            accuracy = calculate_accuracy(predictions, class_folder)
            print(f"类别 '{class_folder}' 的准确率: {accuracy * 100:.2f}%")
            
            all_predictions.update(predictions)
            all_true_labels[class_folder] = len(predictions)

            # 统计总正确和总图片数
            total_images += len(predictions)
            total_correct += sum(1 for img_path, pred_class in predictions.items()
                                 if os.path.basename(os.path.dirname(img_path)) == class_folder)

    overall_accuracy = total_correct / total_images if total_images > 0 else 0
    print(f"整体准确率: {overall_accuracy * 100:.2f}%")

# 使用文件夹路径进行批量预测并计算准确率
test_folder_path = './datasets/test'  # 修改为测试集的根目录路径
evaluate_model_on_test_set(model, test_folder_path)
