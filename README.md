# 动物图像分类

华中科技大学 计算机科学与技术学院 人工智能导论 深度学习实践作业

# ⚠️ 注意
由于本代码写于几个月之前，可能会由于版本更新导致无法直接运行，作者目前还在期末月，实在来不及修改，欢迎联系作者或者在 Issues 里面分享自己的Debug或改进方法！我在寒假会抽时间修改的！🥳

**作业要求：**
- **深度学习方法**：以 Python 作为主要编程语言，使用 PyTorch/TensorFlow/MindSpore 框架，构建卷积神经网络模型 (如 ResNet50、VGG)，实现对动物图像的分类任务。
- **目标**：
  1. 选择几种常见的动物类别(如猫、狗、鸡、马等)，作为识别目标。
  2. 收集并建立这些动物的图像数据集。
  3. 经过数据预处理后，利用卷积层提取图像特征，通过训练生成一个高精度的分类模型。
- **提交内容**：
  - 文档：产生式系统与深度学习这两种方法分析比较(算法原理、步骤、代码实现)。
  - 代码：实现 CNN 模型和动物识别的完整代码。

**作业形式**：
- 文件打包名称：`动物识别 + 姓名`
- 提交邮箱：`ai_2019@sohu.com`
- 截止时间：课程结束前。

**目的**：
通过对比产生式系统与深度学习方法，分析两者在动物图像分类任务中的优劣。

---

## 🖻 数据集
由于GitHub上传文件大小等限制，我没有将所用的数据集（应该放在`datasets/`文件夹中）上传到GitHub，请大家自行下载数据集。提供两种数据集供大家选择：    
[Kaggle 动物图像数据集 (Animals10)](https://www.kaggle.com/datasets/alessiocorrado99/animals10)    
[Kaggle 猫狗数据集 (Cats vs Dogs)](https://www.kaggle.com/datasets/sreetejadusi/cats-vs-dogs)

---

## 📁 文件组织结构
```
datasets/
    ├── train/
    │   ├── class_0/       # 动物类别 0 的训练图片
    │   ├── class_1/       # 动物类别 1 的训练图片
    │   └── ...
    ├── val/
        ├── class_0/       # 动物类别 0 的验证图片
        ├── class_1/       # 动物类别 1 的验证图片
        └── ...
    └── test/
        ├── class_0/       # 动物类别 0 的验证图片
        ├── class_1/       # 动物类别 1 的验证图片
        └── ...
models/
    ├── best_model.pth     # 训练好的最佳模型
    └── final_model.pth    # 最终保存的模型
logs/
    └── train_log_YYYYMMDD_HHMMSS.txt  # 每次训练的日志文件
scripts/
    ├── split_data.py      # 数据集划分脚本
    ├── train.py           # 训练脚本
    ├── train_continue.py  # 继续训练脚本
    ├── test.py            # 单张图片测试脚本
    ├── test_all.py        # 批量测试脚本
    └── test_score.py      # 计算测试集得分的脚本
```
## ▶︎ 使用说明

### 🖼︎ 数据集准备
将下载的数据集放入 `datasets/` 文件夹中，并确保按照以下结构组织：
```
datasets/    
    ├── raw/               # 原始未处理数据集（可选）    
    ├── train/             # 训练集（由 split_data.py 生成）    
    └── val/               # 验证集（由 split_data.py 生成）    
```      
运行 `split_data.py` 脚本以按比例划分训练集和验证集。

### ✒︎ 训练
运行 `train.py` 开始模型训练。    
训练中断后，可使用 `train_continue.py` 继续训练。

**产生的模型文件：**    
`model_epoch_num.pth`：第 num 次训练的模型。    
`best_model.pth`：训练过程中保存的最佳模型。    
`final_model.pth`：训练结束时保存的最终模型。


### 💯 测试
使用 `test.py` 测试单张图片的分类结果。

使用 `test_all.py` 对整个文件夹的图片进行批量预测。

使用 `test_score.py` 计算分类模型在测试集上的准确率。 
