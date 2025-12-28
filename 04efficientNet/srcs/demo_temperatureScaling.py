import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, log_loss

class TemperatureScaling:
    """Temperature scaling for probability calibration"""
    
    def __init__(self):
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def fit(self, logits, labels):
        """Fit temperature scaling on validation set"""
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval():
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval)
        return self.temperature.item()
    
    def predict_proba(self, logits):
        """Get calibrated probabilities"""
        return torch.softmax(logits / self.temperature, dim=1)


# 模拟模型输出和真实标签
# 生成样本数据：1000个样本，3个类别
np.random.seed(42)
torch.manual_seed(42)

# 模拟未校准的模型输出（logits）
logits = torch.randn(1000, 3) * 2  # 增加方差使概率更极端

# 模拟真实标签
labels = torch.randint(0, 3, (1000,))

# 划分训练集和验证集（这里仅用于演示）
# 注意：在实际应用中，温度缩放的参数应在验证集上学习，而不是在测试集上
logits_train, logits_val = logits[:800], logits[800:]
labels_train, labels_val = labels[:800], labels[800:]

# 计算原始模型的概率和性能
original_probs = torch.softmax(logits_val, dim=1)
original_predictions = torch.argmax(original_probs, dim=1)
original_accuracy = accuracy_score(labels_val.numpy(), original_predictions.numpy())
original_loss = log_loss(labels_val.numpy(), original_probs.numpy())

print(f"原始模型 - 准确率: {original_accuracy:.4f}, 交叉熵损失: {original_loss:.4f}")

# 使用温度缩放进行校准
calibrator = TemperatureScaling()
temperature = calibrator.fit(logits_train, labels_train)
print(f"学习到的温度参数: {temperature:.4f}")

# 获取校准后的概率
calibrated_probs = calibrator.predict_proba(logits_val)
calibrated_predictions = torch.argmax(calibrated_probs, dim=1)
calibrated_accuracy = accuracy_score(labels_val.numpy(), calibrated_predictions.numpy())
calibrated_loss = log_loss(labels_val.numpy(), calibrated_probs.detach().numpy())

print(f"校准后模型 - 准确率: {calibrated_accuracy:.4f}, 交叉熵损失: {calibrated_loss:.4f}")

# 比较校准前后的概率分布差异
print("\n校准前后的概率分布示例（前5个样本）:")
for i in range(5):
    print(f"样本 {i+1}:")
    print(f"  原始概率: {original_probs[i].detach().numpy()}")
    print(f"  校准概率: {calibrated_probs[i].detach().numpy()}")
    print(f"  真实标签: {labels_val[i]}")
