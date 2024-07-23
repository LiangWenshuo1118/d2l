import numpy as np
import matplotlib.pyplot as plt

def load_data(file_name):
    data = np.loadtxt(file_name, skiprows=1)
    epochs = data[:, 0]
    train_loss = data[:, 1]
    test_loss = data[:, 2]
    test_accuracy = data[:, 3]
    return epochs, train_loss, test_loss, test_accuracy

# 加载所有数据
epochs_cnn, train_loss_cnn, test_loss_cnn, test_accuracy_cnn = load_data('log/CNN.txt')
epochs_vgg, train_loss_vgg, test_loss_vgg, test_accuracy_vgg = load_data('log/VGG_like.txt')
epochs_vgg2, train_loss_vgg2, test_loss_vgg2, test_accuracy_vgg2 = load_data('log/VGG_like_dropout.txt')

# 创建图形和轴对象
fig, ax1 = plt.subplots(figsize=(9, 6))

# 创建双坐标轴
ax2 = ax1.twinx()

# 绘制训练损失
ax1.plot(epochs_cnn, train_loss_cnn, label='Train Loss CNN', color='blue', linestyle='--')
ax1.plot(epochs_vgg, train_loss_vgg, label='Train Loss VGG-like', color='orange', linestyle=':')
ax1.plot(epochs_vgg2, train_loss_vgg2, label='Train Loss VGG-like Dropout', color='green', linestyle='-.')

# 绘制测试损失
ax1.plot(epochs_cnn, test_loss_cnn, label='Test Loss CNN', color='blue', linestyle='-.')
ax1.plot(epochs_vgg, test_loss_vgg, label='Test Loss VGG-like', color='orange', linestyle='-.')
ax1.plot(epochs_vgg2, test_loss_vgg2, label='Test Loss VGG-like Dropout', color='green', linestyle=':')

# 绘制测试准确性
ax2.plot(epochs_cnn, test_accuracy_cnn, label='Test Accuracy CNN', color='blue', linestyle='-')
ax2.plot(epochs_vgg, test_accuracy_vgg, label='Test Accuracy VGG-like', color='orange', linestyle='-')
ax2.plot(epochs_vgg2, test_accuracy_vgg2, label='Test Accuracy VGG-like Dropout', color='green', linestyle='-')

# 设置坐标轴标签和图标题
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='black')
ax2.set_ylabel('Test Accuracy', color='black')

# 设置图例，放置在图形外部的上方
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

plt.savefig("results.png", dpi = 600)

