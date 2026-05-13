# 1. 安装库
!pip install tensorflow scikit-learn pandas matplotlib -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Phase 2: 深度清洗与预处理 ---
df = pd.read_csv('titanic_clean.csv')

# 核心修正：只保留数值列，防止 ValueError
df_numeric = df.select_dtypes(include=[np.number]).dropna()

X = df_numeric.drop('Survived', axis=1)
y = df_numeric['Survived']

# 必须缩放，否则神经网络无法绘图收敛
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Phase 3: 神经网络训练 ---
# 模型 A (浅层)
model_a = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(1, activation='sigmoid')
])
model_a.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 模型 B (深层)
model_b = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_b.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练并记录历史用于绘图
print("开始训练模型...")
history_a = model_a.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=0)
history_b = model_b.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=0)

# --- 生成可视化图表 ---
plt.figure(figsize=(12, 5))

# 绘制准确率曲线
plt.subplot(1, 2, 1)
plt.plot(history_a.history['accuracy'], label='Model A (Shallow)')
plt.plot(history_b.history['accuracy'], label='Model B (Deep)')
plt.title('Model Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 绘制损失曲线
plt.subplot(1, 2, 2)
plt.plot(history_a.history['loss'], label='Model A Loss')
plt.plot(history_b.history['loss'], label='Model B Loss')
plt.title('Model Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# 保存模型
model_a.save("model_3_layers.h5")
model_b.save("model_5_layers.h5")
print("✅ 模型已保存为 .h5 格式")