import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  # 引入Matplotlib
import os

# 載入CSV檔案至Pandas DataFrame
data = pd.read_csv('distance/1.csv')

# 將資料整理成模型可接受的格式
sequence_length = 200  # 序列長度
num_features = 6  # 特徵數量

# 初始化序列資料和距離列表
sequence_data = []
distances = []

for i in range(0, len(data) - sequence_length, sequence_length):
    distance = np.sum(data['Distance'].values[i:i+sequence_length])
    distances.append(distance)
    seq = data[['gyro_x', 'gyro_y', 'gyro_z', 'acce_x', 'acce_y', 'acce_z']].values[i:i+sequence_length]
    sequence_data.append(seq)

# 轉換成NumPy陣列
X = np.array(sequence_data)
y = np.array(distances)

# 分割成訓練集和驗證集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 不同批次大小的列表
batch_sizes = [16,32,64,128,256]

# 儲存每個批次大小的損失
losses = []

for batch_size in batch_sizes:
    # 創建一個新的模型
    model = Sequential()
    model.add(LSTM(256, input_shape=(sequence_length, num_features)))
    model.add(Dense(1))
    
    # 編譯模型
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # 訓練模型
    history = model.fit(X_train, y_train, epochs=50, batch_size=batch_size, validation_data=(X_val, y_val))
    
    # 儲存損失
    losses.append(history.history['loss'])
    model_name = "lstm_{}.h5".format(batch_size)
    model.save(model_name)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error  (m)')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    # 儲存圖片
    if not os.path.exists('plots'):
        os.makedirs('plots')
    img_path = "plots/loss_plot_{}_50.png".format(batch_size)
    plt.savefig(img_path)

    # 顯示圖片 (如果需要)
    plt.show()

# 繪製不同批次大小的損失比較圖
plt.figure(figsize=(10, 6))
for i, batch_size in enumerate(batch_sizes):
    plt.plot(losses[i], label=f'Batch Size {batch_size}')
plt.title('Training Loss Comparison for Different Batch Sizes')
plt.xlabel('Epochs')
plt.ylabel('Loss(m)')
plt.legend()
plt.grid(True)
# 儲存圖片
if not os.path.exists('plots'):
    os.makedirs('plots')
plt.savefig('plots/loss_plot_all.png')
plt.show()




# 訓練模型
# history_16 = model_16.fit(X_train, y_train, epochs=60, batch_size=32, validation_data=(X_val, y_val))
# history_8 = model_8.fit(X_train, y_train, epochs=60, batch_size=8, validation_data=(X_val, y_val))
# history_4 = model_4.fit(X_train, y_train, epochs=80, batch_size=4, validation_data=(X_val, y_val))
# history_2 = model_2.fit(X_train, y_train, epochs=100, batch_size=2, validation_data=(X_val, y_val))
# history = model.fit(X_train, y_train, epochs=100, batch_size=1, validation_data=(X_val, y_val))

# 儲存模型
# model_16.save('lstm_16.h5')
# plt.plot(history_16.history['loss'], label='train')
# plt.plot(history_16.history['val_loss'], label='validation')
# plt.xlabel('Epoch')
# plt.ylabel('Mean Absolute Error')
# plt.legend()
# plt.title('Training and Validation Loss')
# plt.grid(True)

# # 儲存圖片
# if not os.path.exists('plots'):
#     os.makedirs('plots')
# plt.savefig('plots/loss_plot_16.png')

# # 顯示圖片 (如果需要)
# plt.show()

# 儲存模型
# model_8.save('lstm_8.h5')
# plt.plot(history_8.history['loss'], label='train')
# plt.plot(history_8.history['val_loss'], label='validation')
# plt.xlabel('Epoch')
# plt.ylabel('Mean Absolute Error')
# plt.legend()
# plt.title('Training and Validation Loss')
# plt.grid(True)

# # 儲存圖片
# if not os.path.exists('plots'):
#     os.makedirs('plots')
# plt.savefig('plots/loss_plot_8.png')

# # 顯示圖片 (如果需要)
# plt.show()

# 儲存模型
# model_4.save('lstm_4.h5')
# plt.plot(history_4.history['loss'], label='train')
# plt.plot(history_4.history['val_loss'], label='validation')
# plt.xlabel('Epoch')
# plt.ylabel('Mean Absolute Error')
# plt.legend()
# plt.title('Training and Validation Loss')
# plt.grid(True)

# # 儲存圖片
# if not os.path.exists('plots'):
#     os.makedirs('plots')
# plt.savefig('plots/loss_plot_4_2.png')

# # 顯示圖片 (如果需要)
# plt.show()

# # 儲存模型
# model_2.save('lstm_2.h5')
# plt.plot(history_2.history['loss'], label='train')
# plt.plot(history_2.history['val_loss'], label='validation')
# plt.xlabel('Epoch')
# plt.ylabel('Mean Absolute Error')
# plt.legend()
# plt.title('Training and Validation Loss')
# plt.grid(True)

# # 儲存圖片
# if not os.path.exists('plots'):
#     os.makedirs('plots')
# plt.savefig('plots/loss_plot_2_100.png')

# # 顯示圖片 (如果需要)
# plt.show()

# # 儲存模型
# model.save('lstm_1.h5')
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='validation')
# plt.xlabel('Epoch')
# plt.ylabel('Mean Absolute Error')
# plt.legend()
# plt.title('Training and Validation Loss')
# plt.grid(True)

# # 儲存圖片
# if not os.path.exists('plots'):
#     os.makedirs('plots')
# plt.savefig('plots/loss_plot_1_100.png')

# # 顯示圖片 (如果需要)
# plt.show()