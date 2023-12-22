import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
from keras.models import load_model
import matplotlib.pyplot as plt  # 引入Matplotlib
import os

import os

# 指定資料夾路徑
folder_path = '/home/mcs/hong/test/distance'

# 使用os模組的listdir函數列出資料夾中的檔案名稱
files = os.listdir(folder_path)
sequence_data = []
distances = []
j=0
# 打印檔案名稱
for file in files:
    j=j+1
    step = "第{}個csv".format(j)
    print(step)
    output_file_path_copy = "/home/mcs/hong/test/distance/{}.csv".format(j)
    # 載入CSV檔案至Pandas DataFrame
    data = pd.read_csv(output_file_path_copy)
    sequence_length = 200  # 序列長度
    num_features = 6  # 特徵數量

    # 初始化空的序列資料和距離列表
    
    for i in range(0, len(data) - sequence_length, sequence_length):
        distance = np.sum(data['Distance'].values[i:i+sequence_length])
        distances.append(distance)

    # 分割資料成200筆一組的序列
    for i in range(0, len(data) - sequence_length, sequence_length):
        seq = data[['gyro_x', 'gyro_y', 'gyro_z', 'acce_x', 'acce_y', 'acce_z']].values[i:i+sequence_length]  # 根據實際特徵名稱調整
        sequence_data.append(seq)
    print(len(sequence_data))

    # # # 轉換成NumPy陣列
    X = np.array(sequence_data)
    y = np.array(distances)

    # 分割成訓練集和驗證集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(X.shape)
# 創建一個新的模型
model = Sequential()
model.add(LSTM(256, input_shape=(sequence_length, num_features)))
model.add(BatchNormalization())  # 在LSTM层后添加批标准化
model.add(Dense(1))

# 編譯模型
model.compile(loss='mean_absolute_error', optimizer='adam')

# 訓練模型，並使用自定義回呼函式
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# 儲存更新後的模型
model.save('test.h5')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.title('Training and Validation Loss')
plt.grid(True)

# 儲存圖片
if not os.path.exists('plots'):
    os.makedirs('plots')
plt.savefig('plots/loss_plot_merge.png')

# 顯示圖片 (如果需要)
plt.show()