import pandas as pd
import numpy as np
from keras.models import Model

from keras.layers import LSTM, Dense, Dropout, Bidirectional,Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Bidirectional, Add
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
# 定义 Adam 优化器并设置学习率
optimizers = tf.keras.optimizers.Adam(learning_rate=0.001)

# 定义 ReduceLROnPlateau 学习率调度器
scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=10, verbose=True, epsilon=1e-12)
# 指定資料夾路徑
folder_path = 'C:\\Users\\HONG\\Desktop\\single_bilstm\\test\\distance'

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
    output_file_path_copy = "C:\\Users\\HONG\\Desktop\\single_bilstm\\test\\distance\\{}.csv".format(j)
    # 載入CSV檔案至Pandas DataFrame
    data = pd.read_csv(output_file_path_copy)
    sequence_length = 200  # 序列長度
    num_features = 6  # 特徵數量

    # 初始化空的序列資料和距離列表

    # scaler = MinMaxScaler()
    # data[['gyro_x', 'gyro_y', 'gyro_z', 'acce_x', 'acce_y', 'acce_z', 'Distance']] = scaler.fit_transform(data[['gyro_x', 'gyro_y', 'gyro_z', 'acce_x', 'acce_y', 'acce_z', 'Distance']])
    
    for i in range(0, len(data) - sequence_length, sequence_length):
        distance = np.sum(data['Distance'].values[i:i+sequence_length])
        distances.append(distance)

    # 分割資料成200筆一組的序列
    for i in range(0, len(data) - sequence_length, sequence_length):
        seq = data[['gyro_x', 'gyro_y', 'gyro_z', 'linacce_x', 'linacce_y', 'linacce_z']].values[i:i+sequence_length]  # 根據實際特徵名稱調整
        sequence_data.append(seq)
    print(len(sequence_data))

    # # # 轉換成NumPy陣列
    X = np.array(sequence_data)
    y = np.array(distances)

    # 分割成訓練集和驗證集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(X.shape)

# 不同批次大小的列表
batch_sizes = [32]

# 儲存每個批次大小的損失~
losses = []
# Set the seed for reproducibility
np.random.seed(42)




for batch_size in batch_sizes:

    # Specify the path to save the best model checkpoint
    checkpoint_path = 'checkpoints/best_model_resnet_bilstm_segementation_two_layers_{}.h5'.format(batch_size)
    # Define the ModelCheckpoint callback
    model_checkpoint = ModelCheckpoint(
        checkpoint_path,
        # monitor='val_loss',  # Choose the metric to monitor
        save_best_only=True,  # Save only the best model
        # mode='min',           # 'min' means save the model when the monitored metric is at its minimum
        verbose=1             # Show messages about the checkpointing process
    )


    def build_resnet_block(input_tensor, filters=100, kernel_size=11):
        x = Conv1D(filters, kernel_size, activation='relu', padding='same')(input_tensor)
        x = Conv1D(filters, kernel_size, activation='relu', padding='same')(x)
        x = Conv1D(filters, kernel_size, activation='relu', padding='same')(x)
        shortcut = input_tensor
        x = Add()([x, shortcut])  # Adding a shortcut connection
        return x


    # Input layer
    input_imu = Input(shape=(200, 6))  # 200 frames of IMU data (acceleration, velocity, magnetic field)

    # Feature Detection (ResNet)
    resnet_block = build_resnet_block(input_imu)
    resnet_output = MaxPooling1D(pool_size=2)(resnet_block)
    resnet_output = MaxPooling1D(pool_size=2)(resnet_output)
    resnet_output = MaxPooling1D(pool_size=2)(resnet_output)

    # Trajectory Estimation (Bi-LSTM)
    lstm_output = Bidirectional(LSTM(units=256, return_sequences=True))(resnet_output)
    lstm_output = Dropout(0.5)(lstm_output)  # Add Dropout layer
    # lstm_output = Bidirectional(LSTM(units=256, return_sequences=True))(lstm_output)
    # lstm_output = Dropout(0.5)(lstm_output)  # Add Dropout layer
    lstm_output = Bidirectional(LSTM(units=256, return_sequences=True))(lstm_output)
    lstm_output = Dropout(0.5)(lstm_output)  # Add Dropout layer
    lstm_output = Bidirectional(LSTM(units=256))(lstm_output)

    # Dense layer for final output
    output_layer = Dense(units=1)(lstm_output)

    # Build the model
    model = Model(inputs=input_imu, outputs=output_layer)

    # Compile the model with an appropriate optimizer and loss function
    model.compile(optimizer=optimizers, loss='mean_squared_error', metrics=['accuracy'])

    # Print the model summary
    model.summary()
    history = model.fit(X_train, y_train, epochs=100, batch_size=batch_size, validation_data=(X_val, y_val),callbacks=[model_checkpoint, scheduler])
    model.save('resnet_bilstm_segementation_32_two_layers_.h5')
    # 儲存損失
    losses.append(history.history['loss'])
    # model_name = "1121_{}.h5".format(batch_size)
    # model.save(model_name)
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
    img_path = "plots/resnet_bilstm_segementation_two_layers_{}.png".format(batch_size)
    plt.savefig(img_path)

    # 顯示圖片 (如果需要)
    plt.show()

# 繪製不同批次大小的損失比較圖
plt.figure(figsize=(10, 6))
for i, batch_size in enumerate(batch_sizes):
    plt.plot(losses[i], label=f'Batch Size {batch_size}')
plt.title('Training Loss Comparison for Different Batch Sizes (Resnet_BiLSTM)')
plt.xlabel('Epochs')
plt.ylabel('Loss(m)')
plt.legend()
plt.grid(True)
# 儲存圖片
if not os.path.exists('plots'):
    os.makedirs('plots')
plt.savefig('plots/_resnet_bilstm_segementation_32_two_layers_.png')
plt.show()