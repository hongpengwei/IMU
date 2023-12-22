import os
import pandas as pd

# 假設多個 CSV 檔案存放在指定目錄下
directory_path = 'C:\\Users\\HONG\\Desktop\\test'

# 取得該目錄下的所有 CSV 檔案名稱
csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

# 初始化一個空的 DataFrame 來存放合併後的資料
merged_df = pd.DataFrame()

# 逐一讀取每個 CSV 檔案，並合併到 merged_df
for csv_file in csv_files:
    csv_file_path = os.path.join(directory_path, csv_file)
    df = pd.read_csv(csv_file_path)
    merged_df = pd.concat([merged_df, df], ignore_index=True)
output_file_path = 'C:\\Users\\HONG\\Desktop\\test\\path_to_output_merged_csv_file.csv'
merged_df.to_csv(output_file_path, index=False)