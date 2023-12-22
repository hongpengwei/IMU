import matplotlib.pyplot as plt

# 資料
models = ['The model before fine-tuning', 'first fine-tuning', 'second fine-tuning']
speeds = ['slow', 'normal']
precision = [
    [62.175, 55.58],
    [91.6, 83.05],
    [92.7, 93.175]
]

# 繪製長條圖
bar_width = 0.25  # 較窄的長條
index = range(len(models))

fig, ax = plt.subplots()
for i in range(len(speeds)):
    bars = plt.bar([x + i * bar_width for x in index], [item[i] for item in precision], bar_width, label=speeds[i])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom', ha='center')

plt.xlabel('Models')
plt.ylabel('Precision (%)')
plt.title('Precision of Different Models at Various Speeds')
plt.xticks([x + bar_width / 2 for x in index], models)
plt.legend()

plt.tight_layout()
plt.show()
