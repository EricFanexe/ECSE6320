import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# 读取 CSV 文件
result_file = "./latency_results.csv"
result_data = pd.read_csv(result_file)
result_df = pd.DataFrame(result_data)

# 绘制图表
p1 = sns.relplot(kind="line", data=result_df, x="Buffer_size(KB)", y="Latency(ns)")

# 保存图表到当前文件夹，文件名为 latency_plot.png
plt.savefig("./latency_plot.png")

# 打印数据
print(result_df)
