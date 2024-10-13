import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# 读取数据
result_file = "./miss_out.csv"
result_data = pd.read_csv(result_file)
result_df = pd.DataFrame(result_data)

# 创建图表并保存
p1 = sns.lmplot(data=result_df, x="cache_miss_ratio(%)", y="runtime(ms)")
p1.savefig("cache_miss_vs_runtime.png")  # 保存第一张图

p2 = sns.lmplot(data=result_df, x="dTLB_miss_ratio(%)", y="runtime(ms)")
p2.savefig("dTLB_miss_vs_runtime.png")  # 保存第二张图

# 打印数据框
print(result_df)
