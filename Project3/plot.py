import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

result_file = "SSD_Performance.csv"
result_data = pd.read_csv(result_file)
result_df = pd.DataFrame(result_data)

p1 = sns.lmplot(data=result_df, x="Latency avg (usec)", y="IOPS(k)", hue="Block Size(KiB)", truncate=True)
plt.ylim(-10, 150)
plt.savefig("Latency_vs_IOPS_by_Block_Size.png")

p2 = sns.lmplot(data=result_df, x="Latency avg (usec)", y="IOPS(k)", hue="Percentage of Write(%)", truncate=True)
plt.ylim(-10, 150)
plt.savefig("Latency_vs_IOPS_by_Write_Percentage.png")

p3 = sns.lmplot(data=result_df, x="Latency avg (usec)", y="IOPS(k)", hue="IO Depth", truncate=True)
plt.ylim(-10, 150)
plt.savefig("Latency_vs_IOPS_by_IO_Depth.png")

plt.show()
