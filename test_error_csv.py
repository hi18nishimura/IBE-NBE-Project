import pandas as pd

path = "/workspace/results/peephole_basemodel/error_stats_summary.csv"
df = pd.read_csv(path)
print(f"peephole（変位、平均誤差）：{(df['mean']**2)[:3].sum()**0.5}")
print(f"peephole（変位、中央値誤差）：{(df['median']**2)[:3].sum()**0.5}")
print(f"peephole（垂直応力、平均誤差）：{(df['mean'])[6:9].mean()}")
print(f"peephole（せん断応力、平均誤差）：{(df['median'])[9:11].mean()}")
print("--------------------------------------------------")

path = "/workspace/results/GNN_Global_fc_numlayer3_weight0_alpha8/error_stats_summary.csv"
df = pd.read_csv(path)
#print(df)
print(f"GAT（変位、平均誤差）：{(df['mean']**2)[:3].sum()**0.5}")
print(f"GAT（変位、中央値誤差）：{(df['median']**2)[:3].sum()**0.5}")
print(f"GAT（垂直応力、平均誤差）：{(df['mean'])[6:9].mean()}")
print(f"GAT（せん断応力、平均誤差）：{(df['median'])[9:11].mean()}")
print("--------------------------------------------------")

path = "/workspace/results/GNN_Global_fc_numlayer3_weight_time_alpha8/error_stats_summary.csv"
df = pd.read_csv(path)
#print(df)
print(f"GAT L_time（変位、平均誤差）：{(df['mean']**2)[:3].sum()**0.5}")
print(f"GAT L_time（変位、中央値誤差）：{(df['median']**2)[:3].sum()**0.5}")
print(f"GAT L_time（垂直応力、 平均誤差）：{(df['mean'])[6:9].mean()}")
print(f"GAT L_time（せん断応力、中央値誤差）：{(df['median'])[9:11].mean()}")
print("--------------------------------------------------")


path = "/workspace/results/GNN_Global_fc_numlayer3_weight_oka_alpha8/error_stats_summary.csv"
df = pd.read_csv(path)
#print(df)
print(f"GAT L_oka（変位、平均誤差）：{(df['mean']**2)[:3].sum()**0.5}")
print(f"GAT L_oka（変位、中央値誤差）：{(df['median']**2)[:3].sum()**0.5}")
print(f"GAT L_oka（垂直応力、平均誤差）：{(df['mean'])[6:9].mean()}")
print(f"GAT L_oka（せん断応力、中央値誤差）：{(df['median'])[9:11].mean()}")
print("--------------------------------------------------")


