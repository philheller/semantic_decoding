import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import os

current_dir = os.path.dirname(__file__)
target_file = os.path.join(current_dir, "different_beams_results_pythia-70m-deduped.pkl")
# check if file exists
if not os.path.exists(target_file):
    raise FileNotFoundError(f"File {target_file} not found")
df = pd.read_pickle(target_file)

# shape (amount_of_beams, prompt) and value is amount of tokens
# cast to float
for key, value in df.items():
    df[key] = value.to(torch.float)

df_plot = {
    key: value.mean(-1) for key, value in df.items()
}
df = pd.DataFrame(df_plot)
# find the first lowest value in the columns
percentile = df.quantile(.6)
index_percentile_or_smaller = (df <= percentile).all(axis=1).idxmax()
df = df.iloc[:index_percentile_or_smaller]

df_melted = df.melt(var_name="experiment", value_name="tokens")
df_melted["beams"] = (df_melted.index % df.shape[0]) +1

sns.set_theme(style="white")
sns.lineplot(x="beams", y="tokens", hue="experiment", data=df_melted)

plt.show()

rows_to_plot = [0, 2, 4]
palette = sns.color_palette("husl", len(rows_to_plot))
# set different color for each boxplot
for i in range(len(rows_to_plot)):
    data_boxplot = df.iloc[[i]]
    data_boxplot.index = data_boxplot.index +1
    data_boxplot = data_boxplot.T
    plt.figure()
    sns.boxplot(data=data_boxplot, color=palette[i])
    sns.despine()

    # plt.title('')
    plt.xlabel('Beams')
    plt.ylabel('Tokens')
    plt.show()

print("Done")