import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import os

current_dir = os.path.dirname(__file__)
model_names = [
    "pythia-70m-deduped",
    "pythia-1b-deduped",
    "Meta-Llama-3-8B-Instruct",
]
model_name = model_names[0]
target_file = os.path.join(current_dir, "results", f"different_beams_results_{model_name}.pkl")
# check if file exists
if not os.path.exists(target_file):
    raise FileNotFoundError(f"File {target_file} not found")
df = pd.read_pickle(target_file)
amount_of_prompts_tested = df["b"].shape[-1]
print(f"Amount of prompts tested on: {amount_of_prompts_tested}")

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

sns.set_theme(style="whitegrid")
sns.lineplot(x="beams", y="tokens", hue="experiment", data=df_melted)
plt.title(f"{model_name} @{amount_of_prompts_tested}")
sns.despine()
plt.show()

rows_to_plot = [0, 2, 4]
palette = sns.color_palette("husl", len(rows_to_plot))
# set different color for each boxplot
for idx, i in enumerate(rows_to_plot):
    data_boxplot = df.iloc[[i]]
    data_boxplot.index = data_boxplot.index +1
    data_boxplot = data_boxplot.T
    plt.figure()
    sns.boxplot(data=data_boxplot, color=palette[idx])
    sns.despine()

    plt.title(f"{model_name} @{amount_of_prompts_tested}")
    plt.xlabel('Beams')
    plt.ylabel('Token at Generation Step t')
    # plt.show()

# create these three examples in a single boxplot
data_boxplot = df.iloc[rows_to_plot]
data_boxplot.index = data_boxplot.index +1
data_boxplot = data_boxplot.T
plt.figure()
sns.boxplot(data=data_boxplot, palette=palette)
sns.despine()
plt.xlabel('Beams')
plt.ylabel('Token at Generation Step t')
plt.title(f"{model_name} @{amount_of_prompts_tested}")
plt.show()

print("Done")