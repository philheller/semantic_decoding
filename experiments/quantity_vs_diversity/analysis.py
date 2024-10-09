import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import matplotlib.patches as mpatches
# from matplotlib.colors import LightSource

import re


#### Notes ####
# This graph shows different aspects of semantic tokens and syntactic tokens.
# Three major values have been aggregated:
# 1. Amount of semantic tokens (terrain cmap)
# 2. Amount of unique semantic tokens (inferno cmap)
# 3. Amount of semantic tokens divided by the amount of syntactic hyps (amount of beams) [-> potential of semantic tokens]
#      which is essentially the yield of semantic tokens per syntactic token and therefore
#      measured in percentage. (red wireframe)
# if source files for data are availbale, adapt the folder path in `results_files` variable below


# get all csv files in this dir
os.chdir(os.path.dirname(__file__))
results_files = os.path.join(os.path.dirname(__file__), "results")
files = [os.path.join(results_files, f) for f in os.listdir(results_files) if f.endswith(".csv") and not "generated" in f]

# find all in string before "_noun_chunks" or "_ner"
model_name = re.findall(r'(.+)_noun_chunks|(.+)_ner', os.path.basename(files[0]))[0]

amount_toks_gen = [
    int(re.findall(r'\d+', f)[-1]) for f in files if re.findall(r'\d+', f)
]

# read all csv files
dfs = [pd.read_csv(f) for f in files]
merged_df = pd.DataFrame()

# for df, num in zip(dfs, amount_toks_gen):
#     # Rename columns with the extracted number
#     df.columns = [f"{col}_{num}" for col in df.columns]
#     # Concatenate the DataFrame to the merged DataFrame
#     merged_df = pd.concat([merged_df, df], axis=1)

for df, num in zip(dfs, amount_toks_gen):
    # Add a new column with the extracted number
    df['amount_synt_toks'] = num
    # Concatenate the DataFrame to the merged DataFrame
    merged_df = pd.concat([merged_df, df], ignore_index=True)

df = merged_df
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='raise')


# find relative amount of semantic tokens vs beams (technically reachable semantic tokens)
df["sem_to_beams"] = df["num_semantic_tokens"] / df["num_beams"]
df["sem_to_beams_scaled"] = df["sem_to_beams"] * df["num_semantic_tokens"].max()



# 3d
x = df['num_beams']
y = df['amount_synt_toks']
z = df['num_semantic_tokens']
z_scaled = df['sem_to_beams_scaled']
z_unique = df['num_unique_semantic_tokens']


# Create a grid (mesh) for x and y (beams and syntactic tokens)
X, Y = np.meshgrid(np.unique(x), np.unique(y))

# Interpolate Z values (semantic tokens) and Z_scaled values over the grid
Z = df.pivot_table(index='amount_synt_toks', columns='num_beams', values='num_semantic_tokens').reindex(index=np.unique(y), columns=np.unique(x)).values
Z_scaled = df.pivot_table(index='amount_synt_toks', columns='num_beams', values='sem_to_beams_scaled').reindex(index=np.unique(y), columns=np.unique(x)).values
Z_unique = df.pivot_table(index='amount_synt_toks', columns='num_beams', values='num_unique_semantic_tokens').reindex(index=np.unique(y), columns=np.unique(x)).values

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create surface plot for num_semantic_tokens
ax.plot_surface(X, Y, Z, cmap=cm.inferno, edgecolor='none', alpha=0.3)
ax.plot_surface(X, Y, Z_unique, cmap=cm.terrain, edgecolor='none', alpha=0.95)

# Different alternatives of representing yield below
# ax.scatter(X, Y, Z_scaled, c='r', marker='o', label="Semantic Tokens / Syntactic Beams [%]", s=20)
# ax.plot_surface(X, Y, Z_scaled, cmap=cm.coolwarm, edgecolor='none', alpha=0.5)
# ax.plot_surface(X, Y, Z_scaled, cmap='viridis')

# norm = plt.Normalize(vmin=np.nanmin(Z_scaled), vmax=np.nanmax(Z_scaled))
# colors = cm.viridis(norm(Z_scaled))
# rcount, ccount, _ = colors.shape
# max_attainable = ax.plot_surface(X, Y, Z_scaled, rcount=rcount, ccount=ccount, facecolors=colors)
# max_attainable.set_facecolor((1, 0, 0, 0))

ax.plot_wireframe(X, Y, Z_scaled, color=(1, 0, 0))
# ax.contour3D(X, Y, Z_scaled, 50, cmap='plasma')


# Labels
ax.set_xlabel('Amount Beams')
ax.set_ylabel('Amount Synt Toks')
ax.set_zlabel('Amount Sem Toks')
# ax.text2D(0.05, 0.95, "Semantic Tokens / Syntacitc Beam [%]", transform=ax.transAxes, color="red")

# legend
# cmap legend
cmap = cm.inferno
norm = plt.Normalize(vmin=np.nanmin(Z), vmax=np.nanmax(Z))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cmap2 = cm.terrain
norm2 = plt.Normalize(vmin=np.nanmin(Z_unique), vmax=np.nanmax(Z_unique))
sm2 = plt.cm.ScalarMappable(cmap=cmap2, norm=norm2)
sm2.set_array([])

# Add the colorbar to the plot
cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=8)
cbar.set_label('Amount Semantic Tokens')

cbar2 = plt.colorbar(sm2, ax=ax, shrink=0.5, aspect=8)
cbar2.set_label('Amount Unique Semantic Tokens')


# wireframe legend
red_box = mpatches.Patch(color='red', label='Semantic Token Yield [%]') # (Amount Semantic Tokens / Amount Syntactic Beams) 
ax.legend(handles=[red_box])

# add title to figure
plt.title(f"Model: {model_name[0]}")
ax.view_init(elev=0, azim=0)
plt.show()



# 3d
# x = df['num_beams']
# y = df['amount_synt_toks']
# z = df['num_semantic_tokens']

# # Create a grid (mesh) for x and y (beams and syntactic tokens)
# X, Y = np.meshgrid(np.unique(x), np.unique(y))

# # Interpolate Z values (semantic tokens) over the grid
# Z = df.pivot_table(index='amount_synt_toks', columns='num_beams', values='num_semantic_tokens').reindex(index=np.unique(y), columns=np.unique(x)).values

# # Plotting
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Create surface plot with hillshading
# ax.plot_surface(X, Y, Z, cmap=cm.terrain, edgecolor='none')

# # Labels
# ax.set_xlabel('Num Beams')
# ax.set_ylabel('Amount Synt Toks')
# ax.set_zlabel('Num Semantic Tokens')

# # Show plot
# plt.show()


# 2d
# find out how many "num_semantic_tokens" are 0 for each number
# df_non_empty = df[df['num_semantic_tokens'] > 0]
# dfg = df.groupby('amount_synt_toks')
# dfg_non_empty = df_non_empty.groupby('amount_synt_toks')

# # 1. num semantic tokens
# # all
# se_num_semantic_tokens = dfg.apply(
#     lambda x: x['num_semantic_tokens'].mean()
# ).rename("sem_toks")

# # only non-empty
# se_num_semantic_tokens_non_empty = dfg_non_empty.apply(
#     lambda x: x['num_semantic_tokens'].mean()
# ).rename("sem_toks_n")

# # only empty
# se_num_semantic_tokens_empty = dfg.apply(
#     lambda x: x['num_semantic_tokens'].apply(lambda x: 1 if x == 0 else 0).mean() * 200
# ).rename("sem_toks_e")

# # 2. unique num semantic tokens
# # all
# se_num_unique_semantic_tokens = dfg.apply(
#     lambda x: x['num_unique_semantic_tokens'].mean()
# ).rename("u_sem_toks")

# # non empty
# se_num_unique_semantic_tokens_non_empty = dfg_non_empty.apply(
#     lambda x: x['num_unique_semantic_tokens'].mean()
# ).rename("u_sem_toks_n")

# df_res = pd.concat([
#     se_num_semantic_tokens,
#     se_num_semantic_tokens_non_empty,
#     se_num_semantic_tokens_empty,
#     se_num_unique_semantic_tokens,
#     se_num_unique_semantic_tokens_non_empty
# ], axis=1)

# # now, use this in seaborn lineplot
# df_plot = df_res[['sem_toks', 'sem_toks_e', 'u_sem_toks']]

# # set custom colors 
# # 2 tones of blue, 2 tones of red
# colors = [
#     # blue 1
#     (0.121, 0.46, 0.7),
#     # blue 2
#     (0.65, 0.8, 0.89),
#     # red 1
#     (0.7, 0.09, 0.17),
#     # red 2
#     (0.98, 0.6, 0.6)
# ]
# custom_names = [
#     "Amount of semantic tokens",
#     "Amount of empty semantic tokens",
#     "Amount of unique semantic tokens"
# ]
# sns.lineplot(data=df_plot, palette=colors)
# plt.show()

print(df)