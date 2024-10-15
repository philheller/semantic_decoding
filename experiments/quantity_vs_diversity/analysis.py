import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
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

include_ratio = True
include_absolute = True
include_unique = True


animate = False
render_2d = True
preview = False
save_to_files = True

# Set global font properties (if desired; I left default cause same as latex template looks terrible)
# plt.rcParams['font.family'] = 'serif'  # Set the font family
# plt.rcParams['font.size'] = 20 # Set the font size
# plt.rcParams['font.style'] = 'italic'  # Set the font style

# Conversion functions
def cm_to_inch(cm):
    return cm * 0.393701

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

# get all csv files in this dir
os.chdir(os.path.dirname(__file__))
results_files = os.path.join(os.path.dirname(__file__), "results", "model_0")
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
if include_absolute:
    ax.plot_surface(X, Y, Z, cmap=cm.inferno, edgecolor='none', alpha=0.3)
if include_unique:
    ax.plot_surface(X, Y, Z_unique, cmap=cm.terrain, edgecolor='none', alpha=0.95)

if include_ratio:
    # Different alternatives of representing yield below
    # ax.scatter(X, Y, Z_scaled, c='r', marker='o', label="Semantic Tokens / Syntactic Beams [%]", s=20)
    # ax.plot_surface(X, Y, Z_scaled, cmap=cm.coolwarm, edgecolor='none', alpha=0.5)
    # ax.plot_surface(X, Y, Z_scaled, cmap='viridis')

    # norm = plt.Normalize(vmin=np.nanmin(Z_scaled), vmax=np.nanmax(Z_scaled))
    # colors = cm.viridis(norm(Z_scaled))
    # rcount, ccount, _ = colors.shape
    # max_attainable = ax.plot_surface(X, Y, Z_scaled, rcount=rcount, ccount=ccount, facecolors=colors)
    # max_attainable.set_facecolor((1, 0, 0, 0))

    ax.plot_wireframe(X, Y, Z_scaled, color=(1, 0, 0), linewidth=0.5)
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
if include_absolute:
    cbar = plt.colorbar(sm, ax=ax, shrink=0.4, aspect=9, pad=0)
    cbar.set_label('Amount Semantic Tokens')

if include_unique:
    cbar2 = plt.colorbar(sm2, ax=ax, shrink=0.4, aspect=9, pad=0)
    cbar2.set_label('Amount Unique Semantic Tokens')


# wireframe legend
if include_ratio:
    red_box = mpatches.Patch(color='red', label='Semantic Token Yield [%]') # (Amount Semantic Tokens / Amount Syntactic Beams) 
    ax.legend(handles=[red_box], bbox_to_anchor=(0.9, 0.9))

# add title to figure
plt.title(f"Model: {model_name[0]}", y=.98)
# Adjust figure properties
plt.subplots_adjust(left=0.05, right=1, top=1, bottom=0)
# Change figure size using centimeters
fig.set_size_inches(cm_to_inch(30), cm_to_inch(20))  # 25 cm x 20 cm

# always set to between 0 and amount of max beams
ax.set_zlim(0, np.nanmax(Z))

if preview:
    ax.view_init(elev=20, azim=0)
    plt.show()

if animate:
    suffix = ""
    if not include_absolute:
        suffix += "_no_abs"
    if not include_unique:
        suffix += "_no_unique"
    if not include_ratio:
        suffix += "_no_ratio"

    # Sample function for your 3D graph, modify as needed
    def update_graph(num):
        ax.view_init(elev=0, azim=num / 2)  # Rotate the graph for a walkaround view
        return ax

    # Create an animation
    interval = 35
    frames = 720
    ani = FuncAnimation(fig, update_graph, frames=frames, interval=interval)  # 360 frames, delay in ms
    duration = interval * frames  // 1000

    # Save animation as gif
    create_folder("animations")
    save_to_path = os.path.join("animations", f"3d_graph_animated{suffix}_{duration}s.gif")
    ani.save(save_to_path, writer='pillow')

# ax.view_init(elev=0, azim=0)
# plt.savefig('00_elev_00_azim')
# ax.view_init(elev=20, azim=0)
# plt.savefig('20_elev_00_azim')
# plt.show()
plt.close()

# 2d
if render_2d:
    # reset plot
    fig = plt.figure()

    # 1st 2d plot: show all the unique semanti tokens in a line chart with each line being of different amount of beams
    df_z_unique_pivoted = df.pivot_table(index='amount_synt_toks', columns='num_beams', values='num_unique_semantic_tokens')
    base_cmap = sns.color_palette(None, len(df_z_unique_pivoted.columns))
    num_repeats = len(df_z_unique_pivoted.columns) // len(base_cmap) + 1

    # Repeat the colormap
    extended_cmap = base_cmap * num_repeats
    extended_cmap = extended_cmap[:len(df_z_unique_pivoted.columns)]  # Trim to the required length
    # sns.set_theme("notebook")
    sns.lineplot(data=df_z_unique_pivoted, dashes=False, palette=extended_cmap)
    ax = plt.gca()

    # sns.despine()
    # Get the current legend handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # Reverse the order of the handles and labels
    handles.reverse()
    labels.reverse()

    # Set the legend with the reversed order
    plt.legend(handles, labels, title='Number of Beams', ncol=3, loc='upper right', bbox_to_anchor=(1, 1))
    xticks = np.arange(0, 23, 2)  # Generate ticks from 0 to 22 with a step of 2
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(label)) for label in xticks])  # Ensure labels are integers

    # set labels
    plt.xlabel('Amount Syntactic Tokens')
    plt.ylabel('Amount Unique Semantic Tokens')

    # mark highest value in each line (highest amount of unique semantic tokens for each amount of beams)
    highest_indices = []
    for line in df_z_unique_pivoted:
        # Get the data from the line
        x_data = df_z_unique_pivoted.index
        y_data = df_z_unique_pivoted[line]

        # Find the index of the maximum y-value
        max_index = np.argmax(y_data)
        highest_indices.append(max_index)

        # Get the x and y coordinates of the highest point
        x_max = x_data[max_index]
        y_max = y_data.iloc[max_index]

        # Add a marker at the highest point
        ax.plot(x_max, y_max, 'x', markersize=4, label=f'Max: {y_max:.2f}', color='black')

    # set size and title
    fig.set_size_inches(cm_to_inch(30), cm_to_inch(20))
    plt.title(f"Model: {model_name[0]}")

    # adapt 
    plt.subplots_adjust(left=0.06, right=0.97, top=0.95, bottom=0.086)

    # save to file
    suffix = ""
    create_folder("2d_plots")
    save_to_path = os.path.join("2d_plots", f"beams_to_semantic_tokens{suffix}.pdf")
    save_to_path_png = os.path.join("2d_plots", f"beams_to_semantic_tokens{suffix}.png")
    if save_to_files:
        plt.savefig(save_to_path)
        plt.savefig(save_to_path_png)

    if preview:
        plt.show()

    # 2nd 2d plot: show all the 
    # reset
    fig = plt.figure()
    
    # get all the unique semantic tokens for the most common index in highest_indices
    most_common_index = max(set(highest_indices), key=highest_indices.count)

    sns.lineplot(data=df_z_unique_pivoted.iloc[most_common_index], dashes=False)

    # axes labels
    plt.xlabel('Amount Beams')
    plt.ylabel('Amount Unique Semantic Tokens')

    # title
    plt.title(f"Model: {model_name[0]}")
    fig.set_size_inches(cm_to_inch(30), cm_to_inch(20))
    plt.subplots_adjust(left=0.06, right=0.97, top=0.95, bottom=0.086)

    suffix = ""
    create_folder("2d_plots")
    save_to_path = os.path.join("2d_plots", f"growth_semantic_tokens_with_beams{suffix}.pdf")
    save_to_path_png = os.path.join("2d_plots", f"growth_semantic_tokens_with_beams{suffix}.png")
    if save_to_files:
        plt.savefig(save_to_path)
        plt.savefig(save_to_path_png)
    if preview:
        plt.show()

print(df)