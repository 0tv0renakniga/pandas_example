import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Data Setup from the user's context ---
data = {'org': ['org1' for _ in range(12)],
        'count': [int(i) for i in np.random.randint(100, 1000, 12)],
        'type': ['type_0', 'type_1', 'type_2', 'type_3'] * 3,
        'sc': ['sc_0', 'sc_1', 'sc_2', 'sc_3'] * 3,
        'percent': [float(i / 100) for i in np.random.randint(0, 100, 12)]
        }

data2 = {'org': ['org2' for _ in range(24)],
         'count': [int(i) for i in np.random.randint(100, 1000, 24)],
         'type': ['type_a', 'type_b', 'type_c'] * 8,
         'sc': ['sc0', 'sc1', 'sc2', 'sc3'] * 6,
         'percent': [float(i / 100) for i in np.random.randint(0, 100, 24)]
         }

df_org1 = pd.DataFrame(data)
df_org2 = pd.DataFrame(data2)

# Combine the two dataframes into a single dataframe
combined_df = pd.concat([df_org1, df_org2], ignore_index=True)

# Define a function to calculate and plot the weighted averages for a given DataFrame
# Define a function to calculate and plot the weighted averages for a given DataFrame
def plot_weighted_averages_by_type(data_frame, ax, title):
    """
    Calculates weighted averages by 'type' and plots a bar chart on a given subplot axis.

    Args:
        data_frame (pd.DataFrame): The DataFrame to use for calculations.
        ax (matplotlib.axes.Axes): The subplot axis to plot on.
        title (str): The title for the subplot.
    """
    col_type = 'type'

    # 1. Calculate the weighted average for all types in the current DataFrame
    all_types_avg = np.average(data_frame['percent'], weights=data_frame['count'])

    # 2. Calculate the weighted average for each individual type
    type_averages = data_frame.groupby(col_type).apply(
        lambda x: np.average(x['percent'], weights=x['count'])
    )

    # 3. Prepare data for plotting
    unique_types = data_frame[col_type].unique()
    categories = ['All Types'] + list(unique_types)
    values = [all_types_avg] + [type_averages.loc[t] for t in unique_types]

    # 4. Get the total count for all types and for each individual type
    all_types_count = data_frame['count'].sum()
    type_counts = data_frame.groupby(col_type)['count'].sum()
    counts = [all_types_count] + [type_counts.loc[t] for t in unique_types]
    
    # Define colors for the bar plot
    colors = ['#4682B4'] + plt.cm.Set2(range(len(unique_types))).tolist()

    # Create the bar plot on the specified axis
    ax.bar(categories, values, color=colors[:len(categories)])

    # Add labels on top of the bars
    for i, v in enumerate(values):
        ax.text(i, v + 0.02,
                f"per: {v:.2f}\ncount: {counts[i]}",
                ha='center', va='bottom', fontsize=9)

    # 5. Add titles and labels for clarity
    ax.set_title(title, fontsize=12)
    ax.set_ylabel('Weighted Average of Percent', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    
# --- Plotting the three subplots ---
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), sharey=True)
fig.suptitle('Weighted Averages of Percent by Organization', fontsize=18)

# === Top Plot: Weighted average for all organizations, org1, and org2 ===
all_avg = np.average(combined_df['percent'], weights=combined_df['count'])
org1_avg = np.average(df_org1['percent'], weights=df_org1['count'])
org2_avg = np.average(df_org2['percent'], weights=df_org2['count'])

categories_top_plot = ['All Organizations', 'Organization 1', 'Organization 2']
values_top_plot = [all_avg, org1_avg, org2_avg]
counts_top_plot = [combined_df['count'].sum(), df_org1['count'].sum(), df_org2['count'].sum()]

ax1.bar(categories_top_plot, values_top_plot, color=['#4682B4', '#1f77b4', '#ff7f0e'])
ax1.set_title('Overall Weighted Averages', fontsize=12)
ax1.set_ylabel('Weighted Average of Percent', fontsize=10)
ax1.set_ylim(0, 1.0)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

for i, v in enumerate(values_top_plot):
    ax1.text(i, v + 0.02,
             f"per: {v:.2f}\ncount: {counts_top_plot[i]}",
             ha='center', va='bottom', fontsize=9)


# === Second Plot: Weighted average for org1 by type ===
plot_weighted_averages_by_type(df_org1, ax2, 'Organization 1')

# === Third Plot: Weighted average for org2 by type ===
plot_weighted_averages_by_type(df_org2, ax3, 'Organization 2')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig('test_subplots_v2.png')


