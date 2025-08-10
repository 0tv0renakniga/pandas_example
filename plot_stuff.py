import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

def make_plts(df,col_type,plt_name):
    # 1. Calculate the weighted average for all types
    all_types_avg = np.average(df['percent'], weights=df['count'])

    # 2. Calculate the weighted average for each type using a groupby operation
    type_averages = df.groupby(col_type).apply(
        lambda x: np.average(x['percent'], weights=x['count'])
    )
    print(type(type_averages))
    # 3. Prepare data for plotting
    categories = ['All Types']+list(df[col_type].unique())

    # Create a list of the calculated weighted averages
    # We need to ensure the order matches the categories list
    values = [all_types_avg] + [type_averages[i] for i in df[col_type].unique()]
    colors=['skyblue', 'lightgreen', 'salmon', 'gold', 'lightcoral','blue', 'green', 'red', 'purple', 'orange', 'brown']
    # 4. Create the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(categories, values, color=colors[:len(categories)])

    # 5. Add titles and labels for clarity
    plt.title(plt_name, fontsize=16)
    plt.ylabel('Weighted Average of Percent', fontsize=12)
    plt.xlabel('Category', fontsize=12)
    plt.ylim(0, 1.0) # Set a consistent y-axis range for percentages
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    def make_bar_labels(categories):
        labels = []
        for i,val in enumerate(categories):
            if i == 0:
                print(f"per: {all_types_avg:.2f}\n dod:{df['count'].sum()}")
                labels.append([all_types_avg,f"per: {all_types_avg:.2f}\n dod:{df['count'].sum()}"])
            else:
                print(f"per: {type_averages[val]:.2f}\ndod:{df[df[col_type]==val]['count'].sum()}")
                labels.append([type_averages[val],f"per: {all_types_avg:.2f}\n dod:{df[df[col_type]==val]['count'].sum()}"])
        return(labels)
    labels = make_bar_labels(categories)
    # 6. Add the exact value on top of each bar
    #for i, v in enumerate(values):
    for i, v in enumerate(labels):
        plt.text(i, v[0] + 0.02, v[1], ha='center', va='bottom', fontsize=10)
        #plt.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=10)

    # Display the plot
    plt.tight_layout()
    plt.savefig(f"{plt_name}.png")

#make_plts(dfs,'org')
# faux data
data = {'org': ['org1' for i in range(24)],
        'type': [f'type_0' for i in range(8)] +
                [f'type_1' for i in range(8)] +
                [f'type_2' for i in range(8)],
        'st': [f'stype_{i}' for i in ['a','a','b','b','c','c','d','d']] +
              [f'stype_{i}' for i in ['a','a','b','b','c','c','d','d']] +
              [f'stype_{i}' for i in ['a','a','b','b','c','c','d','d']],
        'sst' : [f'stype_{i}' for i in ['a0','a0','b0','b0','c0','c0','d0','d0']] +
              [f'stype_{i}' for i in ['a1','a1','b1','b1','c1','c1','d1','d1']] +
              [f'stype_{i}' for i in ['a2','a2','b2','b2','c2','c2','d2','d2']],
        'count': [int(i) for i in np.random.randint(100,1000,24)],
        'sc': ['sc_0', 'sc_1', 'sc_0', 'sc_1','sc_0', 'sc_1', 'sc_0', 'sc_1',
               'sc_0', 'sc_1', 'sc_0', 'sc_1','sc_0', 'sc_1', 'sc_0', 'sc_1',
               'sc_0', 'sc_1', 'sc_0', 'sc_1','sc_0', 'sc_1', 'sc_0', 'sc_1',
               ],
        'percent':[float(i/100) for i in np.random.randint(0,100,24)]
        }

data2 = {'org': ['org2' for i in range(24)],
        'count': [int(i) for i in np.random.randint(100,1000,24)],
        'type': ['type_w', 'type_x', 'type_y', 'type_w','type_x', 'type_y', 'type_w', 'type_x','type_y', 'type_w', 'type_x', 'type_y','type_w', 'type_x', 'type_y', 'type_w','type_x', 'type_y', 'type_w', 'type_x','type_y', 'type_w', 'type_x', 'type_y'],
        'sc': ['sc0', 'sc1', 'sc2', 'sc3','sc0', 'sc1', 'sc2', 'sc3','sc0', 'sc1', 'sc2', 'sc3','sc0', 'sc1', 'sc2', 'sc3','sc0', 'sc1', 'sc2', 'sc3','sc0', 'sc1', 'sc2', 'sc3'],
        'percent':[float(i/100) for i in np.random.randint(0,100,24)]
        }

df = pd.DataFrame(data)
df2 = pd.DataFrame(data2)
concat_cols = list(set(df.columns)&set(df2.columns))
dfs =pd.concat([df[concat_cols],df2[concat_cols]],ignore_index=True)
print(dfs[dfs['sc'].str.contains('0',case=False)])

make_plts(dfs,'org',"first_plt")
make_plts(df,'type',"sec_plt")
make_plts(df2,'type','third_plt')

make_plts(df,'st','fourth_plt')

#maybe kinda slick
#montage first_plt.png sec_plt.png third_plt.png fourth_plt.png -geometry +5+5 -tile 2x2 combined_plots.png

