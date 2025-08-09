import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
data = {
    'obs' : [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3],
    'class': [1,2,3,4,5,1,2,3,4,5,1,2,3,4,5],
    'color': ['blue','red','red','red','yellow',
              'red','blue','blue','blue','yellow',
              'yellow','yellow','yellow','red','blue'],
    'col1': [True,True,'',True,True,
             True,True,'',True,True,
             True,True,'',True,True],
    'col2': [True,False,'',False,True,
             True,True,'',True,True,
             True,True,'',False,True,],
    'col3': [True,False,True,True,True,
             True,True,True,True,True,
             True,True,True,False,True,]

}

#df = pd.DataFrame(data)
#print(df.head(15))

# Loop through each observation
def give_me_the_loop(df):
    for obs_val in df['obs'].unique():
        # Check the conditions for class 2
        class2_condition = ((df['obs'] == obs_val) & (df['class'] == 2))
        class2_col1_true = df.loc[class2_condition, 'col1'].item()
        class2_col2_true = df.loc[class2_condition, 'col2'].item()

        # Check the conditions for class 4
        class4_condition = ((df['obs'] == obs_val) & (df['class'] == 4))
        class4_col1_true = df.loc[class4_condition, 'col1'].item()
        class4_col2_true = df.loc[class4_condition, 'col2'].item()

        # If both conditions are met, update col1 for class 3
        if class2_col1_true and class2_col2_true and class4_col1_true and class4_col2_true:
            df.loc[(df['obs'] == obs_val) & (df['class'] == 3), 'col1'] = True
        else:
            df.loc[(df['obs'] == obs_val) & (df['class'] == 3), 'col1'] = False
    # Display the updated DataFrame
    print("Updated DataFrame:")
    print(df.head(15))
def vectorize_me(df):
    #or use vectorized approach
    
    # 1. Isolate the observations that meet the conditions for class 2
    df_class2 = df[(df['class'] == 2) & (df['col1'] == True) & (df['col2'] == True)][['obs', 'color']].copy()

    # 2. Isolate the observations that meet the conditions for class 4
    df_class4 = df[(df['class'] == 4) & (df['col1'] == True) & (df['col2'] == True)][['obs', 'color']].copy()

    # 3. Merge the two temporary DataFrames on 'obs' to find observations with matching colors
    merged_df = pd.merge(df_class2, df_class4, on='obs', suffixes=('_2', '_4'))

    # 4. Find the observations where the colors for class 2 and class 4 are the same
    obs_to_update = merged_df[merged_df['color_2'] == merged_df['color_4']]['obs'].unique()

    # 5. Use .loc with .isin() to set col1 for class 3
    df.loc[(df['obs'].isin(obs_to_update)) & (df['class'] == 3), 'col1'] = True
    df.loc[(~df['obs'].isin(obs_to_update)) & (df['class'] == 3), 'col1'] = False
    # Display the updated DataFrame
    print("Updated DataFrame:")
    print(df.head(15))

def merger(df):
    # if you need two matching cols where two other cols are true
    # Isolate the observations that meet the conditions for class 2
    df_class2 = df[(df['class'] == 2) & (df['col1'] == True) & (df['col2'] == True)][['obs', 'color']].copy()

    # Isolate the observations that meet the conditions for class 4
    df_class4 = df[(df['class'] == 4) & (df['col1'] == True) & (df['col2'] == True)][['obs', 'color']].copy()

    # Merge the two temporary DataFrames on 'obs' to find observations with matching colors
    merged_df = pd.merge(df_class2, df_class4, on='obs', suffixes=('_2', '_4'))

    # Find the observations where the colors for class 2 and class 4 are the same
    obs_to_update = merged_df[merged_df['color_2'] == merged_df['color_4']]['obs'].unique()

    # Use .loc with .isin() to update col1 for class 3
    df.loc[(df['obs'].isin(obs_to_update)) & (df['class'] == 3), 'col1'] = True
    print("Updated DataFrame:")
    print(df.head(15))

#vectorize_me(df)

data = {'count': [int(i) for i in np.random.randint(100,1000,12)],
        'type':['type_0', 'type_1', 'type_2', 'type_3','type_0', 'type_1', 'type_2', 'type_3','type_0', 'type_1', 'type_2', 'type_3'],
        'percent':[float(i/100) for i in np.random.randint(0,100,12)]
        }

df = pd.DataFrame(data)

all_data = {'count':sum(data['count']),
            'type':['all'],
            'percent':np.average(data['percent'],weights = data['count'])}
print(all_data)

# 1. Calculate the weighted average for all types
all_types_avg = np.average(df['percent'], weights=df['count'])

# 2. Calculate the weighted average for each type using a groupby operation
type_averages = df.groupby('type').apply(
    lambda x: np.average(x['percent'], weights=x['count'])
)

# 3. Prepare data for plotting
categories = [
    'All Types',
    'Type 0',
    'Type 1',
    'Type 2',
    'Type 3'
]

# Create a list of the calculated weighted averages
# We need to ensure the order matches the categories list
values = [all_types_avg] + [type_averages[f'type_{i}'] for i in range(4)]

# 4. Create the bar plot
plt.figure(figsize=(10, 6))
plt.bar(categories, values, color=['skyblue', 'lightgreen', 'salmon', 'gold', 'lightcoral'])

# 5. Add titles and labels for clarity
plt.title('Weighted Average of Percent by Type', fontsize=16)
plt.ylabel('Weighted Average of Percent', fontsize=12)
plt.xlabel('Category', fontsize=12)
plt.ylim(0, 1.0) # Set a consistent y-axis range for percentages
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 6. Add the exact value on top of each bar
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=10)

# Display the plot
plt.tight_layout()
plt.savefig("test_plt.png")


