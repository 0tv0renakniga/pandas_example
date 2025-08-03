import pandas as pd 
import numpy as np

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

df = pd.DataFrame(data)
print(df.head(15))

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

vectorize_me(df)