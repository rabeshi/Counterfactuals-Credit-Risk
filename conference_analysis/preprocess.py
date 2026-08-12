import pandas as pd
import numpy as np
import os

def load_german_credit():
    path = os.path.join('data', 'German Credit', 'german_credit.csv')
    df = pd.read_csv(path)
    # target: 1 = good, 2 = bad. Convert to binary default=1 for bad
    df['target'] = (df['target'] == 2).astype(int)
    df.rename(columns={'target': 'default'}, inplace=True)
    return df

def load_give_me_credit():
    path = os.path.join('data', 'Give Me Some Credit', 'CreditScoring.csv')
    df = pd.read_csv(path)
    df = df.drop(columns=['Unnamed: 0'])
    df.rename(columns={'SeriousDlqin2yrs': 'default'}, inplace=True)
    # Impute missing values simply
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(df['NumberOfDependents'].median())
    return df

def load_taiwan_credit():
    path = os.path.join('data', 'Taiwan Credit', 'taiwan.csv')
    df = pd.read_csv(path, sep=';')
    # First row is the actual header in some versions; handle both
    if 'ID' not in df.columns:
        df = pd.read_csv(path, sep=';', header=1)
    df = df.drop(columns=['ID'])
    df.rename(columns={'default payment next month': 'default'}, inplace=True)
    return df

if __name__ == '__main__':
    for name, loader in [('German', load_german_credit),
                         ('Give Me Some Credit', load_give_me_credit),
                         ('Taiwan', load_taiwan_credit)]:
        df = loader()
        print(f'=== {name} ===')
        print('Shape:', df.shape)
        print('Default rate:', df['default'].mean())
        print('Columns:', list(df.columns))
        print(df.head(2))
        print()