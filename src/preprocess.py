import pandas as pd
import numpy as np

import glob

def main():
    csvs  = glob.glob('datasets/*.csv')
    
    dfs = []
    missing_dfs = []
    for csv in csvs:
        df = pd.read_csv(csv)
        df['Production'] = np.log(df['Production'])
        df = df.set_index('AA')
        
        df = df.dropna()
        
        missing_df = df[np.isinf(df['Production'])]
        missing_dfs.append(missing_df)
        
        df = df[~np.isinf(df['Production'])]
        dfs.append(df)
        
    df = pd.concat(dfs, ignore_index=False)
    df.to_csv("production.csv")
    
    missing_df = pd.concat(missing_dfs, ignore_index=False)
    missing_df.to_csv("missing.csv")

if __name__ == '__main__':
    main()