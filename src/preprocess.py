import pandas as pd
import numpy as np

import glob

def main():
    csvs  = glob.glob('datasets/*.csv')
    
    dfs = []
    for csv in csvs:
        df = pd.read_csv(csv)
        df['Production'] = np.log(df['Production'])
        
        df = df.dropna()
        df = df[~np.isinf(df['Production'])]
        df = df.set_index('AA')
        
        dfs.append(df)
        
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv("production.csv")

if __name__ == '__main__':
    main()