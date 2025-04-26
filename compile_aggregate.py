import pandas as pd
import glob
import re


agg_df = pd.DataFrame()

for name in glob.glob('./results/fold_*.csv'):
    df = pd.read_csv(name)
    effect = re.match('\.\/results\/fold_[\d]_results_([\w]+)\.csv', name)[1]
    effect_column = ['0NoEffect']*4 + [effect]*4
    df['Effect'] = effect_column
    

    agg_df = pd.concat([agg_df, df], ignore_index=True)

result = agg_df.groupby(['Effect', 'Model']).agg(['mean'])

result.to_csv('./results/aggregate.csv')