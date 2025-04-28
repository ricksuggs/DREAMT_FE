import pandas as pd
import glob
import re

agg_df = pd.DataFrame()

for name in glob.glob('./results/run2/fold_*.csv'):
    df = pd.read_csv(name)
    effect = re.match('\.\/results\/run2\/fold_[\d]_results_([\w]+)\.csv', name)[1]
    effect_column = ['0NoEffect']*4 + [effect]*4
    df['Effect'] = effect_column

    agg_df = pd.concat([agg_df, df], ignore_index=True)

# Define function to combine mean and std
def combine_mean_std(x):
    """Calculates mean ± std for a pandas Series."""
    mean = x.mean().round(3)
    std = x.std().round(3)
    # Handle potential NaN in std (e.g., if only one sample)
    if pd.isna(std):
        return f"{mean}"
    return f"{mean} ± {std}"

# Group and apply the custom function
result = agg_df.groupby(['Effect', 'Model']).agg(combine_mean_std)

result.to_csv('./results/run2/aggregate.csv')