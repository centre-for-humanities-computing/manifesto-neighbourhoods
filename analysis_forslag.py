# %%
import json
from pathlib import Path

import datasets
import pandas as pd
from tqdm import tqdm
import spacy

# import seaborn as sns
# import matplotlib.pyplot as plt

# %%
prediction_paths = list(Path("data/processed").glob("*.csv"))
all = pd.DataFrame([])
for path in prediction_paths:
    df_one = pd.read_csv(path)
    all = pd.concat([all, df_one])

# %%
(all
    .groupby("norm_text")
    .size()
    .sort_values(ascending=False)
    .head(50)
)

# %%
ds = datasets.load_from_disk("data/interim/mp_1990")
meta = ds.flatten().select_columns(["id", "created", "metadata.party_abbrev", "metadata.party_name"]).to_pandas()
da = da.merge(meta, how="left", on="id")

# %%
def prep_for_wfplot(df: pd.DataFrame, query: str):
    """
    """
    # datetime
    df['created'] = pd.to_datetime(df['created'])
    # make subset
    df_subset = df[df["norm_text"] == query]
    df_subset["count"] = 1
    df_grouped = df_subset.groupby(['created', 'metadata.party_name']).sum().reset_index()
    # generate time range
    date_range = pd.date_range(start=df['created'].min(), end=df['created'].max())
    parties = df['metadata.party_name'].unique()
    df_complete = pd.DataFrame([(d, c) for d in date_range for c in parties], columns=['created', 'metadata.party_name'])
    df_complete = df_complete.merge(df_grouped, on=['created', 'metadata.party_name'], how='left').fillna(0)
    return df_complete

sub = prep_for_wfplot(da, "danmark")
sns.scatterplot(data=sub, x='created', y='count', hue='metadata.party_name')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# %%
# number of named entities per year
def prep_for_baseline(df):
    """
    """
    # prepare data
    df['created'] = pd.to_datetime(df['created'])
    df_count = df.groupby('created').size().reset_index(name='count')
    date_range = pd.date_range(start=df['created'].min(), end=df['created'].max())
    # merge data
    df_complete = pd.DataFrame(date_range, columns=['created'])
    df_complete = df_complete.merge(df_count, on='created', how='left').fillna(0)
    return df_complete

da_base = prep_for_baseline(da)
sns.scatterplot(data=da_base, x="created", y="count")
