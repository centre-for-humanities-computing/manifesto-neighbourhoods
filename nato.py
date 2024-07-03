# %%
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import datasets

import matplotlib.pyplot as plt
import seaborn as sns

# %%
nato_forms = ["nato", "otan", "нато", "navo"]
processed_paths = Path("data/processed/").glob("*.csv")

df_nato = pd.DataFrame([])
for path in tqdm(processed_paths):
    df_one = pd.read_csv(path)
    df_one = df_one.query("norm_text == @nato_forms")
    df_nato = pd.concat([df_nato, df_one])

# %%
def add_metadata(df: pd.DataFrame, select_cols: list|None = None) -> pd.DataFrame:
    """
    good preset for select_cols: 
    ["id", "created", "metadata.party_abbrev", "metadata.party_name"]
    """
    ds = datasets.load_from_disk("data/interim/mp_1990")
    if select_cols:
        meta = ds.flatten().select_columns(select_cols).to_pandas()
    else:
        meta = ds.flatten().to_pandas()
    df_meta = df.merge(meta, how="left", on="id")
    return df_meta

df_nato = add_metadata(df_nato)

# %%
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

df_nato_baseline = prep_for_baseline(df_nato)
sns.scatterplot(data=df_nato_baseline, x="created", y="count")

# %%
def prep_for_wordfreq_bygroup(df: pd.DataFrame, query: str|None = None, group: str|None = None) -> pd.DataFrame:
    """
    Group is an existing column.
    """
    df_ = df.copy(deep=True)
    df_["created"] = pd.to_datetime(df_["created"])
    df_["count"] = 1
    
    # make subset
    if query:
        df_ = df_[df_["norm_text"] == query]

    # generate time range
    date_range = pd.date_range(start=df_["created"].min(), end=df_["created"].max())

    # make grouping
    if group:
        df_ = df_.groupby(["created", group]).sum().reset_index()
        group_members = df[group].unique()
        df_preped = pd.DataFrame([(d, c) for d in date_range for c in group_members], columns=["created", group])
        df_preped = df_preped.merge(df_, on=["created", group], how='left').fillna(0)
    else:
        df_preped = pd.DataFrame(date_range, columns=["created"])
        df_preped = df_preped.merge(df_, on="created", how="left").fillna(0)

    return df_preped

df_nato_country = prep_for_wordfreq_bygroup(df=df_nato, query=None, group="metadata.country_name")
sns.scatterplot(data=df_nato_country, x='created', y='count', hue='metadata.country_name')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# %%
# coding countries aligence to nato
coding_nato_periods = {
    'Austria': "Non-member",
    'Belgium': "Cold war",
    'Bulgaria': "End of history",
    'Croatia': "End of history",
    'Denmark': "Cold war",
    'Finland': "New thread",
    'France': "Cold war",
    'Germany': "Cold war",
    'Greece': "Cold war",
    'Ireland': "Non-member",
    'Italy': "Cold war",
    'Lithuania': "End of history",
    'Luxembourg': "Cold war",
    "Netherlands": "Cold war",
    'North Macedonia': "New thread",
    'Poland': "End of history",
    'Portugal': "Cold war",
    'Russia': "Russia",
    'Slovenia': "End of history",
    'Spain': "Cold war",
    'Sweden': "New thread",
    'Switzerland': "Non-member",
    'Ukraine': "Ukraine",
    'United Kingdom': "Cold war",
}

df_nato["nato_period"] = df_nato["metadata.country_name"].map(lambda x: coding_nato_periods[x])
df_nato_period = prep_for_wordfreq_bygroup(df=df_nato, query=None, group="nato_period")

# %%
sns.scatterplot(data=df_nato_period, x='created', y='count', hue='nato_period')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# %%
sns.violinplot(
    data=df_nato_period, x='nato_period', y='count', hue='nato_period',
    density_norm="width"
)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# %%
# nato country plot
df_nato_country_plot = df_nato_country.query("count != 0")
sns.lineplot(data=df_nato_country_plot, x="created", y="count", hue="metadata.country_name", legend=None, markers=True, dashes=False)
sns.scatterplot(data=df_nato_country_plot, x="created", y="count", hue="metadata.country_name")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# %%
