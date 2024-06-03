"""
Dataset schema
--------------

{
    "id": "...",             # MANDATORY: source-specific identifier
    "text": "foo",           # MANDATORY: textual content of the document
    "source": "...",         # MANDATORY: source of the data, such as peS2o, common-crawl, etc.
    "added": "...",          # OPTIONAL: timestamp ai2 acquired this data
    "created": "..."         # OPTIONAL: timestamp when orig document was created (best-guess if not available)
    "metadata": {...}        # OPTIONAL: source-specific metadata
}
"""

from tqdm import tqdm
import pandas as pd
import datasets


# load csv
df = pd.read_csv("data/raw/mp_corpus_2000-01-01.csv")
# remove index
df = df.drop(columns=["Unnamed: 0"])


# v1: merge text chunks together into a single manifesto
v1 = []
for manifesto_id, gr_df in tqdm(df.groupby("manifesto_id")):
    # concat text
    concat_text = " ".join(gr_df["text"].tolist())
    # turn annotations into a list
    cmp_codes = gr_df["cmp_code"].dropna().tolist()
    # number of words
    n_words = len(concat_text.split(" "))
    # representative observation
    obs = gr_df.reset_index().head(1)

    new_obs = {
        "id": manifesto_id,
        "text": concat_text,
        "source": "mp_corpus_2000-01-01",
        "added": "2024-02-28 00:00:00",
        "created": str(pd.to_datetime(obs["date"][0], format="%Y%m")),
        "metadata": {
            "cmp_codes": cmp_codes,
            "party": obs["party"][0],
            "language": obs["language"][0],
            "title": obs["title"][0],
            "n_words": n_words,
        },
    }

    v1.append(new_obs)


ds_v1 = datasets.Dataset.from_list(v1)
ds_v1.save_to_disk("data/interim/mp_2000")



# v2: new version of the API
