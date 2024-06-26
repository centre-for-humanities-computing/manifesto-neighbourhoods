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


df = pd.read_csv("data/raw_mp_dumps/mp_corpus_1990-01-01.csv")
meta = pd.read_csv("data/raw_mp_dumps/meta_corpus_1990-01-01.csv")
parties = pd.read_csv("data/raw_mp_dumps/mp_parties.csv")


# select countries of interests from the party metadata table
countries_of_interest = [
    "Italy",
    "Serbia",
    "Georgia",
    "Croatia",
    "Ukraine",
    "Poland",
    "Spain",
    "Montenegro",
    "Latvia",
    "Netherlands",
    "Romania",
    "Slovakia",
    "North Macedonia",
    "Lithuania",
    "Bulgaria",
    "Belgium",
    "Czech Republic",
    "Slovenia",
    "Russia",
    "France",
    "Armenia",
    "Turkey",
    "Bosnia-Herzegovina",
    "Estonia",
    "Greece",
    "Portugal",
    "Ireland",
    "Denmark",
    "Germany",
    "Iceland",
    "Switzerland",
    "Moldova",
    "German Democratic Republic",
    "Finland",
    "United Kingdom",
    "Hungary",
    "Albania",
    "Norway",
    "Cyprus",
    "Austria",
    "Sweden",
    "Luxembourg",
    "Belarus",
    "Azerbaijan",
    "Northern Ireland",
    "Malta",
]

parties_subset = parties.query("countryname == @countries_of_interest")
parties_of_interest = set(parties_subset["party"].tolist())

# select parties to process from df
df_subset = df[df["party"].isin(parties_of_interest)]

# select valid manifesto_ids from the manifesto metadata table
meta_subset = meta.dropna(subset=["manifesto_id"])
assert len(set(meta_subset["manifesto_id"])) == len(meta_subset)


v1 = []
for manifesto_id, gr_df in tqdm(df_subset.groupby("manifesto_id")):
    # concat text
    concat_text = " ".join(gr_df["text"].tolist())
    # turn annotations into a list
    cmp_codes = gr_df["cmp_code"].dropna().tolist()
    # number of words
    n_words = len(concat_text.split(" "))
    # representative observation
    obs = gr_df.reset_index().head(1)
    # find additional manifesto metadata
    obs_manifesto_meta = meta_subset[
        meta_subset["manifesto_id"] == manifesto_id
    ].reset_index()
    # find additional party metadata
    obs_party_meta = parties_subset[
        parties_subset["party"] == obs["party"][0]
    ].reset_index()

    new_obs = {
        "id": manifesto_id,
        "text": concat_text,
        "source": obs_manifesto_meta["source"][0],
        "added": "2024-06-03 00:00:00",
        "created": str(pd.to_datetime(obs["date"][0], format="%Y%m")),
        "metadata": {
            "country_code": obs_party_meta["country"][0],
            "country_name": obs_party_meta["countryname"][0],
            "party_code": obs["party"][0],
            "party_abbrev": obs_party_meta["abbrev"][0],
            "party_name": obs_party_meta["name"][0],
            "language": obs["language"][0],
            "title": obs_manifesto_meta["title"][0],
            "n_words": n_words,
            "cmp_codes": cmp_codes,
            "translation_en": obs["translation_en"][0],
        },
    }

    v1.append(new_obs)

df_v1 = pd.DataFrame.from_records(v1)
ds_v1 = datasets.Dataset.from_pandas(df_v1)
ds_v1.save_to_disk("data/interim/mp_1990")
