# %%
import json
import time
import random
from pathlib import Path

from tqdm import tqdm
import pandas as pd
import datasets
from iso639 import Lang
from iso639.exceptions import InvalidLanguageValue
import wikipediaapi
from duckduckgo_search import DDGS

import matplotlib.pyplot as plt
import seaborn as sns

# %%
# load all manifestos
processed_paths = Path("data/processed/").glob("*.csv")

df = pd.DataFrame([])
for path in tqdm(processed_paths):
    df_one = pd.read_csv(path)
    df = pd.concat([df, df_one])

# %%
# add metadata
def add_metadata(df: pd.DataFrame, select_cols: list|None = None, drop_cols: list|None = None) -> pd.DataFrame:
    """
    good preset for select_cols: 
    ["id", "created", "metadata.party_abbrev", "metadata.party_name"]
    """
    ds = datasets.load_from_disk("data/interim/mp_1990")
    if select_cols:
        meta = ds.flatten().select_columns(select_cols).to_pandas()
    else:
        meta = ds.flatten().to_pandas()
    if drop_cols:
        meta = meta.drop(columns=drop_cols)

    df_meta = df.merge(meta, how="left", on="id")
    return df_meta

df_meta = add_metadata(df, drop_cols=["text", "added"])

# %%
# add language tag
def add_language_tag(df: pd.DataFrame) -> pd.DataFrame:
    """
    """
    ds = datasets.load_from_disk("data/interim/mp_1990")
    langs_in_ds = pd.DataFrame.from_records(ds["metadata"])["language"].unique().tolist()
    exceptions = {"Greek": "Modern Greek (1453-)",}

    lang2iso = {}
    for lang in langs_in_ds:
        try:
            iso_code = Lang(lang.title()).pt1
            lang2iso.update({lang: iso_code})
        except InvalidLanguageValue:
            if lang.title() in exceptions:
                alt_lang = exceptions[lang.title()]
                iso_code = Lang(alt_lang).pt1
                lang2iso.update({lang: iso_code})
            else:
                lang2iso.update({lang: "NaN"})

    def convert_language_tag(obs):
        lang_orig = obs["metadata"]["language"]
        lang = lang2iso[lang_orig]
        obs["metadata"]["language"] = lang
        return obs

    ds = ds.map(convert_language_tag)

# %%
# cutoff for very LF entities
ent_counts = df_meta.groupby("norm_text").size().rename("count").sort_values(ascending=False).reset_index()
ent_nonunique = ent_counts.query("count > 2")
print(f"Goes from {len(ent_counts)} unique entities to {len(ent_nonunique)} with more than 1 occurance")

# merge
df_fil = pd.merge(df_meta, ent_nonunique, on="norm_text", how="right")
print(f"Goes from {len(df_meta)} entity instances to {len(df_fil)}")

# year information
df_fil["created"] = pd.to_datetime(df_fil["created"])
df_fil["year"] = [date.year for date in df_fil["created"]]

# filter by label
allowed_labels = ["ORG", "LOC", "MISC", "GPE", "NORP", "PER", "PERSON", "placeName", "orgName", "LANGUAGE", "ORGANIZATION", "NAT_REL_POL", "geogName", "persName"]
df_fil_lab = df_fil.query("label == @allowed_labels")
print(f"Goes from {len(df_fil)} to {len(df_fil_lab)} when dropping grabage labels")

# drop non-alphanumeric
def calculate_alpha_percentage(input_string):
    # Initialize counter for alphabetical characters
    alpha_count = 0
    total_count = len(input_string)
    # Iterate through each character in the string
    for char in input_string:
        if char.isalpha() or char.isspace():
            alpha_count += 1
    # Calculate the percentage
    if total_count == 0:
        return 0.0
    percentage = (alpha_count / total_count) * 100
    return percentage

percent_alpha = df_fil_lab["norm_text"].apply(calculate_alpha_percentage).tolist()
df_fil_lab["percent_alpha"] = percent_alpha
df_alpha = df_fil_lab.query("percent_alpha > 33.3")
print(f"Goes from {len(df_fil_lab)} to {len(df_alpha)} after removing non-alphabetical entities")

# def justify_threshold():
#     """
#     Go through the top 50 and identify exceptions
#     """
#     df_fil_lab.query("percent_alpha < 66")[["norm_text", "count", "percent_alpha"]].drop_duplicates(subset=["norm_text"]).sort_values("percent_alpha", ascending=False).head(50)
#     df_fil_lab.query("percent_alpha < 66")[["norm_text", "count", "percent_alpha"]].drop_duplicates(subset=["norm_text"]).sort_values("count", ascending=False).head(50)
#     exceptions = ["d66", "ja21", "f.d.p.", "ε.ε.", "e.e.", "g7", "g8", "g20", "g-20", "c02", "covid-19", "i+d+i"]

# top n named entities per country
df_top = pd.DataFrame([])
for country_name, df_country in df_alpha.groupby(["metadata.country_name", "year"]):
    top_ne = df_country["norm_text"].value_counts().head(50).keys().tolist()
    df_one = df_country.query("norm_text == @top_ne")
    df_top = pd.concat([df_top, df_one])
print(f"Goes from {len(df_alpha)} filtered NEs to {len(df_top)} top NEs")
print(f"Got {len(df_top["norm_text"].unique())} unique NEs")


# %%
# dictionary to search wikipedia with

# # part 1: absolute matches
# # THEY DON'T MAKE SENSE ON ACCOUNT OF "PS" MEANING ALL KINDS OF DIFFERENT SHIT
# absolute_matches = {}
# grouped_text_lang = df_top.groupby('norm_text')['metadata.language'].unique()
# for token in tqdm(ent_nonunique["norm_text"].unique()):
#     if token in grouped_text_lang:
#         valid_langs = grouped_text_lang[token]
#         translation = {lang: token for lang in valid_langs}
#         absolute_matches[token] = translation


# %%
# part 2: filling in the gaps
def search_iteration(query, sleep=True) -> str|None:
    if sleep:
        time.sleep(random.uniform(1, 5))

    results = DDGS().text(query, max_results=5)
    first_wiki_link = None
    for page in results:
        if "wikipedia.org" in page["href"]:
            first_wiki_link = page
            break
        else:
            continue
    return first_wiki_link


SKIP_TO = 0
finished_iterations = 0
results = []
for name, df_group in tqdm(df_top.groupby(["norm_text", "metadata.language"])):
    if finished_iterations < SKIP_TO:
        finished_iterations += 1
        continue

    token, lang = name
    # first round: raw text
    wiki_1st = search_iteration(token)
    results.append({"token": token, "lang": lang, "round": 1, "wiki": wiki_1st})
    # second round: raw text + context
    query = f"{token} {lang}"
    wiki_2nd = search_iteration(query)
    results.append({"token": token, "lang": lang, "round": 2, "wiki": wiki_2nd})

    # third round raw text + context + wiki
    if not wiki_1st and not wiki_2nd:
        query = f"{token} {lang} wiki"
        wiki_3rd = search_iteration(query)
        results.append({"token": token, "lang": lang, "round": 3, "wiki": wiki_3rd})

    # finished iterations
    finished_iterations += 1

    # save every ten itterations
    if finished_iterations % 10 == 0:
        # save the last 10 named entities
        with open(f"data/ddg/dump_{finished_iterations}.json") as fout:
            json.dump(results, fout)
        # reset results
        results = []

    # long pause every 100 iterations
    if (finished_iterations + 1) % 100 == 0:
        print("Pausing for a while...")
        time.sleep(random.uniform(30, 60))

# %%
counter = 0
for name, df_group in tqdm(df_top.groupby(["norm_text", "metadata.language"])):
    counter += 1 







# %%
# fucked_mask = df_fil[df_fil["metadata.language"] == "french"]["norm_text"].str.contains("eu")
# fucked_france = df_fil[df_fil["metadata.language"] == "french"][fucked_mask]


# %%
# new wiki api
import wikipediaapi

wiki_wiki = wikipediaapi.Wikipedia('manifesto-neighbourhoods/0.0 (https://github.com/centre-for-humanities-computing/manifesto-neighbourhoods; jan.kostkan@cas.au.dk)', 'en')

# %%
# edge cases
df_fil[df_fil["metadata.language"] == "danish"].query("count > 5").query("label == 'ORG'").head(20)

# %%
# cer
cer = ds.filter(lambda x: x["id"] == "53321_201602")
cer[0]["text"][13275 - 100: 13278 + 200]

# %%
# danish nato
nato = ds.filter(lambda x : x["id"] == "13229_199803")
nato[0]["text"][7603 - 100 : 7607 + 200]

# %%
# regions dictionary


# %%
# non-interesting
non_interesting_overlap = ["europa", "eu", "nederland", "lietuva", "estado", "europees", "sozial", "españa", "deutschland", "македонија", "ireland", "österreich", "nederlands", "ps", "labour"]
df_fil.query("norm_text != @non_interesting_overlap")["norm_text"].value_counts().head(20)

# %%
for gr_name, group in df_fil.groupby("metadata.language"):
    print(gr_name)
    print(group["norm_text"].value_counts().head(20))
    print("\n")

# %%
