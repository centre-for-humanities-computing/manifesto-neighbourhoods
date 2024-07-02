# %%
import json

import datasets
import pandas as pd
from iso639 import Lang
from iso639.exceptions import InvalidLanguageValue
from wasabi import msg

from src.features.model_registry import get_model_registry
from src.features.inference_spacy import infer_with_spacy

# %%
DATASET_PATH = "data/interim/mp_1990"

ds = datasets.load_from_disk(DATASET_PATH)
model_registry = get_model_registry()

# %%
# language tag conversion
langs_in_ds = pd.DataFrame.from_records(ds["metadata"])["language"].unique().tolist()
exceptions = {
    "Greek": "Modern Greek (1453-)",
}

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

# %%
def convert_language_tag(obs):
    lang_orig = obs["metadata"]["language"]
    lang = lang2iso[lang_orig]
    obs["metadata"]["language"] = lang
    return obs

ds = ds.map(convert_language_tag)

# %%
# manifestos per language
ds_meta = pd.DataFrame.from_records(ds["metadata"])
ds_meta["created"] = ds["created"]

# keep languages with a spacy ner model
spacy_models = model_registry.query("type == 'spacy' & multilingual == False").set_index("language").to_dict()
spacy_coverage = model_registry.query("type == 'spacy' & multilingual == False")["language"].unique().tolist()
ds_meta = ds_meta[ds_meta["language"].isin(spacy_coverage)]

# count
ds_meta.groupby("language").size().sort_values(ascending=False)

# %%
# keep manifestos that can be processed with spacy
ds_sp = ds.filter(lambda x: x["metadata"]["language"] in spacy_coverage)

# %%
# define long text strategy
long_doc_strategy = "skip"

def _turncate(example):
    example["text"] = example["text"][0:1_000_000]
    return example

# remove langs that we're not doing after all
for l in ["nb"]:
    spacy_coverage.remove(l)

# inference loop
for lang in spacy_coverage:
    msg.info(f"lang: {lang}")
    ds_one_lang = ds_sp.filter(lambda x: x["metadata"]["language"] == lang)

    # long text strategy
    if long_doc_strategy == "turncate":
        msg.info("turncating long documents")
        ds_one_lang = ds_one_lang.map(_turncate)
    elif long_doc_strategy == "skip":
        msg.info("skipping long documents")
        ds_one_lang = ds_one_lang.filter(lambda x: len(x["text"]) < 1_000_000)

    msg.info("inference")
    pred_lang = infer_with_spacy(
        docs=ds_one_lang["text"],
        model_name = spacy_models["model"][lang],
        ids=ds_one_lang["id"],
        batch_size=1,
    )

    json_data = json.dumps(pred_lang, indent=4)
    with open(f"data/predictions/pred_{lang}.json", 'w') as f:
        f.write(json_data)

# %%
