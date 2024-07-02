"""
"""
import json
from pathlib import Path

import datasets
import pandas as pd
from tqdm import tqdm
import spacy
from wasabi import msg

from src.features.model_registry import get_model_registry


def convert_predictions(path: str) -> pd.DataFrame:
    """Path to prediciton json files: list[dict]"""
    with open(path, 'r') as f:
        loaded_data = json.load(f)

    df = pd.DataFrame([])
    for doc_ents in loaded_data:
        partial = pd.DataFrame(doc_ents)
        df = pd.concat([df, partial])

    return df


# load model registry for languages covered by spacy
model_registry = get_model_registry()
spacy_models = model_registry.query("type == 'spacy' & multilingual == False").set_index("language").to_dict()


# prediction paths
prediction_paths = list(Path("data/predictions").glob("*.json"))
for path in prediction_paths:
    # import predictions
    df = convert_predictions(path)
    lang = str(path)[-7::][0:2]
    msg.info(lang)
    # initialize model
    model_name = spacy_models["model"][lang]
    model_name = model_name.split("/")[1]
    spacy.cli.download(model_name)
    nlp = spacy.load(model_name)
    # run normalization
    norm_text = []
    for doc in tqdm(nlp.pipe(df["text"]), total=len(df)):
        lemmatized_ent = " ".join([tok.lemma_ for tok in doc]).lower()
        norm_text.append(lemmatized_ent)
    df["norm_text"] = norm_text
    df.to_csv(f"data/processed/pred_{lang}.csv", index=False)
