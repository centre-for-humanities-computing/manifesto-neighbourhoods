"""
"""

import pandas as pd
from datasets import Dataset


def predictions_to_df(predictions: list[list[str]]) -> pd.DataFrame:
    df = pd.DataFrame([])
    for doc_ents in predictions:
        partial = pd.DataFrame(doc_ents)
        df = pd.concat([df, partial])
    return df.reset_index()


def find_sentence(prediction: dict, dataset: Dataset, window: int = 200) -> str:
    manifesto = dataset.filter(lambda x: x["id"] == prediction["id"])
    assert manifesto.num_rows == 1
    doc_n_char = len(manifesto["text"][0])
    idx_start = prediction["start"] - window if prediction["start"] - window >= 0 else 0
    idx_end = (
        prediction["end"] + window
        if prediction["end"] + window <= doc_n_char
        else doc_n_char
    )
    return manifesto["text"][0][idx_start:idx_end]
