"""
Prediction dict come in
{
    "text": ent.text,           # token recognized as an entity
    "label": ent.label_,        # entity type
    "score": None,              # prediction score
    "start": ent.start_char,    # idx of character where entity starts
    "end": ent.end_char,        # idx of character where entity starts
}
"""

import spacy
from tqdm import tqdm


def infer_with_spacy(
    docs: list[str], model_name: str, ids: list[str] | None = None, **pipe_kwargs
) -> list[list[dict]]:
    """
    Run inference using a spaCy model on a list of documents and return the predictions.

    Parameters
    ----------
    docs : list of str
        A list of text documents to process.
    model_name : str
        The name of the spaCy model to use for inference.
    ids : list of str, optional
        A list of IDs corresponding to the documents. If provided, it must be the same length as `docs`.
    **pipe_kwargs : keyword arguments
        Additional keyword arguments to pass to the spaCy `nlp.pipe` method.
    """

    # ids and docs have to be the same length
    if ids:
        assert len(ids) == len(docs)

    # initialize model
    if model_name.startswith("spacy/"):
        model_name = model_name.split("/")[1]

    spacy.cli.download(model_name)
    nlp = spacy.load(model_name)

    # run inference
    predictions = []
    for doc in tqdm(nlp.pipe(docs, **pipe_kwargs), total=len(docs)):
        doc_ents = []
        for ent in doc.ents:
            doc_ents.append(
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "score": None,
                    "start": ent.start_char,
                    "end": ent.end_char,
                }
            )
        predictions.append(doc_ents)

    # add ids
    if ids:
        for manifesto_id, doc in zip(ids, predictions):
            doc["id"] = manifesto_id

    return predictions
