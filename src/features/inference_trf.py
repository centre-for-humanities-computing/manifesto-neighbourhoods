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

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


def infer_with_trf(
    docs: list[str], model_name: str, ids: list[str] | None = None, **pipe_kwargs
) -> list[list[dict]]:
    # ids and docs have to be the same length
    if ids:
        assert len(ids) == len(docs)

    # initialize model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

    # infer
    raw_predictions = nlp(docs, **pipe_kwargs)
    # restructure predictions
    predictions = []
    for doc in raw_predictions:
        doc_ents = []
        for ent in doc:
            doc_ents.append(
                {
                    "text": ent["word"],
                    "label": ent["entity_group"],
                    "score": ent["score"],
                    "start": ent["start"],
                    "end": ent["end"],
                }
            )
        predictions.append(doc_ents)

    if ids:
        for manifesto_id, doc in zip(ids, predictions):
            doc["id"] = manifesto_id

    return predictions
