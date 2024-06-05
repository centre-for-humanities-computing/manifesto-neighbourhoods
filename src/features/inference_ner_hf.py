# %%
import datasets
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

def transform_lang2model_mapping(lang2model_mapping: dict) -> pd.DataFrame:
    data = []
    for model, languages in lang2model_mapping.items():
        multilingual = len(languages) > 1
        for language in languages:
            data.append({"Language": language, "Model": model, "Multilingual_model": multilingual})
    
    return pd.DataFrame(data)

# %%
# parameters
DATASET_PATH = "data/interim/mp_1990"

lang2model_mapping = {
    "https://huggingface.co/Universal-NER/UniNER-7B-all": ["all"],
    "Babelscape/wikineural-multilingual-ner": ["de", "en", "es", "fr", "it", "nl", "pl", "pt", "ru"],
    "spacy/xx_ent_wiki_sm": ["de", "en", "es", "fr", "it", "nl", "pl", "pt", "ru"],
    "spacy/ca_core_news_trf": ["ca"],
    "spacy/hr_core_news_lg": ["hr"],
    "spacy/da_core_news_trf": ["da"],
    "spacy/nl_core_news_lg": ["nl"],
    "spacy/en_core_web_trf": ["en"],
    "spacy/fi_core_news_lg": ["fi"],
    "spacy/fr_core_news_lg": ["fr"],
    "spacy/de_core_news_lg": ["de"],
    "spacy/el_core_news_lg": ["el"],
    "spacy/it_core_news_lg": ["it"],
    "spacy/lt_core_news_lg": ["lt"],
    "spacy/mk_core_news_lg": ["mk"],
    "spacy/nb_core_news_lg": ["nb"],
    "spacy/pl_core_news_lg": ["pl"],
    "spacy/pt_core_news_lg": ["pt"],
    "spacy/ro_core_news_lg": ["ro"],
    "spacy/ru_core_news_lg": ["ru"],
    "spacy/sl_core_news_trf": ["sl"],
    "spacy/es_core_news_lg": ["es"],
    "spacy/sv_core_news_lg": ["sv"],
    "spacy/uk_core_news_trf": ["uk"],
}

lang2model_mapping = transform_lang2model_mapping(lang2model_mapping)

###
### Comments
### --------
#
# ISO 639-1
#     language code
#
# spacy lg models are CPU optimized. 
#     Apart from downloading a lot of data,
#     they can be run on cheap hardware.



# %%
# initialize dataset
ds = datasets.load_from_disk(DATASET_PATH)

# %%
# initialize model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

# %%
# run inference


# nlp = spacy.load("en_core_web_trf", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
