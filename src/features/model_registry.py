"""
Define models to be used.
Need to have model name on HF hub, languages it was designed for and architecture.
"""
import pandas as pd


lang2model_map = {
    "Universal-NER/UniNER-7B-all": ["all"],
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

model2type_map = {
    "Universal-NER/UniNER-7B-all": "mistral",
    "Babelscape/wikineural-multilingual-ner": "transformer",
    "spacy/xx_ent_wiki_sm": "spacy",
    "spacy/ca_core_news_trf": "spacy",
    "spacy/hr_core_news_lg": "spacy",
    "spacy/da_core_news_trf": "spacy",
    "spacy/nl_core_news_lg": "spacy",
    "spacy/en_core_web_trf": "spacy",
    "spacy/fi_core_news_lg": "spacy",
    "spacy/fr_core_news_lg": "spacy",
    "spacy/de_core_news_lg": "spacy",
    "spacy/el_core_news_lg": "spacy",
    "spacy/it_core_news_lg": "spacy",
    "spacy/lt_core_news_lg": "spacy",
    "spacy/mk_core_news_lg": "spacy",
    "spacy/nb_core_news_lg": "spacy",
    "spacy/pl_core_news_lg": "spacy",
    "spacy/pt_core_news_lg": "spacy",
    "spacy/ro_core_news_lg": "spacy",
    "spacy/ru_core_news_lg": "spacy",
    "spacy/sl_core_news_trf": "spacy",
    "spacy/es_core_news_lg": "spacy",
    "spacy/sv_core_news_lg": "spacy",
    "spacy/uk_core_news_trf": "spacy",
}


def transform_lang2model_mapping(lang2model_mapping: dict, model2type_mapping: dict | None = None) -> pd.DataFrame:
    data = []
    for model, languages in lang2model_mapping.items():
        multilingual = len(languages) > 1
        for language in languages:
            data.append({"language": language, "model": model, "multilingual": multilingual})

    if model2type_map:
        data_with_type = []
        for model_record in data:
            model_name = model_record["model"]
            try:
                model_type = model2type_mapping[model_name] 
            except KeyError:
                model_type = None
            model_record.update({"type": model_type})
            data_with_type.append(model_record)
        data = data_with_type

    return pd.DataFrame(data)


def get_model_registry(lang2model_map, model2type_map):
    return transform_lang2model_mapping(lang2model_map, model2type_map)
