"""
"""

# %%
import datasets
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from turftopic import KeyNMF, GMM, SemanticSignalSeparation
import topicwizard

# %%
import os
os.chdir("../")

# %%
# import preprocessed corpus
ds = datasets.load_from_disk("data/interim/mp_concat")

corpus_multilingual = ds["text"]
corpus_english = ds.filter(lambda obs: obs["metadata"]["language"] == "english")["text"]
corpus_danish = ds.filter(lambda obs: obs["metadata"]["language"] == "danish")["text"]

# %%
# classical model
vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, stop_words="english", lowercase=True)
m_nmf = NMF(n_components=10, max_iter=2000)
classical_pipeline = make_pipeline(vectorizer, m_nmf)
classical_pipeline.fit(corpus_english)

# viz
topicwizard.visualize(corpus_english, model=classical_pipeline)


# %%
# keynmf
m_keynmf = KeyNMF(10, top_n=10).fit(corpus_english)
m_keynmf.print_topics()

# %%
# gmm
model_gmm = GMM(10, weight_prior="dirichlet_process").fit(corpus_english)
model_gmm.print_topics()

# %%
# S3
model_s3 = SemanticSignalSeparation(10, objective="independence", encoder="intfloat/multilingual-e5-large-instruct").fit(corpus_danish)
model_s3.print_topics()

# %%
