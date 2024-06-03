setwd("~/Repositories/misc/marton-manifestos")
install.packages("manifestoR")
library(manifestoR)
library(tidyverse)

mp_setapikey("manifesto_apikey.txt")

df = mp_corpus_df(edate > as.Date("1990-01-01"))
meta = mp_metadata(edate > as.Date("1990-01-01"))
parties = mp_parties()

write_csv(df, "data/raw/mp_corpus_1990-01-01.csv")
write_csv(meta, "data/raw/meta_corpus_1990-01-01.csv")
write_csv(parties, "data/raw/mp_parties.csv")
