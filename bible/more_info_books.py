import pandas as pd
import numpy as np
from googletrans import Translator
from time import sleep
import sys, warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def translate_to_ptbr(val):
    if val is None:
        return None
    sleep(0.3)
    translator = Translator()
    return translator.translate(val, src="en", dest="pt").text

def testamento(val):
    if val <= 39:
        return 'antigo'
    return 'novo'

def trata_cod(row):
    return row['posicao'] if (row['testamento'] == 'Antigo Testamento') else (row['posicao']+39)

df_books = pd.read_csv("./data/BibleBooks.csv", encoding= 'unicode_escape')
df_books['cod'] = df_books['King James Version'].replace(np.nan, 0).astype('int')
df_books['tempo'] = df_books['Time'].astype('int')
#df_books['livro'] = df_books['Book'] #.map(translate_to_ptbr)
df_books['periodo'] = df_books['Period']#.map(translate_to_ptbr)
df_books['localizacao'] = df_books['Location'].map(translate_to_ptbr)
df_books['testamento'] = df_books['cod'].map(testamento)
df_books = df_books[['cod','tempo','periodo','localizacao', 'testamento']]


df_bib = pd.read_csv("./biblia_json.csv")
df_bib['cod'] = df_bib.apply(trata_cod, axis=1)
df_bib['capitulo'] = df_bib['capitulo'].astype('int')
df_bib['versiculo'] = df_bib['versiculo'].astype('int')


df_bib = df_bib[['cod', 'livro', 'capitulo', 'versiculo', 'texto']]

print(df_bib)

df_merge = df_bib.merge(df_books, how='inner',on='cod')

print(df_merge)

df_merge = df_merge.sort_values(by=['cod','capitulo', 'versiculo']).reset_index(drop=True)
df_merge = df_merge.drop_duplicates(subset=['cod', 'capitulo', 'versiculo'], keep='first').reset_index(drop=True)
df_merge.to_csv('./biblia_json_completa.csv', index = False, encoding='utf-8-sig')