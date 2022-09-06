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

df_books = pd.read_csv("./BibleBooks.csv", encoding= 'unicode_escape')
df_books['cod'] = df_books['King James Version'].replace(np.nan, 0).astype('int')
df_books['tempo'] = df_books['Time'].astype('int')
df_books['livro'] = df_books['Book'] #.map(translate_to_ptbr)
df_books['periodo'] = df_books['Period']#.map(translate_to_ptbr)
df_books['localizacao'] = df_books['Location'].map(translate_to_ptbr)
df_books['testamento'] = df_books['cod'].map(testamento)
df_books = df_books[['cod','livro','tempo','periodo','localizacao', 'testamento']]


df_bib = pd.read_csv("./biblia_kjv_ptbr.csv")
df_bib['texto_en'] = df_bib['t']
df_bib['cod'] = df_bib['b'].astype('int')
df_bib['capitulo'] = df_bib['c'].astype('int')
df_bib['versiculo'] = df_bib['v'].astype('int')


df_bib = df_bib[['id', 'cod', 'capitulo', 'versiculo', 'texto_pt', 'texto_en']]

print(df_bib)

df_merge = df_bib.merge(df_books, how='inner',on='cod')

print(df_merge.shape)
print(df_merge)

df_merge.to_csv('./biblia_ptbr_completa.csv', index = False, encoding='utf-8-sig')