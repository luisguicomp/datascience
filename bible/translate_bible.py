import pandas as pd
from googletrans import Translator
from time import sleep
import sys, warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def translate_to_ptbr(val):
    sleep(0.3)
    translator = Translator()
    return translator.translate(val, src="en", dest="pt").text

verse_to_translate = 50
primeiro = False

i = 1
while i <= 10:
    print('\nProcessando ',i,' ...')
    df_pt = pd.read_csv("./biblia_ptbr_.csv")
    max_idx = max(df_pt.index)+1 if ~primeiro else 0
    #max_idx = 1
    df = pd.read_csv("./t_kjv.csv")
    df = df.loc[max_idx:max_idx+verse_to_translate]
    df['texto_pt'] = df['t'].map(translate_to_ptbr)
    df_out = df_pt.append(df, ignore_index=False)
    #df_out = df
    df_out.to_csv('./biblia_ptbr_.csv', index = False, encoding='utf-8-sig')
    print('maxid: ',max(df_out.index))
    i = i+1