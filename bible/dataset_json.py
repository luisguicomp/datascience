import pandas as pd
dfj = pd.read_json('./data/biblia.json')
print(dfj.shape)
print(dfj['livro'].value_counts())

dfj.drop('versao', axis=1, inplace=True)
dfj.to_csv('./biblia_json.csv', index = False, encoding='utf-8-sig')



