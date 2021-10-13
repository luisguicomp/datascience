
#VARIÁVEIS CATEGÓRICAS

countries = so_survey_df['Country']

# Get the counts of each category
country_counts = countries.value_counts()

# Create a mask for only categories that occur less than 10 times
mask = countries.isin(country_counts[country_counts < 10].index)

# Label all other categories as Other
countries[mask] = 'Other'

# Print the updated category counts
print(pd.value_counts(countries))


#VARIÁVEIS NUMÉRICAS

#Binarizando
# Create the Paid_Job column filled with zeros
so_survey_df['Paid_Job'] = 0

# Replace all the Paid_Job values where ConvertedSalary is > 0
so_survey_df.loc[so_survey_df['ConvertedSalary']>0, 'Paid_Job'] = 1

# Print the first five rows of the columns
print(so_survey_df[['Paid_Job', 'ConvertedSalary']].head())


#DISCRETIZANDO COM NUMPY
import numpy as np

# Specify the boundaries of the bins
bins = [-np.inf, 10000, 50000, 100000, 150000, np.inf]

# Bin labels
labels = ['Very low', 'Low', 'Medium', 'High', 'Very high']

# Bin the continuous variable ConvertedSalary using these boundaries
so_survey_df['boundary_binned'] = pd.cut(so_survey_df['ConvertedSalary'], bins=bins, labels=labels)

# Print the first 5 rows of the boundary_binned column
print(so_survey_df[['boundary_binned', 'ConvertedSalary']].head())



#DISTRIBUIÇÃO

# Import packages
import seaborn as sns
import matplotlib.pyplot as plt

# Plot pairwise relationships
sns.pairplot(df)

plt.show()


#NORMALIZAÇÃO
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer

scaler = MinMaxScaler() # varia entre 0 e 1
scaler = StandardScaler() # Graus do desvio padrão sobre a média
scaler = PowerTransformer() # Escala logarítmica

df['coluna_normalizada'] = scaler.fit_transform(df[['coluna']])


#REMOVENDO OUTLIERS

# QUARTIS (95% - REMOVER 5%)
# Find the 95th quantile
quantile = so_numeric_df['ConvertedSalary'].quantile(0.95)

# Trim the outliers
trimmed_df = so_numeric_df[so_numeric_df['ConvertedSalary'] < quantile]

# The original histogram
so_numeric_df[['ConvertedSalary']].hist()
plt.show()
plt.clf()

# The trimmed histogram
trimmed_df[['ConvertedSalary']].hist()
plt.show()


# 3o GRAU DO DESVIO PADRÃO SOBRE A MÉDIA
# Find the mean and standard dev
std = so_numeric_df['ConvertedSalary'].std()
mean = so_numeric_df['ConvertedSalary'].mean()

# Calculate the cutoff
cut_off = std * 3
lower, upper = mean - cut_off, mean+cut_off

# Trim the outliers
trimmed_df = so_numeric_df[(so_numeric_df['ConvertedSalary'] < upper) & (so_numeric_df['ConvertedSalary'] > lower)]

# The trimmed box plot
trimmed_df[['ConvertedSalary']].boxplot()
plt.show()

# OBS: PARA NORMALIZAÇÃO E REMOÇÃO DE OUTLIERS SEMPRE TREINAR (FIT) NA BASE DE TREINO E TRANSFORMAR A BASE DE TESTE ANTES DE VALIDAR


# MANIPULANDO TEXTOS

# Replace all non letter characters with a whitespace
speech_df['text_clean'] = speech_df['text'].replace('[^a-zA-Z]', ' ')

# Change to lower case
speech_df['text_clean'] = speech_df['text_clean'].str.lower()

# Print the first 5 rows of the text_clean column
print(speech_df['text_clean'].head())

#CONTANDO PALAVRAS
# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Specify arguements to limit the number of features generated
cv = CountVectorizer(min_df=.2,max_df=.8)

# Fit, transform, and convert into array
cv_transformed = cv.fit_transform(speech_df['text_clean'])
cv_array = pd.DataFrame(cv_transformed.toarray(), columns=cv.get_feature_names())

# Print the array shape
print(cv_array.shape)


# CONTA AS PRINCIPAIS PALAVRAS IGNORANDO AS STOPS WORDS

# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Instantiate TfidfVectorizer
tv = TfidfVectorizer(max_features=100,stop_words='english')

# Fit the vectroizer and transform the data
tv_transformed = tv.fit_transform(speech_df['text_clean'])

# Create a DataFrame with these features
tv_df = pd.DataFrame(tv_transformed.toarray(), 
                     columns=tv.get_feature_names()).add_prefix('TFIDF_')
print(tv_df.head())


# N-Gram

# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Instantiate a trigram vectorizer
cv_trigram_vec = CountVectorizer(max_features=100, 
                                 stop_words='english', 
                                 ngram_range=(3,3))

# Fit and apply trigram vectorizer
cv_trigram = cv_trigram_vec.fit_transform(speech_df['text_clean'])

# Print the trigram features
print(cv_trigram_vec.get_feature_names())


