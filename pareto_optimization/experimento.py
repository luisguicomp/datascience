import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Perceptron 
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot
from pareto_frontier import paretoFrontier

# Base de dados sintética
X, y = make_classification(n_samples=10000, n_features=30, n_informative=2, n_redundant=1)
# dividindo a base de dados em treno e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

train_scores, test_scores, overfitting_score, pareto_list = [], [], [], []

names = ['KNN', 'SVM', 'RF', 'MLP', 'LR', 'PE', 'CART', 'NB']
models = [KNeighborsClassifier(n_neighbors=3), SVC(), RandomForestClassifier(), 
MLPClassifier(max_iter=500), LogisticRegression(), Perceptron(), DecisionTreeClassifier(), GaussianNB()]

i = 0
for model in models:
    model.fit(X_train, y_train)
    #Medindo acurácia na base de treino
    train_acc = accuracy_score(y_train, model.predict(X_train))
    train_scores.append(round(train_acc,2))
    #Medindo acurácia na base de teste
    test_acc = accuracy_score(y_test, model.predict(X_test))
    acc_value = round(test_acc*100,0)
    test_scores.append(acc_value)
    # Medindo overfitting 
    overfit_acc = test_acc/train_acc
    overf_value = round((overfit_acc if overfit_acc < 1 else 1)*100,0)
    overfitting_score.append(overf_value)
    #Inserindo na array que será usada para medir a fronteira de pareto
    pareto_list.append([acc_value,overf_value])
    i = i+1

#Plotagem do gráfico de barras
x = np.arange(len(names))
width = 0.4

fig, ax = pyplot.subplots()
rects2 = ax.bar(x - (width/2), test_scores, width, label='Acurácia')
rects3 = ax.bar(x + (width/2), overfitting_score, width, label='Overfitting')

ax.set_title('Avaliação dos modelos')
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.legend(bbox_to_anchor=(1, 1))
ax.bar_label(rects2, padding=2)
ax.bar_label(rects3, padding=2)
fig.tight_layout()
pyplot.show()

#Chamada da função para plotar a fronteira de Pareto
paretoFrontier(pareto_list,names)
