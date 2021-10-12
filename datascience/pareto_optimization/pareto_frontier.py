import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def identify_pareto(scores):
    # Quantidade de modelos
    population_size = scores.shape[0]
    # Cria um indíce para a potuação da fronteira de pareto
    population_ids = np.arange(population_size)
    # Cria uma lista onde todos os item iniciam na fronteira de Pareto
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop para comprarar todos os itens
    for i in range(population_size):
        for j in range(population_size):
            # Verifica se `i` é dominado por `j`
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # `i` é demonado por `j`, então da fronteira
                pareto_front[i] = 0
                break
    # Retorna os índices da fronteira de Pareto
    return population_ids[pareto_front]


def paretoTest(pareto_list, names):
    scores = np.array(pareto_list)
    title = 'Análise multi-objetivo'
    x = scores[:, 1]
    y = scores[:, 0]

    pareto = identify_pareto(scores)
    print ('Pareto front index vales')
    print ('Points on Pareto front: \n',pareto)

    pareto_front = scores[pareto]
    print ('\nPareto front scores')
    print (pareto_front)

    pareto_front_df = pd.DataFrame(pareto_front)
    pareto_front_df.sort_values(0, inplace=True)
    pareto_front = pareto_front_df.values

    x_all = scores[:, 1]
    y_all = scores[:, 0]
    x_pareto = pareto_front[:, 1]
    y_pareto = pareto_front[:, 0]


    for i in range(len(scores)):
        x = scores[i][1]
        y = scores[i][0]
        if(x in x_pareto and y in y_pareto):
          pareto, = plt.plot(x, y, 'go')
          plt.text(x+0.025, y-0.007, names[i], fontsize=12)
        else:
          n_pareto, = plt.plot(x, y, 'bo')
          plt.text(x+0.025, y-0.04, names[i], fontsize=12)

    leg = plt.legend([pareto,n_pareto],['Ótimo de pareto', 'Não-ótimo de pareto'], loc='lower left')

    colors=['blue', 'red']

    for i, j in enumerate(leg.legendHandles):
        j.set_color(colors[i])

    plt.plot(x_pareto, y_pareto, color='r')
    plt.title(title)
    plt.xlabel('Overfitting')
    plt.ylabel('Acurácia')
    plt.show()

#names = ['A','B','C','D','E','F','G','H','I','J']
# scores = np.array([
#  [97, 23],
#  [55, 63],
#  [80, 60],
#  [99,  4],
#  [26, 70],
#  [30, 75],
#  [15, 80],
#  [66, 65],
#  [90, 68]])