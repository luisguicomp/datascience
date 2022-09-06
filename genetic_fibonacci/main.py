"""
23/08/222

@author: Luís Guilherme Ribeiro (LGR)

Código fonte do algoritmo genético adaptado para o problema do caixeiro viajante (TSP),
Com o crossover e mutação baseado na sequência e retração de fibonnaci.
"""
from numpy.random import randint, rand
from functions.genetic_fibonacci import genetic_opt
import numpy as np
import pandas as pd
import time, sys, random

# Função para criar a distância entre os estados (script JD)
def Matrix_cities(N): 
    np.random.seed(73)
    M = np.random.rand(N,N)
    M = np.matmul(M,M.T)
    np.random.seed()
    for i in range(N): 
        M[i,i] = 100000 # Eleva distância das viagens entre as mesmas cidades para não repetir a rota
    return M
 
# Função objetivo para avaliar a distância percorrido de uma permutação (script JD)
def obj_function(Top):
    OBJ = 0
    for k in range(N-1):
        OBJ += M[Top[k],Top[k+1]]
    return OBJ


# Experimentos

# Valores benchmark
sizes = [10,	  20,	   100,     200,     1000]
optim = [18.7401, 71.7976, 2320.65, 9470.30, 242787.15]
times = [1.1727,  3.1422,  165.975, 565.92,  1459.699]

# Tamanho do problema (número de estados)
N = 10 
# Gera a Matriz de distâncias
M = Matrix_cities(N) 
# Tamanho da população
n_pop = N*25
# Total de iterações
n_iter = int(n_pop/2)
# Taxa de crossover 
r_cross = 0.9
# Taxa de mutation
r_mut = 0.5

# Experimento
folds = 10
times, bscores= [], []
for f in range(folds):
	print("Executando... ",f)
	start = time.time()
	best, score = genetic_opt(obj_function, n_iter, n_pop, N, r_cross, r_mut, elitism=True, crossType='fibo_part', mutationType='fibo_ret')
	proc_time = time.time()-start
	print("Tempo decorrido: ", round(proc_time,5))
	print('Score: ',score)
	bscores.append(round(score,5))
	times.append(round(proc_time,5))

print('\n\nMédia:', np.mean(bscores))
print('Min:', np.min(bscores))
print('Tempo Médio: ', np.mean(times))

df = pd.DataFrame(bscores, columns=["score"])
df['time'] = times
df.to_csv('./experimentos/exp_'+str(N)+'_('+str(n_pop)+'_'+str(n_iter)+'_'+str(r_cross)+'_'+str(r_mut)+')_tmp.csv', index=False, sep=',')