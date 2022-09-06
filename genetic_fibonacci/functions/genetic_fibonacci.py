"""
23/08/222

@author: Luís Guilherme Ribeiro (LGR)

Código fonte do algoritmo genético adaptado para o problema do caixeiro viajante (TSP),
Com o crossover e mutação baseado na sequência e retração de fibonnaci.
"""
from numpy.random import randint, rand
import numpy as np
import random

# Função de seleção dos melhores indívuos para o crossover
def selection(pop, scores, k=3):
	# Seleciona aleatoriamente o primeiro indivíduo para usar sua pontuação como referência
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		# Verifica o melhor indívuo e o retorna
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]

# Função para considerar elitismos (LGR)
def elitismo(pop,scores):
	# Tamanho da elite
	elite_size = int(len(pop)*0.01) # 1%
	# Unifica listas
	joined = list(zip(pop, scores))
	# Ordena por pontuação
	ordered_individuals = sorted_by_second = sorted(joined, key=lambda tup: tup[1], reverse = False)
	# Seleciona os melhores indivíduos baseado no tamanho da elite
	elite = [x[0] for x in ordered_individuals[0:elite_size]]
	return elite

# Função para trocar a posição de 2 estados (LGR)
def change(ind, l1, l2):
	tmp = ind[l1]
	ind[l1] = ind[l2]
	ind[l2] = tmp

#Função para criar permutações aleatórias
def create_permutations(num):
    arr = []
    tmp = random.randint(0, num-1)
    for x in range(num):
        while tmp in arr:
            tmp = random.randint(0, num-1)
        arr.append(tmp)      
    return arr


# Crossover usando retração de fibonacci
def crossover_fibo_ret(p1, p2, r_cross):
	# Cortes baseado na retração de fibonacci
	cl1 = int(len(p1)*0.146)
	cl2 = int(len(p1)*0.236)
	cl3 = int(len(p1)*0.382)
	cl4 = int(len(p1)*0.618)

	c1, c2 = p1.copy(), p2.copy()
	# Cross 3 e 4
	change(c1, cl4, p1.index(p2[cl3]))
	change(c1, cl3, p1.index(p2[cl4]))
	change(c2, cl4, p2.index(p1[cl3]))
	change(c2, cl3, p2.index(p1[cl4]))
	# Cross 1 e 2
	change(c1, cl2, p1.index(p2[cl1]))
	change(c1, cl1, p1.index(p2[cl2]))
	change(c2, cl2, p2.index(p1[cl1]))
	change(c2, cl1, p2.index(p1[cl2]))
	return [c1, c2]


# Mutação usando retração de fibonacci
def mutation_fibo_ret(ind, r_mut):
	cl1 = int(len(ind)*0.146)
	cl2 = int(len(ind)*0.236)
	cl3 = int(len(ind)*0.382)
	cl4 = int(len(ind)*0.618)
	if rand() < r_mut:
		change(ind, cl1, cl2)
	if rand() < r_mut:
		change(ind, cl3, cl4)


# Função de crossover baseado na sequência de fibonacci (particionado)
def crossover_fibo_part(p1, p2, r_cross):
	c1, c2 = p1.copy(), p2.copy()
	parts = 2 # Número de partes para aplicar o fibonacci
	tam_p = int(len(p1)/parts) #tamanho de cada parte
	for p in range(parts):
		b = 1
		while b <= tam_p:
			if rand() < r_cross:
				bit = ((p*tam_p)+b)-1
				change(c1, bit, p1.index(p2[bit]))
				change(c2, bit, p2.index(p1[bit]))
			b = int(round(b*1.618))
	return [c1, c2]

# Função de crossover baseado na sequência de fibonacci (inicio e fim)
def crossover_fibo_be(p1, p2, r_cross):
	c1, c2 = p1.copy(), p2.copy()
	b = 1
	end = False
	while b <= int(len(p1)/2):
		if rand() < r_cross:
			if end:
				bit = len(p1) - 1 - b
				end = False
			else:
				bit = b
				end = True
			# if rand() < 0.5: # Aplica aleatoriamente Fibonacci para inicio e fim do ciclo
			# 	bit = len(p1) - 1 - b
			change(c1, bit, p1.index(p2[bit]))
			change(c2, bit, p2.index(p1[bit]))
		b = int(round(b*1.618)) #avanço na sequência baseado na retração 61.8%
	return [c1, c2]


# Função de crossover baseado na aleatoriedade e número de bits
def crossover_random(p1, p2, r_cross):
	# Número de mudanças baseado no tamanho da população
	n_changes = int(len(p1)*0.5)
	c1, c2 = p1.copy(), p2.copy()
	for x in range(n_changes):
		if rand() < r_cross:
			bit = randint(len(p1))
			change(c1, bit, p1.index(p2[bit]))
			change(c2, bit, p2.index(p1[bit]))
	return [c1, c2]

# Função de mutação baseado na aleatoriedade
def mutation_random(ind, r_mut):
	if rand() < r_mut:
		change(ind, randint(len(ind)), randint(len(ind)))

# Função de dicionário para selecionar qual método de crossover será usado
def crossover_function(x):
    return {
        'random': crossover_random,
        'fibo_be': crossover_fibo_be,
        'fibo_part': crossover_fibo_part
    }.get(x, crossover_fibo_be)

# Função de dicionário para selecionar qual método de mutação será usado
def mutation_function(x):
    return {
        'random': mutation_random,
        'fibo_ret': mutation_fibo_ret
    }.get(x, mutation_fibo_ret)

# Função principal do algoritmo genético
def genetic_opt(objective, n_iter, n_pop, bits, r_cross, r_mut, elitism=True, crossType='fibo_be', mutationType='fibo_ret'):
	# Settando as os métodos de crossover e mutação que serão usados
	crossover = crossover_function(crossType)
	mutation = mutation_function(mutationType)
	# População inicial aleatória
	pop = [create_permutations(bits) for _ in range(n_pop)]
	# Define o primeiro indivíduo como melhor 
	best, best_eval = 0, objective(pop[0])
	# Laço para cada geração
	for gen in range(n_iter):
		# Avalia os indivíduos da população
		scores = [objective(c) for c in pop]
		# Verifica a nova melhor solução
		for i in range(n_pop):
			if scores[i] < best_eval:
				best, best_eval = pop[i], scores[i]
				#print(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))
		if elitism: # Considera elitismo
			children = elitismo(pop,scores)
			pop_len = n_pop - len(children)
		else:
			children = list()
			pop_len = n_pop
		# Seleciona os pais
		selected = [selection(pop, scores) for _ in range(n_pop)]
		# Cria a próxima geração
		for i in range(0, pop_len, 2):
			# Pega os pares de pais
			p1, p2 = selected[i], selected[i+1]
			# Realiza crossover e mutação
			for c in crossover(p1, p2, r_cross):
				# mutação
				mutation(c, r_mut)
				# armazena para a próxima geração
				children.append(c)
		# substitui a próxima população
		pop = children
	return [best, best_eval]