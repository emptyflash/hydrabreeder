# genetic algorithm search of the one max optimization problem
from numpy.random import randint

# tournament selection
def selection(pop, scores, k=3):
	# first random selection
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]

# genetic algorithm
async def genetic_algorithm(creation, fitness, mutation, crossover, n_iter, n_pop, r_cross, r_mut):
    # initial population of random bitstring
    pop = [creation() for _ in range(n_pop)]
    # keep track of best solution
    best, best_eval = pop[0], await fitness(pop[0])
    yield [best, best_eval]
    # enumerate generations
    for gen in range(n_iter):
        print(f"Starting generation {gen}")
        # evaluate all candidates in the population
        scores = [await fitness(c) for c in pop]
        # check for new best solution
        for i in range(n_pop):
            if scores[i] > best_eval:
                best, best_eval = pop[i], scores[i]
        yield sorted(zip(pop, scores), key=lambda p: p[1], reverse=True)[0]
        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                c = mutation(c, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children
    yield [best, best_eval]

