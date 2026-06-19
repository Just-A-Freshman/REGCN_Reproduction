import csv
import numpy as np
import random
import pandas as pd
import copy
from vmdpy import VMD

tau = 0.
DC = 0
init = 1
tol = 1e-7

def Fun(x, data1, low, ub):
    K = int(x[0])
    alpha = int(x[1])

    if K <= low[0]:
        K = low[0]
    if K >= ub[0]:
        K = ub[0]

    if alpha <= low[1]:
        alpha = low[1]
    if alpha >= ub[1]:
        alpha = ub[1]
    s = 0
    n_features = data1.shape[1]
    for i in range(n_features):
        try:
            u, u_hat, omega = VMD(data1[:,i], alpha, tau, K, DC, init, tol)
        except Exception:
            return float('inf')
        if np.any(np.isnan(u)) or np.any(np.isinf(u)):
            return float('inf')
        u1 = np.sum(u, axis=0)
        u2 = list(map(lambda x: x[0] - x[1], zip(data1[:,i], u1)))

        u3 = list(map(list, zip(*[data1[:,i],u2])))
        df = pd.DataFrame(u3)
        corr_val = df.corr()[0][1]
        s += abs(corr_val) if not np.isnan(corr_val) else 0
    s /= n_features

    return s

class GAIndividual:
    '''

    创建pop中的单个个体
    '''

    def __init__(self, vardim, bound, x1):
        self.vardim = vardim
        self.bound = bound
        self.low = bound[0]
        self.ub = bound[1]
        self.fitness = 0
        self.x1 = x1

    def generate(self):
        '''
        generate a random chromsome for genetic algorithm
        '''
        len = self.vardim
        rnd = np.random.random(size=len)
        self.chrom = np.zeros(len)
        for i in range(0, len):
            self.chrom[i] = self.bound[0][i] + \
                            (self.bound[1][i] - self.bound[0][i]) * rnd[i]
        print(int(self.chrom[0]), self.chrom[1])

    def calculateFitness(self):
        self.fitness = Fun(self.chrom, self.x1, self.low, self.ub)

class GeneticAlgorithm:

    def __init__(self, sizepop, vardim, bound, MAXGEN, params, x1, dataset='SSE'):
        self.sizepop = sizepop
        self.MAXGEN = MAXGEN
        self.vardim = vardim
        self.bound = bound
        self.population = []
        self.fitness = np.zeros((self.sizepop, 1))
        self.trace = np.zeros((self.MAXGEN, 2))
        self.params = params
        self.x1 = x1
        self.dataset = dataset
        # self.i = i
    def initialize(self):

        for i in range(0, self.sizepop):
            ind = GAIndividual(self.vardim, self.bound,self.x1)
            ind.generate()
            self.population.append(ind)

    def evaluate(self):

        for i in range(0, self.sizepop):
            self.population[i].calculateFitness()
            self.fitness[i] = self.population[i].fitness

    def solve(self):

        self.t = 0  # 迭代次数
        self.initialize()  # 初始化种群
        self.evaluate()  # 计算适应度
        best = np.min(self.fitness)  # 选出适应度最小的个体
        bestIndex = np.argmin(self.fitness)  # 最小适应度的索引
        self.best = copy.deepcopy(self.population[bestIndex])
        self.avefitness = np.mean(self.fitness)  # 平均适应度
        self.BEST = []
        while (self.t < self.MAXGEN):
            print('迭代次数：', self.t)
            self.t += 1
            self.selectionOperation()  # 选择
            self.crossoverOperation()  # 交叉
            self.mutationOperation()  # 变异
            self.evaluate()  # 重新计算新种群适应度
            best = np.min(self.fitness)
            bestIndex = np.argmin(self.fitness)
            if best < self.best.fitness:
                self.best = copy.deepcopy(self.population[bestIndex])
            self.avefitness = np.mean(self.fitness)
            self.BEST.append(self.best)

        print("Optimal solution is:", int(self.best.chrom[0]), int(self.best.chrom[1]))
        result = [int(self.best.chrom[0]), int(self.best.chrom[1])]
        with open('../result/Table4/' + self.dataset + '_GA.csv', 'a',newline='', encoding='UTF8',) as f:
            d = csv.writer(f)
            d.writerow(result)


    def selectionOperation(self):
        '''
        selection operation for Genetic Algorithm
        '''
        fitness_flat = self.fitness.flatten()
        min_f = np.min(fitness_flat)
        if min_f < 0:
            fitness_flat = fitness_flat - min_f
        total = np.sum(fitness_flat)
        if total <= 0 or np.isnan(total):
            probs = np.ones(self.sizepop) / self.sizepop
        else:
            probs = fitness_flat / total
        idxs = np.random.choice(self.sizepop, size=self.sizepop, p=probs)
        self.population = [copy.deepcopy(self.population[i]) for i in idxs]

    def crossoverOperation(self):
        '''
        crossover operation for genetic algorithm
        '''
        newpop = []
        # 选出两个个体进行交换
        for i in range(0, self.sizepop, 2):
            idx1 = random.randint(0, self.sizepop - 1)
            idx2 = random.randint(0, self.sizepop - 1)
            while idx2 == idx1:
                idx2 = random.randint(0, self.sizepop - 1)
            newpop.append(copy.deepcopy(self.population[idx1]))
            newpop.append(copy.deepcopy(self.population[idx2]))
            r = random.random()

            if r < self.params[0]:
                crossPos = random.randint(1, self.vardim - 1)
                for j in range(crossPos, self.vardim):
                    p1 = newpop[i].chrom[j]
                    p2 = newpop[i + 1].chrom[j]
                    alpha = self.params[2]
                    newpop[i].chrom[j] = p1 * alpha + (1 - alpha) * p2
                    newpop[i + 1].chrom[j] = p2 * alpha + (1 - alpha) * p1
        self.population = newpop

    def mutationOperation(self):
        '''
        mutation operation for genetic algorithm
        '''
        newpop = []
        for i in range(0, self.sizepop):
            newpop.append(copy.deepcopy(self.population[i]))
            r = random.random()
            if r < self.params[1]:
                mutatePos = random.randint(0, self.vardim - 1)
                theta = random.random()
                if theta > 0.5:

                    newpop[i].chrom[mutatePos] = newpop[i].chrom[mutatePos] - \
                                                 (newpop[i].chrom[mutatePos] - self.bound[0][mutatePos]) * \
                                                 (1 - random.random() ** (1 - self.t / self.MAXGEN))
                else:

                    newpop[i].chrom[mutatePos] = newpop[i].chrom[mutatePos] + \
                                                 (self.bound[1][mutatePos] - newpop[i].chrom[mutatePos]) * \
                                                 (1 - random.random() ** (1 - self.t / self.MAXGEN))
        self.population = newpop

def run(dataset, train_rate=0.7):
    """Run GA-VMD optimisation for the given dataset."""
    data_addr = '../data/data/' + dataset + '.npy'
    data = np.load(data_addr, allow_pickle=True)
    data = data.astype(float)
    for i in range(data.shape[0]):
        tdata = data[i]
        train_size = int(tdata.shape[0] * train_rate)
        train_data = tdata[0:train_size]
        x1 = train_data
        low = [2, x1.shape[0]/2]
        ub = [5, x1.shape[0]*3]
        bound = [low, ub]
        ga = GeneticAlgorithm(60, 2, bound, 100, [0.9, 0.1, 0.5], x1, dataset)
        ga.solve()


if __name__ == '__main__':
    datasets = 'SSE'
    run(datasets)

#
#
#



