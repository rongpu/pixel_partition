from __future__ import division, print_function
import numpy as np
import math

def faster_std(a):
    '''
    3x times faster than np.std
    '''
    m = a.mean()
    return math.sqrt((np.dot(a, a)/a.size) - m**2)

class Subsampler():

    def __init__(self, x, y, labels, neighbors, weights=None, **kwargs):

        self.pop_size = kwargs['pop_size']   
        self.surv_rate = kwargs['surv_rate']
        self.mutate_rate = kwargs['mutate_rate']
        self.equality_weight = kwargs['equality_weight']
        self.tournament_k = kwargs['tournament_k']
        if 'spherical' in kwargs:
            self.spherical = kwargs['spherical']
        else:
            self.spherical = 1 # True
        if 'cython' in kwargs:
            self.cython = kwargs['cython']
        else:
            self.cython = True
        self.npix = len(x)
        self.x, self.y = x.astype(np.float64), y.astype(np.float64)
        self.neighbors = neighbors.astype(np.int32)
        if weights is None:
            self.weights = np.ones(self.npix, dtype=np.float64)
        else:
            self.weights = weights.astype(np.float64)
        ngroup = len(np.unique(labels))
        # Check if labels satisfied the requirements:
        if (labels.min()!=0) or (labels.max()!=ngroup-1):
            raise ValueError('labels must be consecutive integers starting with 0')
        self.ngroup = ngroup
        self.average_count = np.sum(weights)/self.ngroup

        # generate a population of solutions by repeatedly copying of the labels
        self.labels_all = np.tile(labels, self.pop_size).reshape(self.pop_size, self.npix).astype(np.int32)

        if self.cython:
            try:
                from fitness_function_cython_engine import fitness_function_cython_engine
                self.fitness_cython = fitness_function_cython_engine
                print('Cython function loaded')
            except ImportError:
                msg = ("The cython module must be compiled first via "
                    "``python setup.py build_ext --inplace``")
                raise ImportError(msg)

    def fitness(self):

        # compactness score: lower the better
        compactness = np.zeros(self.pop_size)
        # equality score: lower the better
        equality = np.zeros(self.pop_size)
        # weighted counts of each group in each solution
        counts = np.zeros((self.pop_size, self.ngroup))
        # total score
        scores = np.zeros(self.pop_size)

        if self.cython:
            # use cython function to calculate fitness score
            self.fitness_cython(self.x, self.y, self.labels_all, self.neighbors, self.weights, self.spherical, 
                compactness, equality, counts, scores)
        else:
            # use python function to calculate fitness score
            # loop through each individual (solution) in the population
            for idx_pop in range(self.pop_size):
                labels = self.labels_all[idx_pop]
                s = labels.argsort()
                label_edges = np.where(np.ediff1d(labels[s]))[0] + 1

                # solutions with missing labels are giving the lowest score:
                if len(label_edges) < self.ngroup - 1:
                    equality[idx_pop] = -np.inf
                    continue

                # loop through each group
                label_edges = np.append(np.insert(label_edges, 0, 0), self.npix)
                for idx_grp in range(self.ngroup):
                    k1 = label_edges[idx_grp]
                    k2 = label_edges[idx_grp+1]
                    members = s[k1:k2]
                    counts[idx_pop, idx_grp] = np.sum(self.weights[members])
                    if self.spherical:
                        x_rescale = math.cos(np.mean(self.y[members])/180*np.pi)
                    else:
                        x_rescale = 1.
                    compactness[idx_pop] += math.sqrt(x_rescale * faster_std(self.x[members])**2 \
                        + faster_std(self.y[members])**2)
                equality[idx_pop] = self.equality_weight * faster_std(counts[idx_pop])/(self.average_count)

        # # total score: higher the better
        # scores = -(compactness + equality)

        # self.scores = scores
        # self.counts = counts
        # self.compactness = compactness
        # self.equality = equality

        pass

    def selection(self): 
        '''
        Select solutions for the next generation; 
        '''
        # select top scorers as candidates
        candidates = np.argsort(self.scores)[-int(self.pop_size*self.surv_rate):]
        # Tournament Selection - obtain a list of survivers with the original size
        survivers = np.zeros(self.pop_size, dtype=int)
        for idx_pop in range(self.pop_size):
            t_candidates = np.random.choice(candidates, size=self.tournament_k, replace=False)
            # select the highest scorer
            survivers[idx_pop] = t_candidates[np.argmax(self.scores[t_candidates])]
        
        return survivers

    def mutate(self):
        '''
        Only pixels that are bordering a different group are mutated
        '''
        for idx_pop in range(self.pop_size):
            labels = np.copy(self.labels_all[idx_pop])

            # find bordering pixels (to speed up calculation)
            mask1 = labels[self.neighbors] != labels[:, None]
            mask2 = self.neighbors>=0
            mask_border = np.any(mask1 & mask2, axis=1)
            # id of border pixels
            border_idx = np.where(mask_border)[0]

            # mutate with a probablity
            mutator = np.where(np.random.rand(len(border_idx)) < self.mutate_rate)[0]
            for idx_pixel in border_idx[mutator]:
                # pixel id of neighboring groups
                neighbors_idx = self.neighbors[idx_pixel]

                # not an empty neighbor
                mask = neighbors_idx>=0
                # # not the same group
                # mask &= labels[idx_pixel]!=labels[neighbors_idx]

                neighbors_idx = neighbors_idx[mask]
                if len(neighbors_idx)>0:
                    labels[idx_pixel] = np.random.choice(labels[neighbors_idx])
            self.labels_all[idx_pop] = labels

        pass
