# Instead of genetic algorithm, here a simple greedy algorithm is used. 

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

    def __init__(self, x, y, ngroup, neighbors, weights=None, **kwargs):

        self.mutate_rate = kwargs['mutate_rate']
        self.equality_weight = kwargs['equality_weight']
        if 'spherical' in kwargs:
            self.spherical = kwargs['spherical']
        else:
            self.spherical = True
        self.npix = len(x)
        self.x, self.y = x, y
        self.neighbors = neighbors
        if weights is None:
            self.weights = np.ones(self.npix)
        else:
            self.weights = weights
        self.ngroup = ngroup
        self.average_count = np.sum(weights)/self.ngroup

    def fitness(self, labels):

        # weighted counts of each group in each solution
        counts = np.zeros(self.ngroup)

        s = labels.argsort()
        label_edges = np.where(np.ediff1d(labels[s]))[0] + 1
        compactness = 0

        # solutions with missing labels are giving the lowest score:
        if len(label_edges) < self.ngroup - 1:
            equality = np.inf
            compactness = np.inf
        else:
            label_edges = np.append(np.insert(label_edges, 0, 0), self.npix)
            for idx_grp in range(self.ngroup):
                k1 = label_edges[idx_grp]
                k2 = label_edges[idx_grp+1]
                members = s[k1:k2]
                counts[idx_grp] = np.sum(self.weights[members])
                if self.spherical:
                    x_rescale = math.cos(np.mean(self.y[members])/180*np.pi)
                else:
                    x_rescale = 1.
                # compactness score: lower the better
                compactness += math.sqrt(x_rescale * faster_std(self.x[members])**2 \
                    + faster_std(self.y[members])**2)
            # equality score: lower the better
            equality = self.equality_weight * faster_std(counts)/(self.average_count)

        # total score: higher the better
        score = -(compactness + equality)

        return score, compactness, equality, counts

    def mutate(self, labels):
        '''
        Only pixels that are bordering a different group are mutated
        '''
        labels_new = np.copy(labels)

        # find bordering pixels
        mask1 = labels_new[self.neighbors] != labels_new[:, None]
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
            # mask &= labels_new[idx_pixel]!=labels_new[neighbors_idx]

            neighbors_idx = neighbors_idx[mask]
            if len(neighbors_idx)>0:
                labels_new[idx_pixel] = np.random.choice(labels_new[neighbors_idx])

        return labels_new
