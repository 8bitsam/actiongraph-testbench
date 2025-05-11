from sklearn.metrics import pairwise_distances
import numpy as np

class WeightedCosineDistance:
    def __init__(self, chem_weight=0.7):
        self.chem_weight = chem_weight
        
    def __call__(self, X, Y):
        # Ensure inputs are 2D arrays
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        
        chem_dist = pairwise_distances(X[:, :123], Y[:, :123], metric='cosine')
        graph_dist = pairwise_distances(X[:, 123:], Y[:, 123:], metric='cosine')
        return self.chem_weight * chem_dist + (1 - self.chem_weight) * graph_dist
