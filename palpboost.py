from scipy.sparse import csr_matrix
import helper as hlpr
import numpy as np

class PALPBoost:

    def __init__(self, G_train, name, split, eps, calc_upon_init=True):

        hlpr.build_ideal_centroids(G_train, to_save=name)
        self.centroids, self.weights, _ = hlpr.load_ideal_centroids(to_load=name)

        adj_train, _, _, self.val_edges, self.val_edges_false, self.test_edges, self.test_edges_false, self.feat = split

        self.score_matrix = np.zeros(adj_train.shape, dtype='float32')
        self.eps = eps

        if calc_upon_init:
            self.calc_score_matrix(self.val_edges, self.val_edges_false, self.test_edges, self.test_edges_false)
        
    def empty_score_matrix(self):
        self.score_matrix = np.zeros(self.score_matrix.shape, dtype='float32')
        
    def calc_score_matrix(self, *args):

        edges = np.concatenate(args)

        for u, v in edges:
            self.score_matrix[u][v] = self.score(u, v)

        return self.score_matrix

    def score(self, u, v):

        score = 0
        p_v = self.feat[v]

        for i, c in enumerate(self.centroids[u]):
            d = np.linalg.norm(p_v-c)
            score += self.weights[u][i] / (d+self.eps)

        return score
