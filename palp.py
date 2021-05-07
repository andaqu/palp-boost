import numpy as np

class PALP:

    def __init__(self, feat, centroids, weights, shape):

        self.feat = feat
        self.centroids = centroids
        self.weights = weights
        self.shape = shape
        
    def get_personality_score_matrix(self, edges):

        score_matrix = np.zeros(self.shape, dtype='float16')

        for u, v in edges:
            score_matrix[u][v] = self.score(u, v)

        return score_matrix 

    def score(self, u, v):

        score = 0
        p_v = self.feat[v]

        for i, c in enumerate(self.centroids[u]):
            score += self.weights[u][i] * 1/np.linalg.norm(p_v-c)

        return score

