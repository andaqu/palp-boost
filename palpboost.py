from scipy.sparse import csr_matrix
import helper as hlpr
import numpy as np

class PALPBoost:

	# The score mode reflects how the edge (u -> v) should be weighted
	# SOLO : Based on the inverse distance between p_u and p_v
	# CLUSTER : Based on the inverse distance between the follow tendencies of u and p_v
	class ScoreMode:
		SOLO = 0
		CLUSTER = 1

	def __init__(self, G_train, name, split, eps, calc_upon_init=True, score_mode:ScoreMode=ScoreMode.CLUSTER):

		# For the given train subset of a graph, build each node's clusters i.e. follow personality tendecies
		hlpr.build_ideal_centroids(G_train, to_save=name)
		self.centroids, self.weights, _ = hlpr.load_ideal_centroids(to_load=name)

		adj_train, _, _, self.val_edges, self.val_edges_false, self.test_edges, self.test_edges_false, self.feat = split

		self.score_matrix = np.zeros(adj_train.shape, dtype='float32')
		self.eps = eps
		self.score_mode = score_mode

		# The analysis function `plot_palp_scores` requires `calc_upon_init` to be disabled
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
		p_u = self.feat[u]
		p_v = self.feat[v]

		if self.score_mode == PALPBoost.ScoreMode.SOLO:

			d = np.linalg.norm(p_v-p_u)
			score = 1 / (d + self.eps)

		elif self.score_mode == PALPBoost.ScoreMode.CLUSTER:

			for i, c in enumerate(self.centroids[u]):
				d = np.linalg.norm(p_v-c)
				score += self.weights[u][i] / (d + self.eps)
				
		return score
