from sklearn.linear_model import LogisticRegression
from .base_link_predictor import LinkPredictor
from gensim.models import Word2Vec
from enum import Enum  
import networkx as nx
import numpy as np
import random

class Node2Vec(LinkPredictor):

	def predict(self, p=1, q=1, size=10, walks=10, length=80, dim=128, workers=8, iter=1):

		print("[Node2Vec]")

		n2v = n2vGraph(self.G_train, is_directed=True, p=p, q=q)

		n2v.preprocess_transition_probs()

		walks = n2v.simulate_walks(walks, length, verbose=True)
		walks = [list(map(str, walk)) for walk in walks] 

		model = Word2Vec(walks, size=dim, window=size, min_count=0, sg=1, workers=workers, iter=iter)
		embeddings = model.wv

		emb_list = []
		for node_index in range(0, self.adj_train.shape[0]):
			node_embedding = embeddings[str(node_index)]
			emb_list.append(node_embedding)
		self.emb_matrix = np.vstack(emb_list)

		self.score_matrix = np.dot(self.emb_matrix, self.emb_matrix.T)

		test_roc, test_ap = self.evaluate(self.score_matrix, self.test_edges, self.test_edges_false)
		val_roc, val_ap = self.evaluate(self.score_matrix, self.val_edges, self.val_edges_false)

		print({"Test ROC": test_roc, "Test AP": test_ap, "Val ROC": val_roc, "Val AP": val_ap})

		if self.palp:
			score_matrix_boosted = self.score_matrix + self.palp.score_matrix
			test_roc, test_ap = self.evaluate(score_matrix_boosted, self.test_edges, self.test_edges_false)
			val_roc, val_ap = self.evaluate(score_matrix_boosted, self.val_edges, self.val_edges_false)
			
			print({"Test ROC (B)": test_roc, "Test AP (B)": test_ap, "Val ROC (B)": val_roc, "Val AP (B)": val_ap})

class n2vGraph():

	def __init__(self, nx_G, is_directed, p, q):
		self.G = nx_G
		self.is_directed = is_directed
		self.p = p
		self.q = q

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]
			cur_nbrs = sorted(G.neighbors(cur))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					walk.append(cur_nbrs[self.alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					next = cur_nbrs[self.alias_draw(alias_edges[(prev, cur)][0], 
						alias_edges[(prev, cur)][1])]
					walk.append(next)
			else:
				break

		return walk

	def simulate_walks(self, num_walks, walk_length, verbose=False):
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G
		walks = []
		nodes = list(G.nodes())
		if verbose == True:
			print('Walk iteration:')
		for walk_iter in range(num_walks):
			if verbose == True:
				print(str(walk_iter+1), '/', str(num_walks))
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

		return walks

	def get_alias_edge(self, src, dst):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		G = self.G
		p = self.p
		q = self.q

		unnormalized_probs = []
		for dst_nbr in sorted(G.neighbors(dst)):
			if dst_nbr == src:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
			elif G.has_edge(dst_nbr, src):
				unnormalized_probs.append(G[dst][dst_nbr]['weight'])
			else:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		return self.alias_setup(normalized_probs)

	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		G = self.G
		is_directed = self.is_directed

		alias_nodes = {}
		for node in G.nodes():
			unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
			norm_const = sum(unnormalized_probs)
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			alias_nodes[node] = self.alias_setup(normalized_probs)

		alias_edges = {}
		triads = {}

		if is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges

		return

	def alias_setup(self, probs):
		'''
		Compute utility lists for non-uniform sampling from discrete distributions.
		Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
		for details
		'''
		K = len(probs)
		q = np.zeros(K)
		J = np.zeros(K, dtype=np.int)

		smaller = []
		larger = []
		for kk, prob in enumerate(probs):
			q[kk] = K*prob
			if q[kk] < 1.0:
				smaller.append(kk)
			else:
				larger.append(kk)

		while len(smaller) > 0 and len(larger) > 0:
			small = smaller.pop()
			large = larger.pop()

			J[small] = large
			q[large] = q[large] + q[small] - 1.0
			if q[large] < 1.0:
				smaller.append(large)
			else:
				larger.append(large)

		return J, q

	def alias_draw(self, J, q):
		'''
		Draw sample from a non-uniform discrete distribution using alias sampling.
		'''
		K = len(J)

		kk = int(np.floor(np.random.rand()*K))
		if np.random.rand() < q[kk]:
			return kk
		else:
			return J[kk]