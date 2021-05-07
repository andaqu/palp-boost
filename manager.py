from predictors import SpectralClustering, AdamicAcar
import numpy as np
from palp import PALP
import helper as hlpr
import networkx as nx

name = hlpr.build_graph(N=10, components="strong")
name = f"{name}-0.2"

split = hlpr.get_split(name)
G = hlpr.load_graph(name)

hlpr.build_ideal_centroids(G, to_save=name)
centroids, weights, meta = hlpr.load_ideal_centroids(to_load=name)

adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false, feat = split

palp = PALP(feat, centroids, weights, adj_train.shape)

personality_score_matrix = palp.get_personality_score_matrix(np.concatenate((test_edges, test_edges_false)))

# SC = SpectralClustering(split, G)
# SC.predict(P_score_matrix)

average = personality_score_matrix[personality_score_matrix!=0].mean()
print(average)

personality_score_matrix[personality_score_matrix < average + 17] = -0.5

AA = AdamicAcar(split, G)
AA.predict()

# SC = SpectralClustering(split, G)
# SC.predict(P_score_matrix)
