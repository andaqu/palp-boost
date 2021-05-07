from .link_predictor import LinkPredictor
import networkx as nx
import numpy as np

class AdamicAcar(LinkPredictor):

    def predict(self):
        if self.G_train.is_directed(): 
            G_train_undirected = self.G_train.to_undirected()

        adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false, feat = self.split

        score_matrix = np.zeros(adj_train.shape, dtype='uint8')
        for u, v, p in nx.adamic_adar_index(G_train_undirected, ebunch=self.get_ebunch(self.split)):
            score_matrix[u][v] = p
            score_matrix[v][u] = p

        score_matrix = score_matrix / score_matrix.max()

        

        test_roc, test_ap = self.get_roc_score(test_edges, test_edges_false, score_matrix)

        r = {"Test ROC": test_roc, "Test AP": test_ap}
        print(r)

        return r 