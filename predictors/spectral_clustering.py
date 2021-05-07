from sklearn.manifold import spectral_embedding
from .link_predictor import LinkPredictor
import numpy as np

class SpectralClustering(LinkPredictor):

    def predict(self):

        adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false, feat = self.split

        spectral_emb = spectral_embedding(adj_train, n_components=16, random_state=self.seed)
        score_matrix = np.dot(spectral_emb, spectral_emb.T)

        test_roc, test_ap = self.get_roc_score(test_edges, test_edges_false, score_matrix, apply_sigmoid=True)
        val_roc, val_ap = self.get_roc_score(val_edges, val_edges_false, score_matrix, apply_sigmoid=True)

        r = {"Test ROC": test_roc, "Test AP": test_ap, "Validation ROC": val_roc, "Validation AP": val_ap}
        print(r)

        return r 