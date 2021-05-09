from sklearn.manifold import spectral_embedding
from .base_link_predictor import LinkPredictor
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class SpectralClustering(LinkPredictor):

    def predict(self, components=8):

        print("[SpectralClustering]")

        spectral_emb = spectral_embedding(self.adj_train, n_components=components, random_state=self.seed)
        self.score_matrix = np.dot(spectral_emb, spectral_emb.T)

        test_roc, test_ap = self.evaluate(self.score_matrix, self.test_edges, self.test_edges_false)
        val_roc, val_ap = self.evaluate(self.score_matrix, self.val_edges, self.val_edges_false)

        print({"Test ROC": test_roc, "Test AP": test_ap, "Val ROC": val_roc, "Val AP": val_ap})

        if self.palp:
            boosted_score_matrix = self.score_matrix + self.palp.score_matrix
            test_roc, test_ap = self.evaluate(boosted_score_matrix, self.test_edges, self.test_edges_false)
            val_roc, val_ap = self.evaluate(boosted_score_matrix, self.val_edges, self.val_edges_false)

            print({"Test ROC (B)": test_roc, "Test AP (B)": test_ap, "Val ROC (B)": val_roc, "Val AP (B)": val_ap})