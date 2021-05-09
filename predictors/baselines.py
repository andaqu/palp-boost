from .base_link_predictor import LinkPredictor
import networkx as nx
import numpy as np

class AdamicAcar(LinkPredictor):

    def predict(self):

        print("[AdamicAdar]")

        for u, v, p in nx.adamic_adar_index(self.G_train_undirected, ebunch=self.get_ebunch()): 
            self.score_matrix[u][v] = p
            self.score_matrix[v][u] = p 

        self.score_matrix = self.score_matrix / self.score_matrix.max()

        test_roc, test_ap = self.evaluate(self.score_matrix, self.test_edges, self.test_edges_false)
        print({"Test ROC": test_roc, "Test AP": test_ap})

        if self.palp:
            score_matrix_boosted = self.score_matrix + self.palp.score_matrix
            test_roc, test_ap = self.evaluate(score_matrix_boosted, self.test_edges, self.test_edges_false)
            print({"Test ROC (B)": test_roc, "Test AP (B)": test_ap})

class JaccardCoefficient(LinkPredictor):

    def predict(self):

        print("[JaccardCoefficient]")

        for u, v, p in nx.jaccard_coefficient(self.G_train_undirected, ebunch=self.get_ebunch()): 
            self.score_matrix[u][v] = p
            self.score_matrix[v][u] = p 

        self.score_matrix = self.score_matrix / self.score_matrix.max()

        test_roc, test_ap = self.evaluate(self.score_matrix, self.test_edges, self.test_edges_false)
        print({"Test ROC": test_roc, "Test AP": test_ap})

        if self.palp:
            score_matrix_boosted = self.score_matrix + self.palp.score_matrix
            test_roc, test_ap = self.evaluate(score_matrix_boosted, self.test_edges, self.test_edges_false)
            print({"Test ROC (B)": test_roc, "Test AP (B)": test_ap})

class PreferentialAttachment(LinkPredictor):

    def predict(self):

        print("[PreferentialAttachment]")

        for u, v, p in nx.preferential_attachment(self.G_train_undirected, ebunch=self.get_ebunch()): 
            self.score_matrix[u][v] = p
            self.score_matrix[v][u] = p 

        self.score_matrix = self.score_matrix / self.score_matrix.max()

        test_roc, test_ap = self.evaluate(self.score_matrix, self.test_edges, self.test_edges_false)
        print({"Test ROC": test_roc, "Test AP": test_ap})

        if self.palp:
            score_matrix_boosted = self.score_matrix + self.palp.score_matrix
            test_roc, test_ap = self.evaluate(score_matrix_boosted, self.test_edges, self.test_edges_false)
            print({"Test ROC (B)": test_roc, "Test AP (B)": test_ap})