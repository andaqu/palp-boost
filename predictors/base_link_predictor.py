from __future__ import division
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import numpy as np

class LinkPredictor:

    def __init__(self, G_train, split, type="float32", boost_with=None, seed:int=0):
        self.adj_train, self.train_edges, self.train_edges_false, self.val_edges, self.val_edges_false, self.test_edges, self.test_edges_false, self.feat = split
        self.score_matrix = np.zeros(self.adj_train.shape, dtype=type)
        self.G_train = G_train
        self.G_train_undirected = self.G_train.to_undirected()
        self.palp = boost_with
        self.seed = seed

    def predict(self):
        raise NotImplementedError

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def evaluate(self, score_matrix, edges_pos, edges_neg, apply_sigmoid=False):

        if len(edges_pos) == 0 or len(edges_neg) == 0:
            return (None, None, None)

        preds_pos = []
        for edge in edges_pos:
            score = score_matrix[edge[0], edge[1]]
            if apply_sigmoid == True:
                preds_pos.append(self.sigmoid(score))
            else:
                preds_pos.append(score)
            
        preds_neg = []
        for edge in edges_neg:
            score = score_matrix[edge[0], edge[1]]
            if apply_sigmoid == True:
                preds_neg.append(self.sigmoid(score))
            else:
                preds_neg.append(score)
            
        y_score = np.hstack([preds_pos, preds_neg])
        y_true = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])

        roc_score = roc_auc_score(y_true, y_score)
        roc_curve_tuple = roc_curve(y_true, y_score)
        ap_score = average_precision_score(y_true, y_score)
        
        return roc_score*100, ap_score*100

    def evaluate_classifier(self, classifier, embs, labels):

        preds = classifier.predict_proba(embs)[:, 1]
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)

        return roc*100, ap*100

    def get_ebunch(self):

        test_edges_list = self.test_edges.tolist() 
        test_edges_list = [tuple(node_pair) for node_pair in test_edges_list] 
        test_edges_false_list = self.test_edges_false.tolist()
        test_edges_false_list = [tuple(node_pair) for node_pair in test_edges_false_list]

        return (test_edges_list + test_edges_false_list)
