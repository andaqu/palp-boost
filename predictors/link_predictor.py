from __future__ import division
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import numpy as np

class LinkPredictor:

    def __init__(self, split, G_train,  seed:int=0):
        self.split = split
        self.G_train = G_train
        self.seed = seed

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def get_roc_score(self, edges_pos, edges_neg, score_matrix, apply_sigmoid=False):

        # Edge case
        if len(edges_pos) == 0 or len(edges_neg) == 0:
            return (None, None, None)

        # Store positive edge predictions
        preds_pos = []
        for edge in edges_pos:
            score = score_matrix[edge[0], edge[1]]
            if apply_sigmoid == True:
                preds_pos.append(self.sigmoid(score))
            else:
                preds_pos.append(score)
            
        # Store negative edge predictions
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
        
        return roc_score, ap_score

    def get_ebunch(self, split):

        adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false, _ = split
    
        test_edges_list = test_edges.tolist() 
        test_edges_list = [tuple(node_pair) for node_pair in test_edges_list] 
        test_edges_false_list = test_edges_false.tolist()
        test_edges_false_list = [tuple(node_pair) for node_pair in test_edges_false_list]

        return (test_edges_list + test_edges_false_list)