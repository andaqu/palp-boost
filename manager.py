from predictors import SpectralClustering, AdamicAcar, JaccardCoefficient, PreferentialAttachment
from palpboost import PALPBoost
import helper as hlpr
import networkx as nx
import analyser as an
import numpy as np

name = hlpr.build_graph(N=10, components="strong")
name = f"{name}-0.2"

split = hlpr.get_split(name)
G_train = hlpr.load_graph(name)

# palp = PALPBoost(G_train, name, split, eps=1, calc_upon_init=False)
# an.plot_palp_scores(palp)

palp = PALPBoost(G_train, name, split, eps=120)

AA = AdamicAcar(G_train, split, type="uint8", boost_with=palp)
AA.predict()

# JC = JaccardCoefficient(G_train, split, boost_with=palp)
# JC.predict()

# PA = PreferentialAttachment(G_train, split, boost_with=palp)
# PA.predict()

# SC = SpectralClustering(G_train, split, boost_with=palp)
# SC.predict()