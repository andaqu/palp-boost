from sklearn.metrics import silhouette_samples, silhouette_score
from gae.preprocessing import mask_test_edges_directed
from sklearn.cluster import KMeans
from collections import Counter
import networkx as nx
from tqdm import tqdm
import numpy as np
import pickle
import os

def build_graph(N:int, personality_mode:str="normalised", selfloops:bool=False, components:str=""):

    name = f"twitter-{N}{f'-{components}' if components != '' else ''}"
    path = f"saved/{name}.pkl"

    if os.path.exists(path):
        print(f"Graph already exists! Ignoring build call...")
        return name

    if personality_mode not in ["normalised", "raw"] or components not in ["strong", "weak", ""]:
        print("Revise parameters!")
        return ""

    # Read edge-list for G (excluding ego-nodes)
    edgelist = []

    print("Reading edge lists...")
    for ego in os.listdir("data/network/filtered"):

        ego = ego.split(".")[0]  

        with open(f"data/network/filtered/{ego}.edges") as f:
            edgelist.extend(f.read().split("\n"))

    # Construct and subset G accordingly
    if N > 1: edgelist = edgelist[:int(len(edgelist)/N)]

    print("Constructing G...")
    G = nx.parse_edgelist(edgelist, nodetype=int, create_using=nx.DiGraph())
    G.remove_edges_from(nx.selfloop_edges(G))

    if components == "strong":
        G = G.subgraph(max(nx.strongly_connected_components(G), key=len))
    elif components == "weak":
        G = G.subgraph(max(nx.weakly_connected_components(G), key=len)) 

    print(f"Constructed G: [{G.number_of_nodes()}] nodes and [{G.number_of_edges()}] edges.")

    # Read personality from CSV and populate G
    with open(f"data/{personality_mode}_twitter_personality.csv") as f:
        data = f.read().split()
        pers = {}

        for row in data[1:]:
            row = row.split(",")
            pers[int(row[0])] = [float(x) for x in row[1:]]

    print("Populating with personality...")
    for node in set(G.nodes()):
        G.nodes[node]["pers"] = pers[node]

    # Save adjacement matrix and feature list
    adj = nx.adjacency_matrix(G)
    feat = np.zeros((G.number_of_nodes(), 5)) 

    for i, node in enumerate(G.nodes()):
        feat[i] = G.nodes[node]["pers"]

    with open(path, "wb") as f:
        pickle.dump((adj, feat), f)

    return name
    
def load_graph(to_load:str):

    path = f"saved/{to_load}.pkl"

    if not os.path.exists(path):
        print(f"Graph {to_load} does not exist! Call `build_graph()` first...")
        return []

    with open(path, "rb") as f:
        graph = pickle.load(f)

    adj = graph[0]
    feat = graph[-1]

    G = nx.from_numpy_matrix(adj.todense(), create_using=nx.DiGraph())

    for i, node in enumerate(G.nodes()):
        G.nodes[node]["pers"] = feat[i].tolist()

    return G

def get_split(to_split:str, verbose:bool=False):

    try:
        test = float(to_split.split("-")[-1])
    except:
        print("Re-evaluate `to_split` parameter. Ensure it ends by a '-' followed by a float. Aborting...")
        return

    path = f"saved/{to_split}.pkl"

    if os.path.exists(path):
        print(f"Split already exists! Returning {to_split}.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)

    # Ensures consistent training_testing splits
    np.random.seed(0)

    with open(f"saved/{to_split.rsplit('-', 1)[0]}.pkl", "rb") as f:
        adj, feat = pickle.load(f)

    print(f"Splitting network... training: [{round(1-test-0.1, 1)}] testing: [{test}] validation: [0.1]")
    train_test_split = mask_test_edges_directed(adj, test_frac=test, val_frac=0.1, verbose=verbose, prevent_disconnect=False, false_edge_sampling='random')
    
    with open(path, "wb") as f:
        pickle.dump(train_test_split +  (feat,), f, protocol=2)

    return train_test_split +  (feat,)

def get_ideal_centroids_for_node(node, p_u, P_v, k_max:int=10):

    scores = {}
    centroids = {}
    weights = {}

    for k in range(2,k_max):

        if len(P_v) < k:
            break

        km = KMeans(n_clusters=k, random_state=0)
        km.fit_predict(P_v)

        labels = km.labels_
        weights[k] = dict(Counter(labels))
        centroids[k] = km.cluster_centers_

        if len(set(labels)) > len(P_v) - 1:
            break
   
        score = silhouette_score(P_v, labels, metric='euclidean')
        scores[k] = score

    if len(scores) == 0:
        return np.array([p_u]), {0: 1}

    ideal_k = max(scores, key=scores.get)

    centroids_ = centroids[ideal_k]
    weights_ = weights[ideal_k]

    factor = sum(weights_.values())
    for n in weights_:
        weights_[n] = weights_[n] / factor

    return centroids_ , weights_

# For twitter-10-strong: {1: 644, 2: 2654, 3: 679, 4: 327, 5: 183, 6: 125, 7: 86, 8: 66, 9: 61}
def build_ideal_centroids(G, to_save:str, k_max:int=10):

    path = f"saved/ideal_centroids/{to_save}-ideal_centroids.pkl"

    if os.path.exists(path):
        print(f"Ideal centroids for {to_save} already exist! Ignoring build call...")
        return

    centroids = {}
    weights = {}
    meta = {}

    for i in range(1, k_max):
        meta[i] = 0

    for node in tqdm(G.nodes):

        P_v = [G.nodes[v]["pers"] for u, v in G.out_edges(node)]
        p_u = G.nodes[node]["pers"]

        centroids_, weights_ = get_ideal_centroids_for_node(node, p_u, P_v, k_max=k_max)

        centroids[node] = centroids_
        weights[node] = weights_
        meta[len(centroids_)] += 1

    with open(path, "wb") as f:
        pickle.dump((centroids, weights, meta), f)

def load_ideal_centroids(to_load:str):

    path = f"saved/ideal_centroids/{to_load}-ideal_centroids.pkl"

    if not os.path.exists(path):
        print(f"Centroids for {to_load} do not exist! Call `build_ideal_centroids()` first...")
        return []

    with open(path, "rb") as f:
        return pickle.load(f)
