from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import metrics
import helper as hlpr
import networkx as nx
import numpy as np
import collections
import pickle
import os

def plot_followees_personality_2d(node, x=0, y=1, centroids:list=[]):

    plt.xlim([0, 1])
    plt.ylim([0, 1])

    ax = plt.axes()
    ax.set_xlabel(label[x])
    ax.set_ylabel(label[y])

    p_u = G.nodes[node]["pers"]
    plt.scatter(p_u[x], p_u[y], marker='o', color='r', label='point')

    P_v = [G.nodes[v]["pers"] for u, v in G.out_edges(node)]
    P_v_x = list(zip(*P_v))[x]
    P_v_y = list(zip(*P_v))[y]
    plt.scatter(P_v_x, P_v_y, marker='o', color='b', label='point')

    if len(centroids):
        
        C_x = list(zip(*centroids))[x]
        C_y = list(zip(*centroids))[y]
        plt.scatter(C_x, C_y, marker='x', color='g', label='point')

    plt.show()

def plot_followees_personality_3d(follower, x=0, y=1, z=2, centroids:list=[]):
    
    ax = plt.axes(projection='3d')
    ax.set_xlabel(label[x])
    ax.set_ylabel(label[y])
    ax.set_zlabel(label[z])

    plt.gca().set_xlim([0,1])
    plt.gca().set_ylim([0,1])
    plt.gca().set_zlim([0,1])

    p_u = G.nodes[node]["pers"]
    ax.scatter(p_u[x], p_u[y], p_u[z], marker='o', color='r', label='point')

    P_v = [G.nodes[v]["pers"] for u, v in G.out_edges(node)]
    P_v_x = list(zip(*P_v))[x]
    P_v_y = list(zip(*P_v))[y]
    P_v_z = list(zip(*P_v))[z]
    ax.scatter(P_v_x, P_v_y, P_v_z, marker='o', color='b', label='point')

    if len(centroids):

        C_x = list(zip(*centroids))[x]
        C_y = list(zip(*centroids))[y]
        C_z = list(zip(*centroids))[z]
        ax.scatter(C_x, C_y, C_z, marker='x', color='g', label='point')

    plt.show()

def plot_outgoing_degree_histogram(G):

    degree_sequence = sorted([d for n, d in G.out_degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=1, color="b")

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")

    plt.axes([0.4, 0.4, 0.5, 0.5])
    plt.axis("off")
    plt.show()

label = {0: "Openness in Experience", 1: "Conscientiousness", 2: "Extraversion", 3: "Agreeableness", 4: "Neuroticism"}

if __name__ == "__main__":

    name = "twitter-10-strong-0.2"

    G = hlpr.load_graph(name)
    centroids, weights, meta = hlpr.load_ideal_centroids(name)

    nodes = [node for node in G.nodes if len(G.out_edges(node))]

    for node in nodes:
        plot_followees_personality_2d(node, centroids=centroids[node])

    plot_outgoing_degree_histogram(G)
