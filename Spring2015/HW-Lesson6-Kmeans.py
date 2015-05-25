# -*- coding: utf-8 -*-
"""
Created on Sat May 23 17:20:10 2015

@author: kartiks
"""

import numpy as np
import random
import matplotlib.pyplot as plt

 
def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters
 
def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu
 
def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))
    
    
def find_centers(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
        print "mu[0] = %s" % mu[0]
        plot_clusters (clusters,K, mu)
    return(mu, clusters)


def plot_clusters (clusters,K, mu):
    if (K == 3):
#       plt.plot (clusters[0], 'ro', clusters[1] , 'go', clusters[2], 'bo' , markersize=5)
#       plt.plot (mu[0],'r*' , mu[1], 'g*', mu[2] , 'b*', markersize=10) 
        t0 = np.array(clusters[0])
        x0 = t0[:,0]
        y0 = t0[:,1]
        plt.plot (x0, y0,'ro' , markersize=5)
 
        t1 = np.array(clusters[1])
        x1 = t1[:,0]
        y1 = t1[:,1]
        plt.plot (x1, y1,'go' , markersize=5)
  
        t2 = np.array(clusters[2])
        x2 = t2[:,0]
        y2 = t2[:,1]
        plt.plot (x2, y2,'bo' , markersize=5)


        mu = np.array(mu)
        plt.plot (mu[0][0], mu[0][1],'r*', markersize=10) 
        plt.plot (mu[1][0], mu[1][1],'g*', markersize=10) 
        plt.plot (mu[2][0], mu[2][1],'b*', markersize=10) 
        plt.show()
    
 
def init_board(N):
    X = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(N)])
    return X
    

def init_board_gauss(N, k):
    n = float(N)/k
    X = []
    for i in range(k):
        c = (random.uniform(-1, 1), random.uniform(-1, 1))
        s = random.uniform(0.05,0.5)
        x = []
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a,b])
        X.extend(x)
    X = np.array(X)[:N]
    return X
    

X = init_board_gauss(200,3)
find_centers(X,3)