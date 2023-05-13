import numpy as np
from sklearn.mixture import GaussianMixture 
import numpy as np
import networkx as nx
from typing import List
import copy
from collections import Counter

""" Fisher kernel for image classification subquestion """
def compute_fisher_vector(descriptors: np.array, gmm: GaussianMixture) -> np.array:
  """ computes 2*K*D Fisher vector """
  # TODO replace with your own implementation
  T, D = descriptors.shape
  K = gmm.n_components
  gamma = gmm.predict_proba(descriptors)
  pi = gmm.weights_
  mu = gmm.means_
  sigma = gmm.covariances_
  Gmu = np.zeros((K,D))
  Gsigma = np.zeros((K,D))
  for i in range(K):
    Gmu[i] = np.sum(gamma[:,i].reshape(-1,1)*(descriptors - mu[i]), axis=0) / (T*np.sqrt(pi[i]))
    Gsigma[i] = np.sum(gamma[:,i].reshape(-1,1)*((descriptors - mu[i])**2 / sigma[i]**2 -1), axis=0) / (np.sqrt(2*T*pi[i]))
  G = np.concatenate((Gmu.flatten(), Gsigma.flatten()))
  return G


""" WL kernel for graph classification subquestion """
def compute_wl_features(graphs: List[nx.Graph], h: int) -> np.array:
  # """ TODO: replace with your own implementation """
  label_dict = {}
  L = []
  final_labels = set()
  def injection_f(key):
    if key not in label_dict:
      label_dict[key] = len(label_dict)
    return label_dict[key]
  for graph in graphs:
    # Initial features
    l={}
    for node in graph.nodes:
      l[node] = injection_f(str(graph.degree(node)))
    # update features
    for k in range(h):
      ll = copy.deepcopy(l)
      for node in graph.nodes:
        key = f'{l[node]} {sorted([l[m] for m in graph.neighbors(node)])}'
        ll[node] = injection_f(key)
      l=ll
    L.append(l)
    final_labels.update(l.values())
  final_labels = list(final_labels)
  counts = np.zeros((len(graphs), len(final_labels)))
  for i in range(len(L)):
    for j in L[i].values():
      counts[i][final_labels.index(j)]+=1
  return counts