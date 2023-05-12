import numpy as np
from sklearn.mixture import GaussianMixture 
import numpy as np
import networkx as nx
from typing import List

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
  """ TODO: replace with your own implementation """
  def injective_function(l_old, graph):
    l_new = {}
    for node in graph.nodes:
      l_new[node] = l_old[node]
      for neighbor in graph.neighbors(node):
        l_new[node] += l_old[neighbor]
    return l_new
  ret = []
  N = h+1
  for graph in graphs:
    l = {}
    for node in graph.nodes:
      l[node] = graph.degree(node)
    for k in range(h):
      l = injective_function(l, graph)


    feature = np.random.rand(N)
    ret.append(feature)
  return np.array(ret)