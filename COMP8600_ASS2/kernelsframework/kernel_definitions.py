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
  G = np.zeros(2*K*D)
  return G


""" WL kernel for graph classification subquestion """
def compute_wl_features(graphs: List[nx.Graph], h: int) -> np.array:
  """ TODO: replace with your own implementation """
  ret = []
  N = h+1
  for graph in graphs:
    feature = np.random.rand(N)
    ret.append(feature)
  return np.array(ret)