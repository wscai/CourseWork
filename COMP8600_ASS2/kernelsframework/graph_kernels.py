###### DO NOT CHANGE ######

import numpy as np
import networkx as nx
import time
import zipfile
import os
import warnings
import pickle
from typing import List, Tuple
from sklearn import svm
from sklearn.model_selection import train_test_split
from tqdm import trange
from kernel_definitions import compute_wl_features
warnings.filterwarnings("ignore")


# DATASETS = ["MUTAG", "DD", "NCI1", "NCI109", "EXP"]
DATASETS = ["MUTAG", "NCI1", "NCI109"]

def get_graphs(dataset: str) -> Tuple[List[nx.Graph], List[int]]:
  """ Return list of networkx graphs and corresponding labels for input dataset """
  
  path = f"data"
  data_file = f"{path}/{dataset}.pkl"
  if not os.path.exists(data_file):
    with zipfile.ZipFile(f"{path}/graphs.zip", 'r') as zip_ref:
      zip_ref.extractall(path)

  with open(data_file, 'rb') as handle:
    graphs, y = pickle.load(handle)

  n_graphs = len(graphs)
  n_nodes = sum([len(graph.nodes) for graph in graphs])
  n_edges = sum([len(graph.edges) for graph in graphs])
  print(f"Stats for {dataset}:")
  print(f"n_graphs: {len(graphs)}")
  print(f"avg_nodes: {n_nodes/n_graphs:.2f}")
  print(f"avg_edges: {n_edges/n_graphs:.2f}")
  return (graphs, y)

def train_test_pipeline(graphs: List[nx.Graph], y: int) -> None:
  """ No validation set since we are fixing all the parameters. """
  train_accuracies = []
  test_accuracies = []
  t = time.time()
  h = 5
  C = 1
  print(f"Generating WL features for h={h}...")
  X = compute_wl_features(graphs, h)
  print(f"WL features generated! Time taken: {time.time() - t:.2f}s. Shape: {X.shape}.")
  
  # repeat experiment 10 times to reduce variance
  pbar = trange(10)
  for i in pbar:
    pbar.set_description(f"Experiment {i+1}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = svm.LinearSVC(C=C)
    clf.fit(X_train, y_train)
    train_accuracies.append(clf.score(X_train, y_train))
    test_accuracies.append(clf.score(X_test, y_test))

  train_accuracies = np.array(train_accuracies)
  test_accuracies = np.array(test_accuracies)
  print(f"Training and testing completed!")
  print(f"train avg accuracy and std: {train_accuracies.mean():.2f}±{train_accuracies.std():.2f}")
  print(f"test avg accuracy and std: {test_accuracies.mean():.2f}±{test_accuracies.std():.2f}")
  return

def main():
  for dataset in DATASETS:
    graphs, y = get_graphs(dataset)
    print(f"Training and testing for {dataset}...")
    train_test_pipeline(graphs, y)
    print()
  return

if __name__ == "__main__":
  main()
