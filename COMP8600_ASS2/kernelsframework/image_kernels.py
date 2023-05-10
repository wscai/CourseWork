###### DO NOT CHANGE ######

import os
from typing import Dict
import cv2
import datetime 
import numpy as np
import pickle as pkl
import time
import glob
import warnings
import argparse
from zipfile import ZipFile
from sklearn import svm
from sklearn.decomposition import PCA
from tqdm import tqdm, trange
from sklearn.mixture import GaussianMixture
from kernel_definitions import compute_fisher_vector
warnings.filterwarnings("ignore")


def power_normalisation(x: np.array, a: float) -> np.array:
  """ power normalisation (Eq. 15) """
  x = np.sign(x) * np.abs(x)**a
  # x = np.sign(x) * np.log(1 + a*np.abs(x))
  return x

def get_data(**kwargs):
  F = kwargs['f']
  small = kwargs['small']
  size_desc = "_small" if small else ""
  data = {}
  sift_descriptors_file = f'sift_descriptors{size_desc}_F{F}.pkl'
  if not os.path.exists(sift_descriptors_file):
    X = []
    cnt = 0
    sift = cv2.SIFT_create()
    data_path = "data/mnist_png"
    if not os.path.exists(data_path):
      print("Extracting data from zip file...")
      with ZipFile("data/mnist_png.zip", 'r') as zObject:
        zObject.extractall("data/")
      print("Data extracted!")
    print("Constructing sift descriptors for images...")
    for c in trange(10):
      for folder, class_samples in [("training",600), ("testing",100)]:
        files = glob.glob(f"{data_path}/{folder}/{c}/*.png")
        if small:
          files = files[:class_samples]
        for file in files:
          img = cv2.imread(file)
          _, descriptors = sift.detectAndCompute(img, None)
          if descriptors is None:  # skip images with empty descriptors
            continue
          i = int(os.path.basename(file).replace(".png", ""))
          data[i] = (np.arange(cnt, cnt + len(descriptors)), c, folder)
          cnt += len(descriptors)
          X.append(descriptors)
    X = np.vstack(X)
    print(X)
    print("Saving sift descriptors for images...")
    with open(sift_descriptors_file, 'wb') as f:
      pkl.dump((X, data), f)
    print("Sift descriptors for image saved!")
  else:
    print("Loading saved sift descriptors for images...")
    with open(sift_descriptors_file, 'rb') as f:
      X, data = pkl.load(f)
    print("Sift descriptors for image loaded!")
    
  print("Reducing size of sift descriptors with PCA...")
  pca = PCA(n_components=F, random_state=0)
  pca.fit(X)
  X = pca.transform(X)

  print(f"X.shape: {X.shape}")
  return X, data

def get_gmm(X: np.array, **kwargs):
  print("Constructing and fitting GMM...")
  K = kwargs["k"]
  gmm = GaussianMixture(n_components=K, covariance_type="diag", random_state=0, verbose=1)
  gmm.fit(X)
  print("GMM fitted!")
  return gmm

def train_test_pipeline(X: np.array, data: Dict, gmm: GaussianMixture, a: float):
  print("Creating Fisher vectors...")
  X_train = []
  X_test = []
  y_train = []
  y_test = []
  for i in tqdm(data):
    indices, y, desc = data[i]
    x = compute_fisher_vector(descriptors=X[indices], gmm=gmm)
    x = power_normalisation(x, a=a)
    if desc=="training":
      X_train.append(x)
      y_train.append(y)
    else:  # desc=="testing"
      X_test.append(x)
      y_test.append(y)
  clf = svm.LinearSVC(C=1, penalty="l2", random_state=0, dual=False)
  print("Fitting SVM (this might take a while)...")
  print(f"Time of SVM training start time: {datetime.datetime.now()}")
  t = time.time()
  clf.fit(X_train, y_train)
  print(f"SVM fitted! Time taken: {time.time()-t}s.")
  print("Making predictions...")
  train_acc = clf.score(X_train, y_train)
  test_acc = clf.score(X_test, y_test)
  print(f"train_acc: {train_acc:.2f}")
  print(f"test_acc: {test_acc:.2f}")
  return

def main():
  parser = argparse.ArgumentParser()

  parser.add_argument('-k', type=int, help='number of Gaussian components in GMM', default=64)
  parser.add_argument('-f', type=int, help='number of features after PCA', default=32)
  parser.add_argument('-a', type=float, help='power normalisation parameter', default=1.)
  parser.add_argument('--small', action='store_true')

  args = parser.parse_args()

  print("Configuration:")
  for k, v in vars(args).items():
    print('{0:12}  {1}'.format(k, v))

  kwargs = {"k": args.k, "f": args.f, 'small': args.small}

  X, data = get_data(**kwargs)
  gmm = get_gmm(X=X, **kwargs)
  train_test_pipeline(X=X, data=data, gmm=gmm, a=args.a)
  return 

if __name__ == "__main__":
  main()
