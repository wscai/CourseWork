import numpy as np
import torch
torch.set_printoptions(precision=10)
# %% Accuracy, Precision, Recall
# left - right, up - down
tp = 23
fp = 8
fn = 10
tn = 153500
print(f'precision: {tp / (tp + fp)}')
print(f'recall: {tp / (tp + fn)}')
print(f'accuracy: {(tp + tn) / (tp + fp + tn + fn)}')
precision = tp / (tp + fp)
recall = tp / (tp + fn)
alpha = 0.7
print(f'f measure = {1/(alpha/precision+(1-alpha)/recall)}')
# %% Linear Regression
w = np.array([
    [3,5,-3,2],
    [5,1,-2,-1],
    [3,-2,6,-2]
]).T
x = np.array([3,-2,1,1]).T
b = np.array([-5,1,-6]).T
print(f'prediction {np.dot(w.T, x) + b}')

# %% Logistic Regression
w = np.array([
    [3,5,-3,2],
    [5,1,-2,-1],
    [3,-2,6,-2]
]).T
x = np.array([3,-2,1,6]).T
b = np.array([-5,1,-6]).T
print(f'softmax = {torch.nn.functional.softmax(torch.Tensor(np.dot(w.T, x) + b),dim=0)}')
print(f'probability of datapoint {(1 + np.exp(-(np.dot(w.T, x) + b))) ** -1}')

# %% Pointwise Mutual Information (PPMI)
# max( log ( P(wi,wj) / ( P(wi) p(wj) ) ), 0)
total = 175
pwiwj = 5
pwi = 28
pwj =32
if np.log(pwiwj / pwi / pwj * total) > 0:
    print(f'PPMI: {np.log(pwiwj / pwi / pwj * total)}')
    print(f' frac part: {pwiwj}*{total}/({pwi}*{pwj})')
else:
    print(f'PPMI: 0')

# %% PMI:
# log ( P(wi,wj) / ( P(wi) p(wj) ) )
total = 240
pwiwj = 38
pwi = 80
pwj = 77
print(f'PMI: {np.log(pwiwj / pwi / pwj * total)}')
print(f' frac part: {pwiwj}*{total}/({pwi}*{pwj})')

# %% maximum value of the normalised attention weights (seq to seq)
# row for each hi
def score(h,d):
    return np.dot(h, d)
H = np.array([
    [1.2, -0.5, 0.1],
    [0.4, 1.1, -0.3],
    [-0.1, 1.0, -1.4],
    [0.2, -1.0, 0.2]
])
# for the second vector
d2 = np.array([0.3, 1.5, 0.6]).T
print(np.dot(H, d2))
print(f'max value = {torch.nn.functional.softmax(torch.Tensor(score(H,d2)), dim=0)}')

#%% maximum value of the normalised attention weights (self-attention)
def sim(x,y):
    return np.dot(x, y)
WQ = np.array([
    [-0.8,0.6],
    [-0.2,0.7],
    [-0.1,0.5]
])
WK = np.array([
    [0.9,-0.2],
    [0.3,-0.5],
    [0.4,0.1]
])
WV = np.array([
    [1,1],
    [0.6,-0.1],
    [0.6,0.8]
])
x1 = np.array([1.2,-0.5,0.3]).T
x2 = np.array([0.3,-1.6,1.8]).T
x3 = np.array([-0.5,1.7,0.9]).T
# Calculate the maximum value of the normalised attention weights for the second input vector
q2 = np.dot(x2,WQ)
k1 = np.dot(x1,WK)
k2 = np.dot(x2,WK)
k3 = np.dot(x3,WK)
print(q2,k1,k2,k3)
ans = torch.nn.functional.softmax(torch.Tensor([sim(q2,k1),sim(q2,k2),sim(q2,k3)]),dim=0)
print(ans)

#%% stupid backoff
lamb = 0.4
count_word = 3
N = 14
print(f'S = {lamb*count_word/N}')

#%%
x = [0 for _ in range(10)]
x[0] = [0, 12, 1]
x[1] = [1, 4, 0]
x[2] = [3, 8, -1]
x[3] = [6, 12, 3]
x[4] = [0, 7, -1]
x[5] = [6, 10, 1]
x[6] = [5, 6, -1]
x[7] = [0, 3, 1]
x[8] = [7, 6, 2]
x[9] = [4, 11, -1]
