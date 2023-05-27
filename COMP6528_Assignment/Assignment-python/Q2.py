import numpy as np
import cv2

I = np.array([[128, 128, 100, 100, 103,  50],
                [150, 100, 120,  30,  53,  54],
                [150, 112, 127,  40,  35,  20],
                [132, 125, 112,  43,  20,  10],
                [133, 130, 100,  30,  10,  20],
                [140, 130, 120,  20,  10,  20]])
k = 3
l = 2
def w_range(I,k,l,size=5, sigma = 50):
    # initialize a zero matrix with shape (size, size)
    ans = np.zeros((size,size))
    # traverse through the filtered region in I
    for i in range(k-size//2,k+size//2+1):
        for j in range(l-size//2,l+size//2+1):
            # update the zero matrix with value calculated by current traversing and central element
            ans[i-k+size//2,j-l+size//2] = np.exp(-np.linalg.norm(I[i,j]-I[k,l])**2/2/sigma**2)
    return ans
# generate range filter
r_filter = w_range(I,3,2)

domain_filter = np.array([[0.0232, 0.0338, 0.0383, 0.0338, 0.0232],
                [0.0338, 0.0492, 0.0558, 0.0492, 0.0338],
                [0.0383, 0.0558, 0.0632, 0.0558,  0.0383],
                [0.0338, 0.0492, 0.0558, 0.0492, 0.0338],
                [0.0232, 0.0338, 0.0383, 0.0338, 0.0232]])
def filterer_value(I,k,l,size=5):
    # obtain domain and range filter
    domain_kernel = domain_filter
    range_filter = w_range(I,3,2)
    # calculate wp by summing up the elementwise product of the two filters
    wp = np.sum(domain_kernel*range_filter)
    # calculate the filtered matrix
    ans = I[k-size//2:k+size//2+1,l-size//2:l+size//2+1]*range_filter*domain_kernel/wp
    # summing up to obtain filtered value
    return np.sum(ans)
# generate filtered value
f_value = filterer_value(I,3,2)