import numpy as np
import cv2

I = np.array([[58,38,36],
             [32,52,32],
              [80,30,35]])
k = 1
l = 1
def w_range(I,k,l,size=3, sigma = 50):
    # initialize a zero matrix with shape (size, size)
    ans = np.zeros((size,size))
    # traverse through the filtered region in I
    for i in range(k-size//2,k+size//2+1):
        for j in range(l-size//2,l+size//2+1):
            # update the zero matrix with value calculated by current traversing and central element
            ans[i-k+size//2,j-l+size//2] = np.exp(-np.linalg.norm(I[i,j]-I[k,l])**2/(2*sigma**2))
    return ans
# generate range filter
r_filter = w_range(I,1,1)
print(r_filter)
def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)
domain_filter = gkern(3,2)
def filterer_value(I,k,l,size=3):
    # obtain domain and range filter
    domain_kernel = domain_filter
    range_filter = w_range(I,1,1)
    # calculate wp by summing up the elementwise product of the two filters
    wp = np.sum(domain_kernel*range_filter)
    # calculate the filtered matrix
    ans = I[k-size//2:k+size//2+1,l-size//2:l+size//2+1]*range_filter*domain_kernel/wp
    # summing up to obtain filtered value
    return np.sum(ans)
# generate filtered value
f_value = filterer_value(I,1,1)
