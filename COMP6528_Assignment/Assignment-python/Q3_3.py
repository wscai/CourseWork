# Code from Saurabh Gupta
from tqdm import tqdm
import os, sys, numpy as np, cv2

sys.path.insert(0, 'pybsds')
from scipy import signal
from skimage.util import img_as_float
from skimage.io import imread
from pybsds.bsds_dataset import BSDSDataset
from pybsds import evaluate_boundaries
import matplotlib
from helper_function import *

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import time

sigmaa = 2
sizee = 5
GT_DIR = os.path.join('contour-data', 'groundTruth')
IMAGE_DIR = os.path.join('contour-data', 'images')
N_THRESHOLDS = 99


def get_imlist(name):
    imlist = np.loadtxt('contour-data/{}.imlist'.format(name))
    return imlist.astype(np.int32)


def compute_edges_dxdy(I):
    """Returns the norm of dx and dy as the edge response function."""
    I = I.astype(np.float32) / 255.
    sigma = sigmaa
    ksize = sizee
    # obtain Gaussian kernel with hyperparameters provided
    gaussian_kernel = cv2.getGaussianKernel(ksize=ksize, sigma=sigma)
    gaussian_kernel = np.dot(gaussian_kernel, gaussian_kernel.T)
    # obtain derivative of Gaussian kernel
    dx_g = signal.convolve2d(gaussian_kernel, np.array([[-1, 0, 1]]), mode='same', boundary='symm')
    dy_g = signal.convolve2d(gaussian_kernel, np.array([[-1, 0, 1]]).T, mode='same', boundary='symm')
    # convolve image with derivative of Gaussian kernel
    dx = signal.convolve2d(I, dx_g, mode='same', boundary='symm')
    dy = signal.convolve2d(I, dy_g, mode='same', boundary='symm')
    mag = np.sqrt(dx ** 2 + dy ** 2)
    # perform non-maximum suppression
    mag = non_maximum_suppression(mag, dx, dy)
    mag = mag / np.max(mag)
    mag = mag * 255.
    mag = np.clip(mag, 0, 255)
    mag = mag.astype(np.uint8)
    return mag

@timing
def detect_edges(imlist, fn, out_dir):
    img_count = len(imlist)
    totoal_time = 0.0
    for imname in tqdm(imlist):
        I = cv2.imread(os.path.join(IMAGE_DIR, str(imname) + '.jpg'))
        gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        start_time = time.time()
        mag = fn(gray)
        totoal_time += time.time() - start_time
        out_file_name = os.path.join(out_dir, str(imname) + '.png')
        cv2.imwrite(out_file_name, mag)
    print("average time: {}".format(totoal_time / img_count))


def load_gt_boundaries(imname):
    gt_path = os.path.join(GT_DIR, '{}.mat'.format(imname))
    return BSDSDataset.load_boundaries(gt_path)


def load_pred(output_dir, imname):
    pred_path = os.path.join(output_dir, '{}.png'.format(imname))
    return img_as_float(imread(pred_path))


def display_results(ax, f, im_results, threshold_results, overall_result):
    out_keys = ['threshold', 'f1', 'best_f1', 'area_pr']
    out_name = ['threshold', 'overall max F1 score', 'average max F1 score',
                'area_pr']
    for k, n in zip(out_keys, out_name):
        print('{:>20s}: {:<10.6f}'.format(n, getattr(overall_result, k)))
        f.write('{:>20s}: {:<10.6f}\n'.format(n, getattr(overall_result, k)))
    res = np.array(threshold_results)
    recall = res[:, 1]
    precision = res[recall > 0.01, 2]
    recall = recall[recall > 0.01]
    label_str = '{:0.2f}, {:0.2f}, {:0.2f}'.format(
        overall_result.f1, overall_result.best_f1, overall_result.area_pr)
    # Sometimes the PR plot may look funny, such as the plot curving back, i.e,
    # getting a lower recall value as you lower the threshold. This is because of
    # the lack on non-maximum suppression. The benchmarking code does some
    # contour thinning by itself. Unfortunately this contour thinning is not very
    # good. Without having done non-maximum suppression, as you lower the
    # threshold, the contours become thicker and thicker and we lose the
    # information about the precise location of the contour. Thus, a thined
    # contour that corresponded to a ground truth boundary at a higher threshold
    # can end up far away from the ground truth boundary at a lower threshold.
    # This leads to a drop in recall as we decrease the threshold.
    ax.plot(recall, precision, 'r', lw=2, label=label_str)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')


def non_maximum_suppression(gxy, gx, gy):
    ans = np.zeros(gxy.shape)
    for y in range(1, gxy.shape[0] - 1):
        for x in range(1, gxy.shape[1] - 1):
            # if no gradient then continue
            if gxy[y, x] == 0:
                continue
            # if the gradient is close to y direction
            elif np.abs(gx[y, x]) <= np.abs(gy[y, x]):
                # the main direction
                g_main_b = gxy[y - 1, x]
                g_main_f = gxy[y + 1, x]
                # degree from y direction
                w = abs(gx[y, x]) / abs(gy[y, x])
                # minor direction gradient x
                if gx[y, x] * gy[y, x] > 0:
                    g_minor_b = gxy[y - 1, x - 1]
                    g_minor_f = gxy[y + 1, x + 1]
                else:
                    g_minor_b = gxy[y - 1, x + 1]
                    g_minor_f = gxy[y + 1, x - 1]
            # if the gradient is close to x direction
            else:
                # the main direction
                g_main_b = gxy[y, x - 1]
                g_main_f = gxy[y, x + 1]
                # degree from x direction
                w = abs(gy[y, x]) / abs(gx[y, x])
                # minor direction gradient y
                if gx[y, x] * gy[y, x] > 0:
                    g_minor_b = gxy[y + 1, x - 1]
                    g_minor_f = gxy[y - 1, x + 1]
                else:
                    g_minor_b = gxy[y - 1, x - 1]
                    g_minor_f = gxy[y + 1, x + 1]
            # mix the gradients according to the degree
            g_mix_b = w * g_minor_b + (1 - w) * g_main_b
            g_mix_f = w * g_minor_f + (1 - w) * g_main_f
            # suppress the pixel if it is not the maximum
            if gxy[y, x] >= g_mix_b and gxy[y, x] >= g_mix_f:
                ans[y, x] = gxy[y, x]
    return ans


if __name__ == '__main__':
    imset = 'val'
    imlist = get_imlist(imset)
    output_dir = 'contour-output/nms'
    fn = compute_edges_dxdy
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('Running detector:')
    detect_edges(imlist, fn, output_dir)

    _load_pred = lambda x: load_pred(output_dir, x)
    print('Evaluating:')
    sample_results, threshold_results, overall_result = \
        evaluate_boundaries.pr_evaluation(N_THRESHOLDS, imlist, load_gt_boundaries,
                                          _load_pred, fast=True, progress=tqdm)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()
    file_name = os.path.join(output_dir + '_out.txt')
    with open(file_name, 'wt') as f:
        display_results(ax, f, sample_results, threshold_results, overall_result)
    fig.savefig(os.path.join(output_dir + '_pr.pdf'), bbox_inches='tight')
