# Metrics/Indexes
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from functools import partial
import numpy as np
import lpips
from torchvision import transforms
import torch
from PIL import Image

class Bandwise(object):
    def __init__(self, index_fn):
        self.index_fn = index_fn

    def __call__(self, X, Y):
        C = X.shape[-1]
        bwindex = []
        for ch in range(C):
            x = X[..., ch]
            y = Y[..., ch]
            index = self.index_fn(x, y)
            bwindex.append(index)
        return bwindex


cal_bwpsnr = Bandwise(partial(compare_psnr, data_range=255))
cal_bwssim = Bandwise(partial(compare_ssim, data_range=255))


def compare_ncc(x, y):
    return np.mean((x - np.mean(x)) * (y - np.mean(y))) / (np.std(x) * np.std(y))


def ssq_error(correct, estimate):
    """Compute the sum-squared-error for an image, where the estimate is
    multiplied by a scalar which minimizes the error. Sums over all pixels
    where mask is True. If the inputs are color, each color channel can be
    rescaled independently."""
    assert correct.ndim == 2
    if np.sum(estimate ** 2) > 1e-5:
        alpha = np.sum(correct * estimate) / np.sum(estimate ** 2)
    else:
        alpha = 0.
    return np.sum((correct - alpha * estimate) ** 2)


def local_error(correct, estimate, window_size, window_shift):
    """Returns the sum of the local sum-squared-errors, where the estimate may
    be rescaled within each local region to minimize the error. The windows are
    window_size x window_size, and they are spaced by window_shift."""
    M, N, C = correct.shape
    ssq = total = 0.
    for c in range(C):
        for i in range(0, M - window_size + 1, window_shift):
            for j in range(0, N - window_size + 1, window_shift):
                correct_curr = correct[i:i + window_size, j:j + window_size, c]
                estimate_curr = estimate[i:i + window_size, j:j + window_size, c]
                ssq += ssq_error(correct_curr, estimate_curr)
                total += np.sum(correct_curr ** 2)
    # assert np.isnan(ssq/total)
    return ssq / total


def cal_lpips(image1, image2):
    lpips_model = lpips.LPIPS(net="alex")
    
    resize = transforms.Resize((256, 256))
    image1 = resize(Image.fromarray(np.uint8(image1)))
    image2 = resize(Image.fromarray(np.uint8(image2)))
    tensor1 = transforms.ToTensor()(image1).unsqueeze(0)
    tensor2 = transforms.ToTensor()(image2).unsqueeze(0)
    lpips_score = lpips_model(tensor1, tensor2)
    lpips_score = lpips_score.item()
    return lpips_score

def quality_assess(X, Y):
    # Y: correct; X: estimate
    psnr = np.mean(cal_bwpsnr(Y, X))
    ssim = np.mean(cal_bwssim(Y, X))
    return {'PSNR': psnr, 'SSIM': ssim}
    # lmse = local_error(Y, X, 20, 10)
    # ncc = compare_ncc(Y, X)
    # lpips = cal_lpips(Y, X)
    # return {'PSNR': psnr, 'SSIM': ssim, 'LMSE': lmse, 'NCC': ncc, 'lpips': lpips}
