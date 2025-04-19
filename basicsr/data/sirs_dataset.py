import sys
import math
import os.path
import random
from os.path import join
import time
import cv2
import numpy as np
import torch.utils.data
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.signal import convolve2d
import torchvision.utils as utils
from basicsr.data.image_folder import make_dataset
from basicsr.data.torchdata import Dataset as BaseDataset
from basicsr.data.transforms import to_tensor
from basicsr.data.image_folder import read_fns, read_fns_clean
# import basicsr.models.archs.clip as clip
# import CLIP.clip as oriclip
import json
from glob import glob
from skimage import measure
import torchvision.transforms as transforms
# from dift_src.models.dift_sd import SDFeaturizer
import torch.nn.functional as F
import gc
import torch.nn as nn
import datetime
from torchvision.transforms import GaussianBlur
from basicsr.utils.amg import build_all_layer_point_grids
from basicsr.data.mask_generator import SamMaskGenerator
import matplotlib.pyplot as plt
from basicsr.utils.utils import show_mask, show_points, show_lbk_masks, show_largeRef_masks, show_lbk_graymasks
from basicsr.data.build_sam import sam_model_registry
import torchvision
from PIL import Image
import torch
import zipfile
from torchvision.transforms import ToTensor
import skimage.transform as transform
from torchvision.ops.boxes import batched_nms, box_area
from skimage.measure import label, regionprops        

# from segment_anything.utils.amg import (
#     batched_mask_to_box,
#     calculate_stability_score,
#     mask_to_rle_pytorch,
#     remove_small_regions,
#     rle_to_mask,)

GRID_SIZE = 16

def process_small_region(rles):
        new_masks = []
        scores = []
        min_area = 100
        nms_thresh = 0.8
        for rle in rles:
            mask = rle_to_mask(rle[0])

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                rles[i_mask] = mask_to_rle_pytorch(mask_torch)
        masks = [rle_to_mask(rles[i][0]) for i in keep_by_nms]
        return masks

def get_predictions_given_embeddings_and_queries(img, points, point_labels, model):
    predicted_masks, predicted_iou = model(
        img[None, ...], points, point_labels
    )
    sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
    predicted_iou_scores = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
    predicted_masks = torch.take_along_dim(
        predicted_masks, sorted_ids[..., None, None], dim=2
    )
    predicted_masks = predicted_masks[0]
    iou = predicted_iou_scores[0, :, 0]
    index_iou = iou > 0.7
    iou_ = iou[index_iou]
    masks = predicted_masks[index_iou]
    score = calculate_stability_score(masks, 0.0, 1.0)
    score = score[:, 0]
    index = score > 0.8
    score_ = score[index]
    masks = masks[index]
    iou_ = iou_[index]
    masks = torch.ge(masks, 0.0)
    return masks, iou_


def run_everything_ours(img, model):
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    image = np.array(img)
    red_min_val, red_max_val = np.min(image[:, :, 0]), np.max(image[:, :, 0])
    green_min_val, green_max_val = np.min(image[:, :, 1]), np.max(image[:, :, 1])
    blue_min_val, blue_max_val = np.min(image[:, :, 2]), np.max(image[:, :, 2])

    image[:, :, 0] = (image[:, :, 0] - red_min_val) / (red_max_val - red_min_val) * 255.0
    image[:, :, 1] = (image[:, :, 1] - green_min_val) / (green_max_val - green_min_val) * 255.0
    image[:, :, 2] = (image[:, :, 2] - blue_min_val) / (blue_max_val - blue_min_val) * 255.0
    
    # h, w, _ = image.shape
    # if h > 512 or w > 512:
    #     image = cv2.resize(image, (512, 512))
    
    img_tensor = ToTensor()(image.astype(np.uint8))
    _, original_image_h, original_image_w = img_tensor.shape
    
    xy = []
    for i in range(GRID_SIZE):
        curr_x = 0.5 + i / GRID_SIZE * original_image_w
        for j in range(GRID_SIZE):
            curr_y = 0.5 + j / GRID_SIZE * original_image_h
            xy.append([curr_x, curr_y])
            
    xy = torch.from_numpy(np.array(xy))
    points = xy
    num_pts = xy.shape[0]
    point_labels = torch.ones(num_pts, 1)
    
    with torch.no_grad():
      predicted_masks, predicted_iou = get_predictions_given_embeddings_and_queries(
              img_tensor.to(device),
              points.reshape(1, num_pts, 1, 2).to(device),
              point_labels.reshape(1, num_pts, 1).to(device),
              model.to(device),
          )
    rle = [mask_to_rle_pytorch(m[0:1]) for m in predicted_masks]
    predicted_masks = process_small_region(rle)
    # predicted_masks = [transform.resize(mask, (h, w), order=0, preserve_range=True) for mask in predicted_masks]

    return predicted_masks

def __scale_width(img, target_width, W=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_width:
        return img
    w = target_width
    h = int(target_width * oh / ow)
    h = math.ceil(h / 2.0) * 2  # round up to even
    return img.resize((w, h), W)


def __scale_height(img, target_height, W=Image.BICUBIC):
    ow, oh = img.size
    if oh == target_height:
        return img
    h = target_height
    w = int(target_height * ow / oh)
    w = math.ceil(w / 2.0) * 2
    return img.resize((w, h), W)

def paired_data_transforms(img_1, img_2, img_3, img_4, unaligned_transforms=False):
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    target_size = int(random.randint(224, 448) / 2.0) * 2
    ow, oh = img_1.size
    if ow >= oh:
        img_1 = __scale_height(img_1, target_size)
        img_2 = __scale_height(img_2, target_size)
        img_3 = __scale_height(img_3, target_size)
        img_4 = __scale_height(img_4, target_size, Image.NEAREST)
    else:
        img_1 = __scale_width(img_1, target_size)
        img_2 = __scale_width(img_2, target_size)
        img_3 = __scale_width(img_3, target_size)
        img_4 = __scale_width(img_4, target_size, Image.NEAREST)

    if random.random() < 0.5:
        img_1 = TF.hflip(img_1)
        img_2 = TF.hflip(img_2)
        img_3 = TF.hflip(img_3)
        img_4 = TF.hflip(img_4)
       

    if random.random() < 0.5:
        angle = random.choice([90, 180, 270])
        img_1 = TF.rotate(img_1, angle)
        img_2 = TF.rotate(img_2, angle)
        img_3 = TF.rotate(img_3, angle)
        img_4 = TF.rotate(img_4, angle)
    

    i, j, h, w = get_params(img_1, (224, 224))
    img_1 = TF.crop(img_1, i, j, h, w)

    if unaligned_transforms:
        i_shift = random.randint(-10, 10)
        j_shift = random.randint(-10, 10)
        i += i_shift
        j += j_shift

    img_2 = TF.crop(img_2, i, j, h, w)
    img_3 = TF.crop(img_3, i, j, h, w)
    img_4 = TF.crop(img_4, i, j, h, w)

    return img_1, img_2, img_3, img_4

def paired_data_transforms0(img_1, img_2, unaligned_transforms=False):
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    target_size = int(random.randint(224, 448) / 2.) * 2
    ow, oh = img_1.size
    if ow >= oh:
        img_1 = __scale_height(img_1, target_size)
        img_2 = __scale_height(img_2, target_size)
    else:
        img_1 = __scale_width(img_1, target_size)
        img_2 = __scale_width(img_2, target_size)

    if random.random() < 0.5:
        img_1 = TF.hflip(img_1)
        img_2 = TF.hflip(img_2)

    if random.random() < 0.5:
        angle = random.choice([90, 180, 270])
        img_1 = TF.rotate(img_1, angle)
        img_2 = TF.rotate(img_2, angle)

    i, j, h, w = get_params(img_1, (224, 224))
    img_1 = TF.crop(img_1, i, j, h, w)

    if unaligned_transforms:
        # print('random shift')
        i_shift = random.randint(-10, 10)
        j_shift = random.randint(-10, 10)
        i += i_shift
        j += j_shift

    img_2 = TF.crop(img_2, i, j, h, w)

    return img_1, img_2

def Uncrop_paired_data_transforms(img_1, img_2, img_3, unaligned_transforms=False):
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    target_size = int(random.randint(224, 448) / 2.0) * 2
    ow, oh = img_1.size
    if ow >= oh:
        img_1 = __scale_height(img_1, target_size)
        img_2 = __scale_height(img_2, target_size)
        img_3 = __scale_height(img_3, target_size)
        # img_3 = __scale_height(img_3, target_size, Image.NEAREST)
        # img_4 = __scale_height(img_4, target_size)
        # img_5 = __scale_height(img_5, target_size)
    else:
        img_1 = __scale_width(img_1, target_size)
        img_2 = __scale_width(img_2, target_size)
        img_3 = __scale_width(img_3, target_size)        
        # img_3 = __scale_width(img_3, target_size, Image.NEAREST)
        # img_4 = __scale_width(img_4, target_size)
        # img_5 = __scale_width(img_5, target_size)

    if random.random() < 0.5:
        img_1 = TF.hflip(img_1)
        img_2 = TF.hflip(img_2)
        img_3 = TF.hflip(img_3)
        # img_4 = TF.hflip(img_4)
        # img_5 = TF.hflip(img_5)

    if random.random() < 0.5:
        angle = random.choice([90, 180, 270])
        img_1 = TF.rotate(img_1, angle)
        img_2 = TF.rotate(img_2, angle)
        img_3 = TF.rotate(img_3, angle)
        # img_4 = TF.rotate(img_4, angle)
        # img_5 = TF.rotate(img_5, angle)

    return img_1, img_2, img_3

class ReflectionSynthesis(object):
    def __init__(self):
        # Kernel Size of the Gaussian Blurry
        # self.kernel_sizes = [3, 5, 7, 9, 11]
        # self.kernel_probs = [0.2, 0.2, 0.2, 0.2, 0.2]
        # 0.25, 0.25, 0.25, 0.25
        self.kernel_sizes = [5, 7, 9, 11]
        self.kernel_probs = [0.1, 0.2, 0.3, 0.4]
        # Sigma of the Gaussian Blurry
        self.sigma_range = [2, 5]
        self.alpha_range = [0.6, 1.0]
        self.beta_range = [0.2, 1.0]

    def __call__(self, T_, R_):
        T_ = np.asarray(T_, np.float32) / 255.0
        R_ = np.asarray(R_, np.float32) / 255.0

        kernel_size = np.random.choice(self.kernel_sizes, p=self.kernel_probs)
        sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel2d = np.dot(kernel, kernel.T)
        for i in range(3):
            R_[..., i] = convolve2d(R_[..., i], kernel2d, mode='same')
        
        a = np.random.uniform(self.alpha_range[0], self.alpha_range[1])
        b = np.random.uniform(self.beta_range[0], self.beta_range[1])
        
        if random.random() < 0.5:
            T, R = a * T_, b * R_
        
            if random.random() < 0.7:
                I = T + R - T * R
                I = np.clip(I, 0, 1)
            else:
                I = T + R
                if np.max(I) > 1:
                    m = I[I > 1]
                    m = (np.mean(m) - 1) * 1.3
                    I = np.clip(T + np.clip(R - m, 0, 1), 0, 1)
        else:
            # gamma inverse correction
            T_inv = np.power(T_, 2.2)
            R_inv = np.power(R_, 2.2)
            
            T, R = a * T_inv, b * R_inv
        
            if random.random() < 0.7:
                I = T + R - T * R
                I = np.clip(I, 0, 1)
            else:
                I = T + R
                if np.max(I) > 1:
                    m = I[I > 1]
                    m = (np.mean(m) - 1) * 1.3
                    I = np.clip(T + np.clip(R - m, 0, 1), 0, 1)

            # gamma correction
            I = np.power(I, 1 / 2.2)
            T = np.power(T, 1 / 2.2)
            R = np.power(R, 1 / 2.2)
     
        return T, R, I


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, *args, **kwargs)
        self.shuffle = shuffle

    def reset(self):
        if self.shuffle:
            print('Reset Dataset...')
            self.dataset.reset()

def sobel_gradient_map(image, percentile_threshold=70):

    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=7)  # X 方向梯度
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=7)  # Y 方向梯度

    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    flat_gradient = gradient_magnitude.ravel()
    gradient_70th_percentile = np.percentile(flat_gradient, percentile_threshold)
    
    _, gradient_magnitude = cv2.threshold(gradient_magnitude, gradient_70th_percentile, 255, cv2.THRESH_BINARY)
    return gradient_magnitude

def prepare_image(image, img_resolution):
    trans = torchvision.transforms.Compose([torchvision.transforms.Resize((img_resolution, img_resolution))])
    image = torch.as_tensor(image)
    return trans(image.permute(2, 0, 1))

def traverse_neighbors(mask, instance_value, neighbor_distance=1):
    """
    Traverse the neighboring masks of a selected instance mask.

    Args:
        mask (numpy array): Instance segmentation mask.
        instance_value (int): Value of the selected instance mask.
        neighbor_distance (int, optional): Distance to consider as neighbors. Defaults to 1.

    Returns:
        list: List of neighboring instance values.
    """
    # Get the shape of the mask
    h, w, _ = mask.shape

    # Get the coordinates of the selected instance mask
    pixel_match = np.all(mask == instance_value, axis=-1).astype(np.uint8)

    # coords = np.where(pixel_match)    
    contours, _ = cv2.findContours(pixel_match, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outer_contour = contours[0]
    coords = outer_contour[:, 0, :][:, [1, 0]].T
    x, y = coords[0], coords[1]

    # Initialize a set to store unique neighboring instance values
    neighbors = set()

    # Iterate over the coordinates of the selected instance mask
    for i, j in zip(x, y):
        # Iterate over the neighboring pixels
        for dx in range(-neighbor_distance, neighbor_distance + 1):
            for dy in range(-neighbor_distance, neighbor_distance + 1):
                # Skip the current pixel
                if dx == 0 and dy == 0:
                    continue

                # Calculate the neighboring pixel coordinates
                ni, nj = i + dx, j + dy

                # Check if the neighboring pixel is within the mask boundaries
                if 0 <= ni < h and 0 <= nj < w:
                    # Get the instance value of the neighboring pixel
                    neighbor_value = mask[ni, nj]

                    # Add the neighboring instance value to the set
                    # print(neighbor_value, instance_value)
                    if not np.array_equal(neighbor_value, instance_value):
                    # if neighbor_value != instance_value:
                        neighbors.add(tuple(neighbor_value))

    # Return the list of unique neighboring instance values
    return list(neighbors)

class DSRDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None, enable_transforms=True):
        super(DSRDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.enable_transforms = enable_transforms
        sortkey = lambda key: os.path.split(key)[-1]
        self.paths = sorted(make_dataset(datadir, fns), key=sortkey)
        if size is not None:
            self.paths = np.random.choice(self.paths, size)

        self.syn_model = ReflectionSynthesis()
        self.reset(shuffle=False)

    def reset(self, shuffle=True):
        if shuffle:
            random.shuffle(self.paths)
        num_paths = len(self.paths) // 2
        self.B_paths = self.paths[0:num_paths]
        self.R_paths = self.paths[num_paths:2 * num_paths]
        # self.sim_map = sorted(make_dataset(datadir.replace('blended', 'similarity_map'), fns), key=sortkey)
        # self.T_paths = [path.replace('similarity_map', 'transmission_layer') for path in self.sim_map]
        # self.R_paths = [path.replace('similarity_map', 'reflection_layer') for path in self.sim_map]

        # self.rsam_mask = [path.replace('similarity_map', 'rsam_mask').replace('.jpg', '.npy') for path in self.sim_map]
        # self.B_paths = [path.replace('similarity_map', 'blended') for path in self.sim_map]
        # if size is not None:
        #     self.paths = np.random.choice(self.B_paths, size)

    def align(self, x1, x2, x3):
        h, w = x1.height, x1.width
        h, w = h // 32 * 32, w // 32 * 32
        x1 = x1.resize((w, h))
        x2 = x2.resize((w, h))
        x3 = x3.resize((w, h), Image.NEAREST)
        return x1, x2, x3

    # def data_synthesis(self, t_img, r_img, mask):
    #     if self.enable_transforms:
    #         t_img, r_img, mask1, _ = paired_data_transforms(t_img, r_img, mask, mask)

    #     t_img, r_img, m_img, thresmap = self.syn_model(t_img, r_img)
        
    #     return np.array(t_img), np.array(r_img), np.array(m_img)

    # def __getitem__(self, index):
    #     index = index % len(self.B_paths)

    #     B_path = self.B_paths[index]
    #     T_path = self.T_paths[index]
    #     R_path = self.R_paths[index]
    #     R_mask_path = self.rsam_mask[index]
    #     simmap_path = self.sim_map[index]
    #     fn = self.B_paths[index]
    #     B = Image.open(B_path).convert('RGB')
    #     T = Image.open(T_path).convert('RGB')
    #     R = Image.open(R_path).convert('RGB')
    #     simmap = Image.open(simmap_path).convert('L')
    #     r_sammask = Image.fromarray(np.array(np.load(R_mask_path))).convert('L')

    #     # the contrastive mask size is [224,224,1] with 255 denotes reflection area and 125 denotes transmission area
    #     simmap_array = np.mean(np.array(R), axis=2)
    #     r_sammask_array = np.array(r_sammask)
    #     labeled_mask = label(r_sammask_array)
    #     regions = regionprops(labeled_mask)
    #     lowest_similarity = float('inf')
    #     highest_similarity = 0
    #     ref_mask = None
    #     trans_mask = None

    #     for region in regions:
    #         mask = (labeled_mask == region.label).astype(np.uint8)
    #         similarity = np.sum(simmap_array * mask) / np.sum(mask)
    #         if np.sum(mask) > 100 and similarity > highest_similarity:
    #             highest_similarity = similarity
    #             ref_mask = mask * 255
            
    #         if np.sum(mask) > 100 and similarity < lowest_similarity:
    #             lowest_similarity = similarity
    #             trans_mask = mask * 255
        
    #     B = TF.to_tensor(B)
    #     R = TF.to_tensor(R)
    #     T = TF.to_tensor(T)
    #     trans_mask[trans_mask == 255] = 125
    #     ref_mask = ref_mask + trans_mask
    #     ref_mask[ref_mask > 255] = 255 
    #     ref_mask = TF.to_tensor(ref_mask) 

    #     # from torchvision.utils import save_image
    #     # current_timestamp = time.time()
    #     # save_image(B, str(current_timestamp) + 'B.png')
    #     # save_image(T, str(current_timestamp) + 'T.png')
    #     # save_image(ref_mask, str(current_timestamp) + 'mask.png')
    
    def data_synthesis(self, t_img, r_img):
        if self.enable_transforms:
            t_img, r_img = paired_data_transforms0(t_img, r_img)

        t_img, r_img, m_img = self.syn_model(t_img, r_img)

        B = TF.to_tensor(t_img)
        R = TF.to_tensor(r_img)
        M = TF.to_tensor(m_img)

        return B, R, M

    def __getitem__(self, index):
        index_B = index % len(self.B_paths)
        index_R = index % len(self.R_paths)

        B_path = self.B_paths[index_B]
        R_path = self.R_paths[index_R]

        t_img = Image.open(B_path).convert('RGB')
        r_img = Image.open(R_path).convert('RGB')

        B, R, M = self.data_synthesis(t_img, r_img)
        residual = M - B
        residual_np = np.mean(residual.numpy(), axis=0)
        max_residual_sum = -float('inf')
        min_residual_sum = float('inf')
        ref_mask= None
        trans_mask= None
        
        if np.random.random() < 0.5: 
            labeled_ref_mask = label(residual_np > 0.1)
            ref_regions = regionprops(labeled_ref_mask)
           
            for region in ref_regions:
                mask = (labeled_ref_mask == region.label).astype(np.uint8)

                if np.sum(mask) > 400:
                    region_residual_sum = np.mean(residual_np[mask])
                    if region_residual_sum > max_residual_sum:
                        max_residual_sum = region_residual_sum
                        ref_mask= mask * 255
        
        if np.random.random() < 0.5: 
            labeled_trans_mask = label(residual_np < 0.03)
            trans_regions = regionprops(labeled_trans_mask)
        
            for region in trans_regions:
                mask = (labeled_trans_mask == region.label).astype(np.uint8)

                if np.sum(mask) > 400:
                    region_residual_sum = np.mean(residual_np[mask])
                    if region_residual_sum < min_residual_sum:
                        min_residual_sum = region_residual_sum
                        trans_mask= mask * 255
        
        if trans_mask is None:
            trans_mask = np.zeros_like(residual_np, dtype=np.uint8)
        if ref_mask is None:
            ref_mask = np.zeros_like(residual_np, dtype=np.uint8)
            
        trans_mask[trans_mask == 255] = 125
        mask = trans_mask + ref_mask
        mask[mask > 255] = 255
        mask = torch.from_numpy(mask[np.newaxis, :, :] / 255).float()
        fn = os.path.basename(B_path)

        # from torchvision.utils import save_image
        # current_timestamp = time.time()
        # save_image(M, str(current_timestamp) + 'M.png')
        # save_image(B, str(current_timestamp) + 'B.png')
        # save_image(mask, str(current_timestamp) + 'mask.png')

        return {'input': M,  'target_t': B, 'target_r': R, 'mask': mask, 'fn': fn, 'real': False}

    def __len__(self):
        if self.size is not None:
            return min(max(len(self.B_paths), len(self.R_paths)), self.size)
        else:
            return max(len(self.B_paths), len(self.R_paths))

class DSRTestDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None, enable_transforms=False, unaligned_transforms=False, round_factor=1, flag=None, if_align=False, test_flag=False):
        super(DSRTestDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns or [fn for fn in os.listdir(join(datadir, 'allPairMask')) if fn.endswith('.png') or fn.endswith('.jpg')]
        self.enable_transforms = enable_transforms
        self.unaligned_transforms = unaligned_transforms
        self.round_factor = round_factor
        self.flag = flag
        self.if_align = if_align
        self.dataset_name = datadir.split('/')[-1]
        self.test_flag = test_flag

        if size is not None:
            self.fns = self.fns[:size]

    def align(self, x1, x2, x3, x4):
        h, w = x1.height, x1.width
        h, w = h // 32 * 32, w // 32 * 32
        x1 = x1.resize((w, h))
        x2 = x2.resize((w, h))
        x3 = x3.resize((w, h))
        x4 = x4.resize((w, h), Image.NEAREST)

        return x1, x2, x3, x4

    def __getitem__(self, index):
        fn = self.fns[index]

        t_img = Image.open(join(self.datadir, 'transmission_layer', fn)).convert('RGB')
        m_img = Image.open(join(self.datadir, 'blended', fn)).convert('RGB')
        r_img = Image.open(join(self.datadir, 'reflection_layer', fn)).convert('RGB')
        # r_sammask = Image.fromarray(np.load(join(self.datadir, 'rsam_mask', fn.replace('.jpg', '.npy').replace('.png', '.npy'))))
        
        save_mask = Image.open(join(self.datadir, 'allPairMask', fn)).convert('L')
        save_mask = np.array(save_mask)
        save_mask[save_mask < 100] = 0
        save_mask[save_mask > 200] = 255
        save_mask[(save_mask != 0) & (save_mask != 255)] = 125
        save_mask = Image.fromarray(save_mask)

        if self.if_align:
            t_img, m_img, r_img, save_mask = self.align(t_img, m_img, r_img, save_mask)

        if self.enable_transforms:
            t_img, m_img, r_img, save_mask = paired_data_transforms(t_img, m_img, r_img, save_mask, self.unaligned_transforms)
        
   
        save_mask = np.array(save_mask)
        
        B = TF.to_tensor(t_img)
        M = TF.to_tensor(m_img)
        R = TF.to_tensor(r_img)
        save_mask = TF.to_tensor(save_mask)

        dic = {'input': M, 'target_t': B, 'fn': fn, 'real': True, 'target_r': R, 'mask': save_mask}        
        
        if self.flag is not None:
            dic.update(self.flag)
        return dic

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)

class DSRTestDataset_wosim(BaseDataset):
    def __init__(self, datadir, fns=None, size=None, enable_transforms=False, unaligned_transforms=False, round_factor=1, flag=None, if_align=False):
        super(DSRTestDataset_wosim, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns or [fn for fn in os.listdir(join(datadir, 'blended')) if fn.endswith('.png') or fn.endswith('.jpg')]
        self.enable_transforms = enable_transforms
        self.unaligned_transforms = unaligned_transforms
        self.round_factor = round_factor
        self.flag = flag
        self.if_align = if_align
        self.dataset_name = datadir.split('/')[-1]

        if size is not None:
            self.fns = self.fns[:size]

    def align(self, x1, x2, x3, x4):
        h, w = x1.height, x1.width
        h, w = h // 32 * 32, w // 32 * 32
        x1 = x1.resize((w, h))
        x2 = x2.resize((w, h))
        x3 = x3.resize((w, h), Image.NEAREST)
        x4 = x4.resize((w, h), Image.NEAREST)

        return x1, x2, x3, x4

    def __getitem__(self, index):
        fn = self.fns[index]

        t_img = Image.open(join(self.datadir, 'transmission_layer', fn)).convert('RGB')
        m_img = Image.open(join(self.datadir, 'blended', fn)).convert('RGB')
        r_img = Image.open(join(self.datadir, 'reflection_layer', fn)).convert('L')
        r_sammask = Image.fromarray(np.load(join(self.datadir, 'rsam_mask', fn.replace('.jpg', '.npy').replace('.png', '.npy'))))
        
        if self.if_align:
            t_img, m_img, r_img, r_sammask = self.align(t_img, m_img, r_img, r_sammask)

        if self.enable_transforms:
            t_img, m_img, r_img, r_sammask, r_sammask = paired_data_transforms(t_img, m_img, r_img, r_sammask, r_sammask, self.unaligned_transforms)
        
        r_gray = np.array(r_img)
        
        ref_masks = []
        pixel_number = []
        color_ids = []
        bg_masks = []

        r_sammask = np.array(r_sammask)
        r_sammask_colors = np.unique(r_sammask.reshape(-1, 3), axis=0)
        
        for color in r_sammask_colors:
            mask = np.all(r_sammask == color, axis=-1)
            rg = r_gray[mask]
            avg_pixel_value = np.mean(rg)
            # reflections with different semantics 
            if tuple(color) != (255, 255, 255) and np.sum(rg > 10) > 100:
                pixel_number.append(np.mean(rg))
                color_ids.append(color)
                ref_masks.append(mask)
            elif tuple(color) != (255, 255, 255) and np.sum(rg > 10) < 100 and np.mean(rg) < 5:
                bg_masks.append(mask)
            
        sorted_indices = np.argsort(pixel_number)[::-1]
        sorted_masks = np.array(ref_masks)

        if len(sorted_indices) >= 1:
            save_mask = np.zeros_like(r_gray, dtype=np.uint8)

            for id in sorted_indices:
                max_avg_pixel_mask = sorted_masks[id]
                save_mask[max_avg_pixel_mask] = 255
            for min_avg_pixel_mask in bg_masks:
                save_mask[min_avg_pixel_mask] = 125

        else:
            save_mask = np.zeros_like(r_gray, dtype=np.uint8)
            for min_avg_pixel_mask in bg_masks:
                save_mask[min_avg_pixel_mask] = 125
            if len(sorted_indices) > 0:
                save_mask[sorted_masks[sorted_indices[0]]] = 255

        B = TF.to_tensor(t_img)
        M = TF.to_tensor(m_img)
        R = TF.to_tensor(r_img)
        Sim = TF.to_tensor(save_mask)
        
        dic = {'input': M, 'target_t': B, 'fn': fn, 'real': True, 'target_r': R, 'simmap': Sim}        
        
        if self.flag is not None:
            dic.update(self.flag)
        return dic

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)


class DSRTestDataset1(BaseDataset):
    def __init__(self, datadir, fns=None, size=None, enable_transforms=False, unaligned_transforms=False, round_factor=1, flag=None, if_align=False):
        super(DSRTestDataset1, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns or [fn for fn in os.listdir(join(datadir, 'blended')) if fn.endswith('.png') or fn.endswith('.jpg')]
        self.enable_transforms = enable_transforms
        self.unaligned_transforms = unaligned_transforms
        self.round_factor = round_factor
        self.flag = flag
        self.if_align = if_align
        self.dataset_name = datadir.split('/')[-1]

        if size is not None:
            self.fns = self.fns[:size]

    def align(self, x1, x2, x3, x4):
        h, w = x1.height, x1.width
        h, w = h // 32 * 32, w // 32 * 32
        x1 = x1.resize((w, h))
        x2 = x2.resize((w, h))
        x3 = x3.resize((w, h), Image.NEAREST)
        x4 = x4.resize((w, h), Image.NEAREST)

        return x1, x2, x3, x4

    def __getitem__(self, index):
        fn = self.fns[index]

        t_img = Image.open(join(self.datadir, 'transmission_layer', fn)).convert('RGB')
        m_img = Image.open(join(self.datadir, 'blended', fn)).convert('RGB')
        r_img = Image.open(join(self.datadir, 'reflection_layer', fn)).convert('L')
        r_sammask = Image.fromarray(np.load(join(self.datadir, 'rsam_mask', fn.replace('.jpg', '_filtered.npy').replace('.png', '_filtered.npy'))))
                 
        if self.if_align:
            t_img, m_img, r_img, r_sammask = self.align(t_img, m_img, r_img, r_sammask)

        if self.enable_transforms:
            t_img, m_img, r_img, r_sammask = paired_data_transforms(t_img, m_img, r_img, r_sammask, self.unaligned_transforms)
        
        r_sammask = np.array(r_sammask)
        print(r_sammask.shape)  
        B = TF.to_tensor(t_img)
        M = TF.to_tensor(m_img)
        R = TF.to_tensor(r_img)
        Sim = TF.to_tensor(r_sammask)
        
        dic = {'input': M, 'target_t': B, 'fn': fn, 'real': True, 'target_r': R, 'simmap': Sim}        
        
        if self.flag is not None:
            dic.update(self.flag)
        return dic

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)



class DSRTestDataset_CDR(BaseDataset):
    def __init__(self, datadir, fns=None, size=None, enable_transforms=False, unaligned_transforms=False, round_factor=1, flag=None, if_align=False):
        super(DSRTestDataset_CDR, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns or os.listdir(join(datadir, 'M'))
        self.enable_transforms = enable_transforms
        self.unaligned_transforms = unaligned_transforms
        self.round_factor = round_factor
        self.flag = flag
        self.if_align = if_align

        if size is not None:
            self.fns = self.fns[:size]

    def align(self, x1, x2, x3):
        h, w = x1.height, x1.width
        # h, w = h // 32 * 32, w // 32 * 32
        h, w = h // 48 * 32, w // 48 * 32
        # h, w = 1024, 1024
        x1 = x1.resize((w, h))
        x2 = x2.resize((w, h))
        x3 = x3.resize((w, h), Image.NEAREST)

        return x1, x2, x3

    def __getitem__(self, index):
        fn = self.fns[index]

        # t_img = Image.open(join(self.datadir, 'transmission_layer', fn)).convert('RGB')
        # m_img = Image.open(join(self.datadir, 'blended', fn)).convert('RGB')
        r_img = Image.open(join(self.datadir, 'R', fn.replace('_M_', "_R_"))).convert('RGB')
        t_img = Image.open(join(self.datadir, 'T', fn.replace('_M_', "_T_"))).convert('RGB')
        m_img = Image.open(join(self.datadir, 'M', fn)).convert('RGB')
        r_mask = Image.open(join(self.datadir, 'R_binary_masks', fn.replace('_M_', "_R_"))).convert('L')
        # t_mask = Image.open(join(self.datadir, 'Sam_masks', fn)).convert('L')
        # t_mask = Image.open(join(self.datadir, 'canny_edgemaps', fn)).convert('L')
        # r_mask = Image.open(join(self.datadir, 'RSam_masks', fn)).convert('L')
        # r_mask = Image.open(join(self.datadir, 'point_mask', fn.replace('.jpg', '.png'))).convert('L')
        # r_mask = Image.open(join(self.datadir, 'point_sam_mask', fn.replace('.jpg', '.png'))).convert('L')
        # t_text = read_fns(join(self.datadir, 'transmission_captions', fn.rstrip(fn.split('.')[-1]) + 'txt'))
        if os.path.exists(join(self.datadir, 'reflection_captions', fn)):
            r_text = read_fns(join(self.datadir, 'reflection_captions', fn.rstrip(fn.split('.')[-1]) + 'txt'))
        else:
            r_text = 'reflection glare flare'

        tokenized_r_text = longclip.tokenize(r_text)[0]

        if self.if_align:
            t_img, m_img, r_mask = self.align(t_img, m_img, r_mask)
        if self.enable_transforms:
            t_img, m_img, m_img, m_img, r_mask = paired_data_transforms(t_img, m_img, m_img, m_img, r_mask, self.unaligned_transforms)

        B = TF.to_tensor(t_img)
        M = TF.to_tensor(m_img)
        # R = TF.to_tensor(r_img)
        # t_mask = TF.to_tensor(t_mask)
        r_mask = TF.to_tensor(r_mask)
        # dic = {'input': M, 'target_t': B, 't_mask': t_mask, 'r_canny': r_mask, 'target_r_text': tokenized_r_text, 'fn': fn, 'real': True, 'target_r': M - B}
        # dic = {'input': M, 'target_t': B, 'r_mask': r_mask, 'target_r_text': tokenized_r_text, 'fn': fn, 'real': True, 'target_r': R}
        dic = {'input': M, 'target_t': B, 'r_mask': r_mask, 'target_r_text': tokenized_r_text, 'fn': fn, 'real': True, 'target_r': M - B}
        # dic = {'input': M, 'target_t': B, 'fn': fn, 'real': True, 'target_r': M - B}
        if self.flag is not None:
            dic.update(self.flag)
        return dic

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)
        
class DSRTestDataset_c(BaseDataset):
    def __init__(self, datadir, fns=None, size=None, enable_transforms=False, unaligned_transforms=False, round_factor=1, flag=None, if_align=False):
        super(DSRTestDataset_c, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns or os.listdir(join(datadir, 'blended'))
        self.enable_transforms = enable_transforms
        self.unaligned_transforms = unaligned_transforms
        self.round_factor = round_factor
        self.flag = flag
        self.if_align = if_align

        if size is not None:
            self.fns = self.fns[:size]

    def align(self, x1, x2, x3, x4, x5):
        h, w = x1.height, x1.width
        h, w = h // 32 * 32, w // 32 * 32
        # h, w = 1024, 1024
        x1 = x1.resize((w, h))
        x2 = x2.resize((w, h))
        x3 = x3.resize((w, h))
        x4 = x4.resize((w, h))
        x5 = x5.resize((w, h))

        return x1, x2, x3, x4, x5

    def __getitem__(self, index):
        fn = self.fns[index]

        t_img = Image.open(join(self.datadir, 'transmission_layer', fn)).convert('RGB')
        m_img = Image.open(join(self.datadir, 'blended', fn)).convert('RGB')
        r_img = Image.open(join(self.datadir, 'reflection_layer', fn)).convert('RGB')
        t_mask = Image.open(join(self.datadir, 'Sam_masks', fn)).convert('L')
        r_mask = Image.open(join(self.datadir, 'R1Sam_masks', fn)).convert('L')
        if os.path.exists(join(self.datadir, 'reflection_captions', fn)):
            r_text = read_fns(join(self.datadir, 'reflection_captions', fn.rstrip(fn.split('.')[-1]) + 'txt'))
        else:
            r_text = 'reflection glare flare'

        tokenized_r_text = longclip.tokenize(r_text)[0]

        if self.if_align:
            t_img, m_img, r_img, t_mask, r_mask = self.align(t_img, m_img, r_img, t_mask, r_mask)

        if self.enable_transforms:
            t_img, m_img, r_img, t_mask, r_mask = paired_data_transforms(t_img, m_img, r_img, t_mask, r_mask, self.unaligned_transforms)

        B = TF.to_tensor(t_img)
        M = TF.to_tensor(m_img)
        R = TF.to_tensor(r_img)
        t_mask = TF.to_tensor(t_mask)
        r_mask = TF.to_tensor(r_mask)
        dic = {'input': M, 'target_t': B, 't_mask': t_mask, 'r_mask': r_mask, 'target_r_text': tokenized_r_text, 'fn': fn, 'real': True, 'target_r': R}
        if self.flag is not None:
            dic.update(self.flag)
        return dic

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)


class SIRTestDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None, if_align=False):
        super(SIRTestDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns or os.listdir(join(datadir, 'blended'))
        self.if_align = if_align

        if size is not None:
            self.fns = self.fns[:size]

    def align(self, x1, x2, x3):
        h, w = x1.height, x1.width
        h, w = h // 32 * 32, w // 32 * 32
        x1 = x1.resize((w, h))
        x2 = x2.resize((w, h))
        x3 = x3.resize((w, h))
        return x1, x2, x3

    def __getitem__(self, index):
        fn = self.fns[index]

        t_img = Image.open(join(self.datadir, 'transmission_layer', fn)).convert('RGB')
        r_img = Image.open(join(self.datadir, 'reflection_layer', fn)).convert('RGB')
        m_img = Image.open(join(self.datadir, 'blended', fn)).convert('RGB')

        if self.if_align:
            t_img, r_img, m_img = self.align(t_img, r_img, m_img)

        B = TF.to_tensor(t_img)
        R = TF.to_tensor(r_img)
        M = TF.to_tensor(m_img)

        dic = {'input': M, 'target_t': B, 'fn': fn, 'real': True, 'target_r': R, 'target_r_hat': M - B}
        return dic

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)


class RealDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None):
        super(RealDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns or os.listdir(join(datadir))

        if size is not None:
            self.fns = self.fns[:size]

    def align(self, x):
        h, w = x.height, x.width
        h, w = h // 32 * 32, w // 32 * 32
        x = x.resize((w, h))
        return x

    def __getitem__(self, index):
        fn = self.fns[index]
        B = -1
        m_img = Image.open(join(self.datadir, fn)).convert('RGB')
        M = to_tensor(self.align(m_img))
        data = {'input': M, 'target_t': B, 'fn': fn}
        return data

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)


class FusionDataset(BaseDataset):
    def __init__(self, datasets, fusion_ratios=None):
        self.datasets = datasets
        self.size = sum([len(dataset) for dataset in datasets])
        self.fusion_ratios = fusion_ratios or [1.0 / len(datasets)] * len(datasets)
        ('[i] using a fusion dataset: %d %s imgs fused with ratio %s' % (self.size, [len(dataset) for dataset in datasets], self.fusion_ratios))

    def reset(self):
        for dataset in self.datasets:
            dataset.reset()

    def __getitem__(self, index):
        residual = 1
        for i, ratio in enumerate(self.fusion_ratios):
            if random.random() < ratio / residual or i == len(self.fusion_ratios) - 1:
                dataset = self.datasets[i]
                return dataset[index % len(dataset)]
            residual -= ratio

    def __len__(self):
        return self.size


if __name__ == "__main__":
#     B_path = '/home/chenxiao/NAFNet/datasets/reflection-dataset/train/VOCdevkit/VOC2012/PNGImages/2007_000039.png'
#     R_path = '/home/chenxiao/NAFNet/datasets/reflection-dataset/train/VOCdevkit/VOC2012/PNGImages/2007_000256.png'
#     B_mask_path = '/home/chenxiao/NAFNet/datasets/reflection-dataset/train/VOCdevkit/VOC2012/Sam_masks/2007_000027.npy'
    ds = DSRDataset('/home/chenxiao/NAFNet/datasets/reflection-dataset')
#     t_img = Image.open(B_path).convert('RGB')
#     r_img = Image.open(R_path).convert('RGB')
#     sam_mask = Image.fromarray(np.array(np.load(B_mask_path)))
#     t_img, r_img, sam_mask = ds.align(t_img, r_img, sam_mask)
    syn_model = ReflectionSynthesis()
    efficient_sam_vits_model = build_efficient_sam_vits()

    # file_list = glob('/home/chenxiao/NAFNet/datasets/reflection-dataset/train/VOCdevkit/VOC2012/PNGImages/*g')
    file_list = ['/home/chenxiao/NAFNet/datasets/reflection-dataset/train/VOCdevkit/VOC2012/PNGImages/2007_000032.png', '/home/chenxiao/NAFNet/datasets/reflection-dataset/train/VOCdevkit/VOC2012/PNGImages/2007_000323.png']
    len_list = len(file_list)
    for f in file_list:
        t_img = Image.open(f).convert('RGB')
        remaining_files = [file for file in file_list if file != f]
        random_file = random.choice(remaining_files)
        r_img = Image.open(random_file).convert('RGB')
        mask = Image.open('/home/chenxiao/NAFNet/datasets/reflection-dataset/train/VOCdevkit/VOC2012/SegmentationObject/2007_000323.png').convert('RGB')
        mask = mask.resize((224, 224), resample=Image.NEAREST)  # 使用 BICUBIC 插值算法进行缩放

        print(f"Image size: {mask.size}")

        B, R, M, _ = syn_model(t_img, r_img)
        pseudo_R = M - B
        # masks = run_everything_ours(pseudo_R, efficient_sam_vits_model)

        # r_sammask = np.ones((masks[0].shape[0], masks[0].shape[1], 3)) * 255

        # colors = []
        # while len(colors) < len(masks):
        #     r, g, b = np.random.random(3)
        #     color = (int(r * 255), int(g * 255), int(b * 255))
        #     if color not in colors:
        #         colors.append(color)
     
        # for ann, color_mask in zip(sorted(masks, key=np.sum), colors):
        #     m = ann
        #     r_sammask[m & (r_sammask[:, :, 0]== 255)] = color_mask
        # torch.cuda.empty_cache()
        
        save_M = (M * 255).astype(np.uint8)
        save_B = (B * 255).astype(np.uint8)
        # save_R = (pseudo_R * 255).astype(np.uint8)
        save_R = (R * 255).astype(np.uint8)
        pseudo_R[pseudo_R < 0.2] = 0
        pseudo_R[pseudo_R > 0.2] = 1
        save_mask = (mask * pseudo_R).astype(np.uint8)
        save_mask[:, :130] = 0
        # print(np.unique(save_mask, return_counts=True))
        timestep = str(time.time())
        
        Image.fromarray(save_M).save(os.path.join('/home/chenxiao/NAFNet', timestep +'m.jpg'))
        Image.fromarray(save_B).save(os.path.join('/home/chenxiao/NAFNet', timestep +'b.jpg'))
        Image.fromarray(save_R).save(os.path.join('/home/chenxiao/NAFNet', timestep +'r.jpg'))
        Image.fromarray(save_mask).save(os.path.join('/home/chenxiao/NAFNet', timestep +'mask.jpg'))

        break
        # Image.fromarray((r_sammask).astype(np.uint8)).save(os.path.join('/home/chenxiao/NAFNet/datasets/reflection-dataset/train/voc_syn/rsam_mask', timestep +'.jpg'))
        # np.save(os.path.join('/home/chenxiao/NAFNet/datasets/reflection-dataset/train/voc_syn/rsam_mask', timestep +'.npy'), r_sammask.astype(np.uint8))

        
        # utils.save_image(B, timestep +'transmission.jpg')
    
#     print(np.unique(M))    
#     img_tensor = prepare_image(np.array(M * 255).astype(np.int8), 1024).to("cuda:7")
#     input_point = torch.as_tensor(build_all_layer_point_grids(32, 0, 1)[0] * 1024, dtype=torch.int64).to("cuda:7")
#     input_label = torch.tensor([1 for _ in range(input_point.shape[0])]).to("cuda:7")
#     batched_input = [{'image': img_tensor, 'point_coords': input_point, 'point_labels': input_label, 'original_size': img_tensor.shape[1:]}]
#     refined_masks = ds.sam.individual_forward(batched_input, multimask_output=True)
#     masks = refined_masks[0].cpu().numpy()
        
#     mask_diffmap = np.zeros_like(sam_mask, dtype=np.uint8)
#     color_mask = np.zeros_like(diffmap, dtype=np.uint8)
        
#     h, w, _ = np.array(M).shape
#     colors = [
#     (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
#     (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
#     (255, 128, 0), (128, 255, 0), (0, 255, 128), (0, 128, 255), (128, 0, 255),
#     (255, 0, 128), (255, 128, 128), (128, 255, 128), (128, 128, 255), (255, 128, 255),
#     (192, 192, 192), (128, 0, 64), (64, 128, 0), (0, 128, 64), (64, 0, 128),
#     (192, 128, 128), (128, 192, 128), (128, 128, 192), (192, 128, 192), (192, 192, 128),
#     (64, 64, 0), (0, 64, 64), (64, 0, 64), (192, 64, 64), (64, 192, 64),
#     (64, 64, 192), (192, 64, 192), (192, 192, 64), (64, 192, 192), (192, 64, 192),
#     (0, 0, 64), (0, 64, 0), (64, 0, 0), (128, 64, 0), (0, 128, 64),
#     (0, 64, 128), (64, 0, 128), (128, 0, 64), (128, 64, 128), (64, 128, 64)
#     ]
    
#     for single_mask, color in zip(masks[:50], colors):
#         mask = single_mask.astype(int)
#         mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
#         r, g, b = color
#         mask_rgb = np.dstack((mask * r, mask * g, mask * b))
#         color_mask = np.maximum(color_mask, mask_rgb)
        
#         diff = diffmap[mask == 1] 
        
#         print('diff', np.average(diff), diff.shape)
#         if np.average(diff) > 0.7:
#             mask_diffmap += mask * 255
        
#     mask_diffmap[mask_diffmap > 255] = 255
#     fn = os.path.basename(B_path)
        
#     B = TF.to_tensor(B)
#     R = TF.to_tensor(R)
#     M = TF.to_tensor(M)
    
#     mask_diffmap = TF.to_tensor(mask_diffmap).float()
#     diffmap = TF.to_tensor(diffmap).float()
#     color_mask = TF.to_tensor(color_mask).float()
        
#     timestep = str(time.time()) 
#     utils.save_image(M, timestep +'mix.jpg')
#     utils.save_image(B, timestep +'transmission.jpg')
#     utils.save_image(mask_diffmap, timestep +'maskdiffmap.jpg')
#     utils.save_image(diffmap, timestep +'diffmap.jpg')
#     utils.save_image(color_mask, timestep +'colormask.jpg')
