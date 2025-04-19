# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import importlib
import torch
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info
import torch
import torch.nn as nn
import gc
import torch.nn.functional as F
from basicsr.models.vgg import Vgg19
# from LongCLIP.model import longclip
from PIL import Image
from torchvision import transforms
# from dift_src.models.dift_sd import SDFeaturizer
import datetime

def compute_grad(img):
    gradx = img[..., 1:, :] - img[..., :-1, :]
    grady = img[..., 1:] - img[..., :-1]
    return gradx, grady


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, predict, target):
        predict_gradx, predict_grady = compute_grad(predict)
        target_gradx, target_grady = compute_grad(target)

        return self.loss(predict_gradx, target_gradx) + self.loss(predict_grady, target_grady)


class Mask_GradientLoss(nn.Module):
    def __init__(self):
        super(Mask_GradientLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, predict, target, mask):
        weight_map = torch.zeros_like(mask)
        weight_map[mask != 0] = 1
        weight_map[mask == 0] = 1
        # predict = predict * mask        
        # target = target * masks
        
        predict_gradx, predict_grady = compute_grad(predict * weight_map)
        target_gradx, target_grady = compute_grad(target * weight_map)

        return self.loss(predict_gradx, target_gradx) + self.loss(predict_grady, target_grady)


class Mask_MSELoss(nn.Module):
    def __init__(self):
        super(Mask_MSELoss, self).__init__()
        # 定义Sobel滤波器
        # self.sobel_x = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        # self.sobel_y = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        
        # # 初始化Sobel滤波器的权重
        # sobel_x_kernel = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3).repeat(3, 1, 1, 1).cuda(torch.device('cuda:6'))
        # sobel_y_kernel = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view(1, 1, 3, 3).repeat(3, 1, 1, 1).cuda(torch.device('cuda:6'))
        # self.sobel_x.weight.data = sobel_x_kernel
        # self.sobel_y.weight.data = sobel_y_kernel
        
    def forward(self, predict, target, mask):
        """_summary_

        Args:
            predict (_type_): prediction reflection img
            target (_type_): edge map

        Returns:
            _type_: _description_
        """
        # predict = torch.mean(predict, dim=1, keepdim=True)
        # pred_sobel = torch.sqrt(self.sobel_x(predict) ** 2 + self.sobel_y(predict) ** 2)
        # weight_map[target == 1.0] = 0.95
        # weight_map[target == 0.0] = 0.05
        # return torch.mean(torch.abs(pred_sobel - target) * weight_map)
        weight_map = torch.zeros_like(mask)
        weight_map[mask != 0] = 1
        weight_map[mask == 0] = 1
        
        squared_error = (predict - target) ** 2
        masked_squared_error = squared_error * weight_map
        loss = masked_squared_error.mean()
        return loss
    
class MultipleLoss(nn.Module):
    def __init__(self, losses, weight=None):
        super(MultipleLoss, self).__init__()
        self.losses = nn.ModuleList(losses)
        self.weight = weight or [1 / len(self.losses)] * len(self.losses)

    def forward(self, predict, target):
        total_loss = 0
        for weight, loss in zip(self.weight, self.losses):
            total_loss += loss(predict, target) * weight
           
        return total_loss


class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False


class VGGLoss(nn.Module):
    def __init__(self, vgg=None, device=None, weights=None, indices=None, normalize=True):
        super(VGGLoss, self).__init__()
        if vgg is None:
            self.vgg = Vgg19().to(device)
        else:
            self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10 / 1.5]
        self.indices = indices or [2, 7, 12, 21, 30]
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).to(device)
        else:
            self.normalize = None

    def forward(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
   
        x_vgg, y_vgg = self.vgg(x , self.indices), self.vgg(y , self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

        return loss


class ReconsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, out_t, out_r, out_rr, input_i):
        content_diff = self.criterion(out_t + out_r + out_rr, input_i[:, :3, :, :])
        return content_diff


class ExclusionLoss(nn.Module):
    def __init__(self, level=3, eps=1e-6):
        super().__init__()
        self.level = level
        self.eps = eps

    def forward(self, img_T, img_R):
        grad_x_loss = []
        grad_y_loss = []

        for l in range(self.level):
            grad_x_T, grad_y_T = compute_grad(img_T)
            grad_x_R, grad_y_R = compute_grad(img_R)

            alphax = (2.0 * torch.mean(torch.abs(grad_x_T))) / (torch.mean(torch.abs(grad_x_R)) + self.eps)
            alphay = (2.0 * torch.mean(torch.abs(grad_y_T))) / (torch.mean(torch.abs(grad_y_R)) + self.eps)

            gradx1_s = (torch.sigmoid(grad_x_T) * 2) - 1  # mul 2 minus 1 is to change sigmoid into tanh
            grady1_s = (torch.sigmoid(grad_y_T) * 2) - 1
            gradx2_s = (torch.sigmoid(grad_x_R * alphax) * 2) - 1
            grady2_s = (torch.sigmoid(grad_y_R * alphay) * 2) - 1

            grad_x_loss.append(((torch.mean(torch.mul(gradx1_s.pow(2), gradx2_s.pow(2)))) + self.eps) ** 0.25)
            grad_y_loss.append(((torch.mean(torch.mul(grady1_s.pow(2), grady2_s.pow(2)))) + self.eps) ** 0.25)

            img_T = F.interpolate(img_T, scale_factor=0.5, mode='bilinear')
            img_R = F.interpolate(img_R, scale_factor=0.5, mode='bilinear')
        loss_gradxy = torch.sum(sum(grad_x_loss) / 3) + torch.sum(sum(grad_y_loss) / 3)

        return loss_gradxy / 2

class clip_distillLoss(nn.Module):
    def __init__(self):
        super(clip_distillLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.model, self.preprocess = longclip.load("/home/chenxiao/NAFNet/longclip-L.pt", device="cuda:6")

    def forward(self, predict_middle, lq_img):
        with torch.no_grad():
            _, clip_visual_feat = self.model.encode_image(lq_img)
            
        return self.loss(F.normalize(clip_visual_feat, dim=-1), F.normalize(predict_middle, dim=-1)) 

# class ti_alignLoss(nn.Module):
#     def __init__(self):
#         super(ti_alignLoss, self).__init__()
#         self.model, self.preprocess = clip.load("/home/chenxiao/NAFNet/longclip-L.pt", device="cuda:6")
#         means = [0.48145466, 0.4578275, 0.40821073]
#         stds = [0.26862954, 0.26130258, 0.27577711]   
#         self.means = torch.tensor(means).reshape(1, 3, 1, 1)
#         self.stds = torch.tensor(stds).reshape(1, 3, 1, 1)

#     def forward(self, pred_t, pred_r, text_t):
#         # normalized_t = (pred_t - self.means) / self.stds
#         # normalized_r = (pred_r - self.means) / self.stds
#         print(text_t.shape)
#         print(normalized_t.shape, normalized_r.shape, text_t.shape)
#         # normalized features
#         image_features = image_features / image_features.norm(dim=1, keepdim=True)
#         text_features = text_features / text_features.norm(dim=1, keepdim=True)

#         # cosine similarity as logits
#         logit_scale = self.logit_scale.exp()
#         logits_per_image = logit_scale * image_features @ text_features.t()
#         logits_per_text = logits_per_image.t()
import pytorch_ssim
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, reduction='mean'):
        """
        SSIM Loss class for comparing two images.

        Args:
            window_size (int): Size of the Gaussian filter window. Default is 11.
            reduction (str): Specifies the reduction to apply to the output: 'mean', 'sum', or 'none'.
        """
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.reduction = reduction
        self.channel = 1  # To initialize later based on input

    def gaussian_window(self, window_size, channel):
        """
        Create a 2D Gaussian kernel.

        Args:
            window_size (int): Size of the Gaussian kernel.
            channel (int): Number of input channels (used for broadcasting).
        
        Returns:
            torch.Tensor: Gaussian kernel of shape (channel, 1, window_size, window_size).
        """
        sigma = 1.5  # Standard deviation for Gaussian
        coords = torch.arange(window_size).float() - (window_size - 1) / 2.0
        g = torch.exp(-coords**2 / (2 * sigma**2))
        g = g / g.sum()
        kernel_2d = g[:, None] @ g[None, :]
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        kernel = kernel_2d.repeat(channel, 1, 1, 1)  # Repeat for each channel
        return kernel

    def ssim(self, img1, img2):
        """
        Compute SSIM between two images.

        Args:
            img1 (torch.Tensor): Input image 1 of shape (N, C, H, W).
            img2 (torch.Tensor): Input image 2 of shape (N, C, H, W).

        Returns:
            torch.Tensor: SSIM value for each batch.
        """
        channel = img1.size(1)
        kernel = self.gaussian_window(self.window_size, channel).to(img1.device)
        padding = self.window_size // 2

        # Compute local mean, variance, and covariance
        mu1 = F.conv2d(img1, kernel, padding=padding, groups=channel)
        mu2 = F.conv2d(img2, kernel, padding=padding, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, kernel, padding=padding, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, kernel, padding=padding, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, kernel, padding=padding, groups=channel) - mu1_mu2

        # Constants to stabilize division
        C1 = 0.01**2
        C2 = 0.03**2

        # SSIM calculation
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        if self.reduction == 'mean':
            return ssim_map.mean()
        elif self.reduction == 'sum':
            return ssim_map.sum()
        else:
            return ssim_map

    def forward(self, img1, img2):
        """
        Compute SSIM loss.

        Args:
            img1 (torch.Tensor): Input image 1 of shape (N, C, H, W).
            img2 (torch.Tensor): Input image 2 of shape (N, C, H, W).

        Returns:
            torch.Tensor: SSIM loss value.
        """
        ssim_value = self.ssim(img1, img2)
        return 1 - ssim_value  # SSIM loss = 1 - SSIM
       
def init_loss():
    loss_dic = {}
    # pixel_loss = MultipleLoss([nn.MSELoss(), GradientLoss()], [0.3, 0.6])
    mask_pixel_loss = MultipleLoss([nn.MSELoss(), GradientLoss()], [0.3, 0.6])
    pixel_loss2 = MultipleLoss([nn.MSELoss(), GradientLoss()], [0.3, 0.6])

    loss_dic['t_pixel'] = mask_pixel_loss
    loss_dic['r_pixel'] = pixel_loss2
    loss_dic['t_ssim'] = SSIMLoss(window_size=11, reduction='mean')
    # loss_dic['recons'] = ReconsLoss()
    loss_dic['exclu'] = ExclusionLoss(level=3)
    # loss_dic['clip_distill'] = clip_distillLoss()
    # loss_dic['contrast'] = ti_alignLoss()
    return loss_dic


loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')


class ImageRestorationModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageRestorationModel, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
                   self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        self.scale = int(opt['scale'])
        # self.dift = SDFeaturizer('stabilityai/stable-diffusion-2-1')

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        
        self.vgg = Vgg19(requires_grad=False).to(device=self.device)
        loss_dic = init_loss()
        loss_dic['vgg'] = VGGLoss(self.vgg, self.device)
        self.loss_dic = loss_dic
    
        self.setup_optimizers()
        self.setup_schedulers()
        # self.dift = SDFeaturizer('stabilityai/stable-diffusion-2-1', device=self.device)

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
        #         if k.startswith('module.offsets') or k.startswith('module.dcns'):
        #             optim_params_lowlr.append(v)
        #         else:
                optim_params.append(v)
            # else:
            #     logger = get_root_logger()
            #     logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        # ratio = 0.1

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
                                                **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params,
                                               **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                                **train_opt['optim_g'])
            pass
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data, is_val=False):
        self.lq = data['input'].to(self.device)
        self.gt = data['target_t'].to(self.device)
        self.sim = data['mask'].to(self.device)
        self.r_gt = data['target_r'].to(self.device)
    
    def grids(self):
        b, c, h, w = self.gt.size()
        self.original_size = (b, c, h, w)

        assert b == 1
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)


        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale
        #adaptive step_i, step_j
        num_row = (h - 1) // crop_size_h + 1
        num_col = (w - 1) // crop_size_w + 1

        import math
        step_j = crop_size_w if num_col == 1 else math.ceil((w - crop_size_w) / (num_col - 1) - 1e-8)
        step_i = crop_size_h if num_row == 1 else math.ceil((h - crop_size_h) / (num_row - 1) - 1e-8)

        scale = self.scale
        step_i = step_i//scale*scale
        step_j = step_j//scale*scale

        parts = []
        idxes = []

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size_h >= h:
                i = h - crop_size_h
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size_w >= w:
                    j = w - crop_size_w
                    last_j = True
                parts.append(self.lq[:, :, i // scale :(i + crop_size_h) // scale, j // scale:(j + crop_size_w) // scale])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w))
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += self.outs[cnt]
            count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

        self.output = (preds / count_mt).to(self.device)
        self.lq = self.origin_lq

    def optimize_parameters(self, current_iter, tb_logger):
        
        self.optimizer_g.zero_grad()

        if self.opt['train'].get('mixup', False):
            self.mixup_aug()
        
        b, _, _, _ = self.lq.size()
        
        # for i in range(b):
        #     with torch.no_grad():
        #         if self.sim[i].sum() == 0:
        #            img_tensor = (self.lq[i] - 0.5) * 2
        #            img_tensor1 = (self.gt[i]  - 0.5) * 2         
            
        #            ft, ft1 = self.dift.forward(img_tensor, img_tensor1,
        #                   prompt='', t=51, up_ft_index=1, ensemble_size=8)
        
        #            _, _, h, w = ft.shape
        #            ft_flat = ft.view(ft.size(0), ft.size(1), -1) 
        #            ft1_flat = ft1.view(ft1.size(0), ft1.size(1), -1)
        #            similarity = F.cosine_similarity(ft_flat, ft1_flat, dim=1).view(1, 1, h, w)
        #            self.sim[i] = nn.Upsample(size=(224, 224), mode='bilinear')(similarity)[0]
        #            gc.collect()
        #            torch.cuda.empty_cache()

        preds_t, preds_r = self.net_g(torch.cat([self.lq, self.sim], dim=1))
        
        self.output = preds_t[-1]

        l_total = 0
        loss_dict = OrderedDict()
        
        loss_t_pixel = self.loss_dic['t_pixel'](preds_t, self.gt)
        loss_dict['t_pixel'] = loss_t_pixel
        
        loss_t_vgg = self.loss_dic['vgg'](preds_t, self.gt) 
        loss_dict['t_vgg'] = loss_t_vgg
        
        loss_t_ssim = self.loss_dic['t_ssim'](preds_t, self.gt)
        loss_dict['t_ssim'] = loss_t_ssim
        # loss_feat_align = self.loss_dic['clip_distill'](middle_pred, self.lq)
        # loss_dict['feat_align'] = loss_feat_align
        
        loss_r_pixel = self.loss_dic['r_pixel'](preds_r, self.r_gt)
        loss_dict['r_pixel'] = loss_r_pixel
        
        loss_exclu = self.loss_dic['exclu'](preds_t, preds_r)
        loss_dict['exclu'] = loss_exclu

        # l_total = 0.01 * loss_t_vgg + loss_t_pixel         
        # l_total = 0.01 * loss_t_vgg + loss_t_pixel + loss_r_pixel + loss_exclu  + loss_feat_align
        # else:
        l_total = 0.01 * loss_t_vgg + loss_t_pixel + loss_r_pixel + loss_exclu + loss_t_ssim

        l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

        l_total.backward()
        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()


        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = len(self.lq)
            outs = []
            r_outs = []
            
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                
                pred_t, pred_r  = self.net_g(torch.cat([self.lq[i:j], self.sim[i:j]], dim=1), Train=False)
                
                # if isinstance(pred, list):
                #     pred = pred[-1]
                
                # if isinstance(pred, tuple):
                #     pred, refl = pred
                outs.append(pred_t.detach().cpu())
                r_outs.append(pred_r.detach().cpu())
                i = j

            self.output = torch.cat(outs, dim=0)
            self.r_output = torch.cat(r_outs, dim=0)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = 'real'
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        rank, world_size = get_dist_info()
        if rank == 0:
            pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            if idx % world_size != rank:
                continue
            
            # img_name = osp.splitext(osp.basename(val_data['fn'][0]))[0]
            img_name = val_data['fn'][0]
            self.feed_data(val_data, is_val=True)
            if self.opt['val'].get('grids', False):
                self.grids()

            self.test()

            if self.opt['val'].get('grids', False):
                self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            sim_img = tensor2img([visuals['sim']], rgb2bgr=rgb2bgr)
            r_img = tensor2img([visuals['r_result']], rgb2bgr=rgb2bgr)

            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if True:
                if sr_img.shape[2] == 6:
                    L_img = sr_img[:, :, :3]
                    R_img = sr_img[:, :, 3:]

                    # visual_dir = osp.join('visual_results', dataset_name, self.opt['name'])
                    visual_dir = osp.join(self.opt['path']['visualization'], dataset_name)

                    imwrite(L_img, osp.join(visual_dir, f'{img_name}_L.png'))
                    imwrite(R_img, osp.join(visual_dir, f'{img_name}_R.png'))
                else:
                    if self.opt['is_train']:

                        save_img_path = osp.join(self.opt['path']['visualization'],
                                                 img_name,
                                                 f'{img_name}_{current_iter}.png')

                        save_ref_path = osp.join(self.opt['path']['visualization'],
                                                 img_name,
                                                 f'{img_name}_{current_iter}_reflection.png')

                        save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                 img_name,
                                                 f'{img_name}_{current_iter}_gt.png')
                        save_sim_img_path = osp.join(self.opt['path']['visualization'],
                                                 img_name,
                                                 f'{img_name}_{current_iter}_sim.png')

                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}.png')
                        save_ref_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_reflection.png')
                        save_gt_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_gt.png')
                        save_sim_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_sim.png')
                    imwrite(sr_img, save_img_path)
                    imwrite(gt_img, save_gt_img_path)
                    imwrite(r_img, save_ref_path)
                    imwrite(sim_img, save_sim_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test {img_name}')
        if rank == 0:
            pbar.close()

        # current_metric = 0.
        collected_metrics = OrderedDict()
        if with_metrics:
            for metric in self.metric_results.keys():
                collected_metrics[metric] = torch.tensor(self.metric_results[metric]).float().to(self.device)
            collected_metrics['cnt'] = torch.tensor(cnt).float().to(self.device)

            self.collected_metrics = collected_metrics
        
        keys = []
        metrics = []
        for name, value in self.collected_metrics.items():
            keys.append(name)
            metrics.append(value)
        metrics = torch.stack(metrics, 0)
        # torch.distributed.reduce(metrics, dst=0)
        if self.opt['rank'] == 0:
            metrics_dict = {}
            cnt = 0
            for key, metric in zip(keys, metrics):
                if key == 'cnt':
                    cnt = float(metric)
                    continue
                metrics_dict[key] = float(metric)

            for key in metrics_dict:
                metrics_dict[key] /= cnt

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger, metrics_dict)
        return 0.

    def nondist_validation(self, *args, **kwargs):
        logger = get_root_logger()
        logger.warning('nondist_validation is not implemented. Run dist_validation.')
        self.dist_validation(*args, **kwargs)


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger, metric_dict):
        log_str = f'Validation {dataset_name}, \t'
        for metric, value in metric_dict.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)

        log_dict = OrderedDict()
        # for name, value in loss_dict.items():
        for metric, value in metric_dict.items():
            log_dict[f'm_{metric}'] = value

        self.log_dict = log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        out_dict['sim'] = self.sim.detach().cpu()
        out_dict['r_result'] = self.r_output.detach().cpu()

        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("/home/chenxiao/NAFNet/RN50.pt", device=device)

    image = preprocess(Image.open("/home/chenxiao/NAFNet/CLIP/CLIP.png")).unsqueeze(0).to(device)
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
 
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        print(image.shape, text.shape)
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]