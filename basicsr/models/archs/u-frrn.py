from __future__ import print_function
import torch
import torch.nn as nn
import random
import math
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2


args = {

    'input_resolution': 256,

    'guas_sigma_min': 0.5,
    'guas_sigma_max': 4.0,
    'guas_kernel': 7,
    'guas_m': 1.8,

    'scut_masknum': 500,
    'scut_bmin': 60,
    'scut_bmax': 100,
    'scut_thrs': 0.05,

    'iccv17_sigma_min': 0.0,
    'iccv17_sigma_max': 4.0,
    'iccv17_kernel': 7,

    'blend_mu': 0.15,
    'blend_sigma': 0.05,
    'blend_min': 0.05,
    'blend_max': 0.25,

}

##########################################################################
#
# guassian image systhesis
#
##########################################################################
def get_blend_ratio():
    ratio = random.gauss(args['blend_mu'], args['blend_sigma'])
    ratio = max(args['blend_min'], min(args['blend_max'], ratio))
    return ratio

def get_gauss_val(mu, sigma, mins, maxs):
    ratio = random.gauss(mu, sigma)
    ratio = max(mins, min(maxs, ratio))
    return ratio

def blend_images(x1, x2, ratio):
    return x1 * ratio + x2 * (1 - ratio)

def blend_gauss_part(B,R,mask):
    B2 = B * (1-mask)
    R2 = blur_iccv17(R) * mask
    I = B2 + R2
    torch.clamp(I, min=0, max=1)
    return I, B2, R2

def blur_iccv17(R):
    kernel_size = (args['iccv17_kernel'] , args['iccv17_kernel'])
    sigma = random.uniform(args['iccv17_sigma_min'],args['iccv17_sigma_max'])
    npR = np.transpose(R.numpy(),(0,2,3,1))

    # calculate the m
    for i in range(0,R.size()[0]):
        npR_i = npR[i][:][:][:]
        npR_i = cv2.GaussianBlur(npR_i, kernel_size, sigma)
        tensorR = transforms.ToTensor()(npR_i*255)
        R[i][:][:][:] = tensorR
    return R


def Gshape(x, y, a1, a2, b1, b2, c):
    A = 1 / (2 * math.pi * b1 * b2 * math.sqrt(1 - c * c))
    B = (-1) / (2 - 2 * c * c)
    C = (x - a1) * (x - a1) / (b1 * b1) - 2 * c * (x - a1) * (y - a2) / (b1 * b2) + (y - a2) * (y - a2) / (b2 * b2)
    D = A * math.exp(B * C)
    return math.exp(-C)


def Gauss_mask():
    mask = torch.zeros(3, args['input_resolution'], args['input_resolution'])
    a1 = random.randint(0, args['input_resolution'] - 1)
    a2 = random.randint(0, args['input_resolution'] - 1)
    b1 = random.uniform(args['scut_bmin'], args['scut_bmax'])
    b2 = random.uniform(args['scut_bmin'], args['scut_bmax'])

    c = random.uniform(-0.7, 0.7)
    v, w, h = mask.size()
    Rweight = get_blend_ratio()
    for i in range(w):
        for j in range(h):
            mask[0:3, i, j] = mask[0:3, i, j] + Gshape(i, j, a1, a2, b1, b2, c) * Rweight

    return mask

def build_masks():
    mask_loader = torch.zeros(args['scut_masknum'],3,args['input_resolution'],args['input_resolution'])
    for i in range(0,args['scut_masknum']):
        print(i)
        gauss_num = random.randint(5,8)
        for j in range(0,gauss_num):
            mask_loader[i] = mask_loader[i] + Gauss_mask()

        # torchvision.utils.save_image(mask_loader[i], './noise_image/image/noise_'+str(i)+'.jpg')
    return mask_loader

##########################################################################
#
# user-guidence
#
##########################################################################

def My_Canny(x, th_ratio):
    gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

    # Otsu filter
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # th_ratio = random.random()
    lower_ths = ret*th_ratio
    upper_ths = ret

    edge = 255 - cv2.Canny(gray, lower_ths, upper_ths) # sensitive to lightness

    return edge


def H_map(x, e_type, th_ratio, iscuda = None):
    if iscuda is not None:
        x = x.data.cpu()
    npX = np.transpose(x.numpy(), (0, 2, 3, 1))
    R = torch.zeros((x.size(0), 1, x.size(2), x.size(3)))
    for i in range(0, x.size(0)):
        x_i = npX[i][:][:][:]*255

        x_i = np.uint8(x_i)

        if e_type == 'canny':
            edge_i = My_Canny(x_i, th_ratio)
        elif e_type == 'sobel':
            edge_i = My_Sobel(x_i)

        edge_i = edge_i[:, :, np.newaxis]
        tensorR = transforms.ToTensor()(edge_i)
        R[i][:][:][:] = tensorR

    if iscuda is not None:
        return torch.autograd.Variable(R).cuda()
    else:
        return R

def ToBinary(x, th_ratio=0, iscuda=None):
    if iscuda is not None:
        x = x.data.cpu()
    npX = np.transpose(x.numpy(), (0, 2, 3, 1))
    R = torch.zeros((x.size(0), 1, x.size(2), x.size(3)))
    for i in range(0, x.size(0)):
        x_i = npX[i][:][:][:]*255

        x_i = np.uint8(x_i)
        gray_i = cv2.cvtColor(x_i, cv2.COLOR_BGR2GRAY)
        th = int(255.0*th_ratio)
        _, bin_i = cv2.threshold(gray_i, th, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        bin_i = bin_i[:, :, np.newaxis]
        tensorR = transforms.ToTensor()(bin_i)
        R[i][:][:][:] = tensorR

    if iscuda is not None:
        return torch.autograd.Variable(R).cuda()
    else:
        return R

def Partial_Map(edge, size_ratio=1.0, scale_ratio=1.0):
    down_edge = edge
    (batch, channel, width, height) = down_edge.size()
    patial_edge = torch.ones(batch, channel, width, height)

    for i in range(batch):
        w = int(width * scale_ratio)
        h = int(height * scale_ratio)
        x = random.randint(0, width - w)
        y = random.randint(0, height - h)

        mask_xl = int(x)
        mask_xr = int(x + w)
        mask_yu = int(y)
        mask_yd = int(y + h)

        down_edge_total = torch.sum(1.0 - down_edge[i,:,:,:])
        if down_edge_total < 3:
            break


        patial_edge[i, :, :, :] = Cal_Pos(down_edge_total, down_edge[i,:,:,:], mask_xl, mask_xr, mask_yu, mask_yd)

    return patial_edge

def Cal_Pos(down_edge_total, down_edge, mask_xl, mask_xr, mask_yu, mask_yd):
    (channel, width, height) = down_edge.size()
    edge_mask = torch.ones((down_edge.size(0), down_edge.size(1), down_edge.size(2)))
    mask_xl -= int(random.uniform(0, mask_xl))
    mask_xr += int(random.uniform(0, width-1-mask_xr))
    mask_yu -= int(random.uniform(0, mask_yu))
    mask_yd += int(random.uniform(0, height-1-mask_yd))

    edge_mask[:, mask_xl:mask_xr, mask_yu:mask_yd] = 0.0
    patial_edge = torch.clamp(down_edge + edge_mask, min=0, max=1)

    partial_edge_total = torch.sum(1.0 - patial_edge[:, :, :])
    # print('partial: ', down_edge_total, partial_edge_total)
    if partial_edge_total > down_edge_total / 3.0:
        return patial_edge
    else:
        return Cal_Pos(down_edge_total, down_edge, mask_xl, mask_xr, mask_yu, mask_yd)

def Down_H_map(edge, size_ratio):
    (batch, channel, width, height) = edge.size()
    sizex = (int(width/size_ratio), int(height/size_ratio))
    down_edge = F.upsample(edge, size=sizex, mode='bilinear')
    return down_edge


def ToGray(x, x_type=None):
    if x_type is not None:
        x = x.data.cpu()
    npX = np.transpose(x.numpy(), (0, 2, 3, 1))
    R = torch.zeros((x.size(0), 1, x.size(2), x.size(3)))
    for i in range(0, x.size(0)):
        x_i = npX[i][:][:][:]*255

        x_i = np.uint8(x_i)
        x_i = cv2.GaussianBlur(x_i, (args['guas_kernel'], args['guas_kernel']), 3)
        gray_i = cv2.cvtColor(x_i, cv2.COLOR_BGR2GRAY)
        # ret, gray_i = cv2.threshold(gray_i, 127, 255, cv2.THRESH_BINARY)

        gray_i = gray_i[:, :, np.newaxis]
        tensorR = transforms.ToTensor()(gray_i)
        R[i][:][:][:] = tensorR

    if x_type is not None:
        return torch.autograd.Variable(R)#.cuda()
    else:
        return R

def Edge_Sparse(B, R, mask):
    for b in range(B.size(0)):
        for c in range(B.size(1)):
            for i in range(B.size(2)):
                for j in range(B.size(3)):
                    if B[b, c, i, j] == 0 and R[b, c, i, j] == 0 and mask[b, c, i, j] > 0.7:
                        B[b, c, i, j] = 1
                    elif B[b, c, i, j] == 0 and R[b, c, i, j] == 0 and mask[b, c, i, j] < 0.3:
                        R[b, c, i, j] = 1
                    else:
                        continue


    return B, R

class VGG16Net(nn.Module):

    def __init__(self):
        super(VGG16Net, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            # nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.features(x)

    def load_pretrained(self, pretrained_pth):
        dst_state = self.state_dict()
        src_state = torch.load(pretrained_pth)
        # print(type(dst_state.values()[0]))
        # print(type(src_state.values()[0]))
        for dst_param, src_param in zip(dst_state.values(), src_state.values()):
            dst_param.copy_(src_param)



class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        h = self.residual(x)
        return h + x


class multiDeblendNet(nn.Module):
    def __init__(self):
        super(multiDeblendNet, self).__init__()

        # 4 x 256 x 256
        # 32 x 256 x 256
        self.conv0 = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=5 , stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # 64 x 128 x 128
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # 128 x 64 x 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(66, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # 256 x 32 x 32
        self.conv3 = nn.Sequential(
            nn.Conv2d(130, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # 512 x 32 x 32
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.res = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
        )
        # 256 x 64 x 64
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # 128 x 64 x 64
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # 64 x 128 x 128
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(130, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # 32 x 256 x 256
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(66, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # 3 x 256 x 256
        self.conv = nn.Sequential(
            nn.Conv2d(34, 7, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x, edge_x1, edge_x2 = x[:, :3, :, :], x[:, 3, :, :].unsqueeze(1), x[:, 4, :, :].unsqueeze(1)
        e1 = edge_x1
        e2 = edge_x2
        x0 = torch.cat((x, e1), 1)
        x0 = torch.cat((x0, e2), 1)
        h0 = self.conv0(x0)
        h1 = self.conv1(h0)
        e3 = Down_H_map(edge_x1, 2)
        e4 = Down_H_map(edge_x2, 2)
        h1 = torch.cat((h1, e3), 1)
        h1 = torch.cat((h1, e4), 1)
        h2 = self.conv2(h1)
        e5 = Down_H_map(edge_x1, 4)
        e6 = Down_H_map(edge_x2, 4)
        h2 = torch.cat((h2, e5), 1)
        h2 = torch.cat((h2, e6), 1)
        h3 = self.conv3(h2)
        h4 = self.conv4(h3)

        h_res = self.res(h4)

        h = self.deconv4(h_res)
        h = self.deconv3(h)
        h = torch.cat((h, e5), 1)
        h = torch.cat((h, e6), 1)
        h = self.deconv2(h)
        h = torch.cat((h, e3), 1)
        h = torch.cat((h, e4), 1)
        h = self.deconv1(h)
        h = torch.cat((h, e1), 1)
        h = torch.cat((h, e2), 1)
        h = self.conv(h)

        y1 = h[:, 0:3, :, :]
        y2 = h[:, 3:6, :, :]
        mask = h[:, 6:7, :, :]
        # y1, y2 = h.chunk(2, dim=1)
        return y1, y2 , mask



class InpNet(nn.Module):

    def __init__(self):
        super(InpNet, self).__init__()
        # 4 x 256 x 256
        # 32 x 256 x 256
        self.conv0 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=5 , stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # 64 x 128 x 128
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # 128 x 64 x 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # 256 x 32 x 32
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # 512 x 32 x 32
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.res = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
        )
        # 256 x 64 x 64
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # 128 x 64 x 64
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # 64 x 128 x 128
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # 32 x 256 x 256
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # 3 x 256 x 256
        self.conv = nn.Sequential(
            # nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2),
            # nn.Sigmoid(),
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid(),
        )


    def forward(self, x):
        h0 = self.conv0(x)
        h1 = self.conv1(h0)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h4 = self.conv4(h3)

        h_res = self.res(h4)

        h = self.deconv4(h_res)
        h = torch.cat([h, h3], 1)
        h = self.deconv3(h)
        h = torch.cat([h, h2], 1)
        h = self.deconv2(h)
        h = torch.cat([h, h1], 1)
        h = self.deconv1(h)
        h = torch.cat([h, h0], 1)
        y = self.conv(h)

        return y


class CriticNet(nn.Module):

    def __init__(self):
        super(CriticNet, self).__init__()

        self.critic = nn.Sequential(
            # 3 x 256 x 256
            nn.Conv2d(3, 32, kernel_size=6 , stride=2, padding=2),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2),
            # 32 x 128 x 128
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            # 64 x 64 x 64
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            # 128 x 32 x 32
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            # 128 x 32 x 32
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            # 256 x 16 x 16
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            # 512 x 8 x 8
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            # 512 x 4 x 4
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid(),
            # 1 x 1 x 1
        )

    def forward(self, x):
        return self.critic(x)

if __name__ == '__main__':
    import time
    # net = multiDeblendNet().to('cuda:2')  
    net = InpNet().to('cuda:2') 


    num_runs = 2  # 总共运行的次数
    start_time = time.time()
    input_tensor = torch.randn(1, 4, 224, 224).to('cuda:2')  

    with torch.no_grad():
        for _ in range(num_runs):
            _ = net(input_tensor)

    end_time = time.time()
    total_time = end_time - start_time
    throughput = (num_runs) / total_time
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {throughput:.2f} img/s")

    # input2 = torch.randn(1, 1, 224, 224)
    # input3 = torch.randn(1, 1, 224, 224)
    
    # from thop import profile, clever_format
    # flops, params = profile(net, inputs=(input1, ))
    # flops, params = clever_format([flops, params], "%.3f")
    # print(flops, params)


    # from ptflops import get_model_complexity_info
    # macs, params = get_model_complexity_info(net, (5, 224, 224), as_strings=True, print_per_layer_stat=False)
    # print(f"MACs: {macs}")
    # print(f"Params: {params}") 

    # 12.543G 24.487M
    # 17.271G 25.158M
    # 29.814G 49.635M

    # 17.28 GMac  25.16 M
    #  12.55 GMac 24.49 M