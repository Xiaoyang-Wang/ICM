import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

import numpy as np

def gray_scale(img):
    if img.shape[1] is not 3:
        return img
    else:
        gray_img = 0.3 * img[:, 0, :, :] + 0.59 * img[:, 1, :, :] + 0.11 * img[:, 2, :, :]

        return gray_img.reshape(gray_img.shape[0], 1, gray_img.shape[1], gray_img.shape[2])

# Sample a random latent space vector
def sample_z(shape=64, cluster_dim=1, shared_dim=10, num_class=10, fix_class=-1, req_grad=False, device='cpu'):

    assert (fix_class == -1 or (fix_class >= 0 and fix_class < num_class) ), "Requested class %i outside bounds."%fix_class

    ######### zc, zc_idx variables with grads, and zc to one-hot vector
    # Pure one-hot vector generation
    zc_FT = torch.Tensor(shape, num_class).fill_(0).to(device)
    zc_idx = torch.empty(shape, dtype=torch.long).to(device)

    if (fix_class == -1):
        zc_idx = zc_idx.random_(num_class)
        zc_FT = zc_FT.scatter_(1, zc_idx.unsqueeze(1), 1.)
    else:
        zc_idx[:] = fix_class
        zc_FT[:, fix_class] = 1

    zc = Variable(zc_FT, requires_grad=req_grad).to(device)

    # Sample index specific latent variable
    zy_np = np.zeros((zc.shape[0], zc.shape[1] * cluster_dim))
    for i in range(0, zc_idx.shape[0]):
        for j in range(0, cluster_dim):
            zy_np[i, zc_idx[i] * cluster_dim + j] = 0.75*np.random.normal(0, 1)

    zy = Variable(torch.FloatTensor(zy_np), requires_grad=req_grad).to(device)

    # Sample noise as generator input, zn
    zs = Variable(torch.FloatTensor(0.75*np.random.normal(0, 1, (shape, shared_dim))), requires_grad=req_grad).to(device)

    # Return components of latent space variable
    return zc, zc_idx, zy, zs


# latent space traversal
def gen_label_mask(targets, latent_dim, shared_dim, num_class, device):
    masks = []
    if isinstance(targets, int) is True:
        target = targets
        mask = torch.zeros([num_class + latent_dim * num_class + shared_dim], dtype=torch.float32, device=device)
        mask[target] = 1.0
        mask[num_class + latent_dim * target: num_class + latent_dim * (target + 1)] = 1
        mask[num_class + latent_dim * num_class: ] = 1
        masks.append(mask)
    else:
        for target in targets:
            mask = torch.zeros([num_class + latent_dim * num_class + shared_dim], dtype=torch.float32, device=device)
            mask[target] = 1.0
            mask[num_class + latent_dim * target: num_class + latent_dim * (target + 1)] = 1
            mask[num_class + latent_dim * num_class: ] = 1
            masks.append(mask)

    return torch.stack(masks)


def idx2onehot(idx, n):
    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)

    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx, 1)

    return onehot

# TODO: refactor the gen_mask code with idx2onehot
def idx2onehot(idx, n, dim):
    if type(idx) is torch.Tensor:
        idx = torch.tensor(idx)

    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)

    onehot = torch.zeros(idx.size(0), n * dim)
    for i in range(0, dim):
      onehot.scatter_(1, idx * dim + i, 1)

    return onehot

def calc_gradient_penalty(netD, real_data, generated_data, device):
    # GP strength
    LAMBDA = 10

    b_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(b_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.to(device)

    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.to(device)

    # Calculate probability of interpolated examples
    prob_interpolated = netD(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(b_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return LAMBDA * ((gradients_norm - 1) ** 2).mean()


def calc_gradient_penalty_joint(netD, real_data, generated_data, real_z, gen_z, device):
    # GP strength
    LAMBDA = 10

    b_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(b_size, 1, 1, 1)

    alpha = alpha.expand_as(real_data)
    alpha = alpha.to(device)
    interpolated_data = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated_data = Variable(interpolated_data, requires_grad=True)
    interpolated_data = interpolated_data.to(device)

    alpha = alpha.expand_as(real_data)
    alpha = alpha.to(device)
    interpolated_z = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated_data = Variable(interpolated_data, requires_grad=True)
    interpolated_data = interpolated_data.to(device)

    # Calculate probability of interpolated examples
    prob_interpolated = netD(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(b_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return LAMBDA * ((gradients_norm - 1) ** 2).mean()


# Weight Initializer
def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


class Color(object):
    def __init__(self, num_channels, effective_channels):
        super(Color, self).__init__()

        self.num_channels = num_channels
        self.effective_channels = effective_channels

    def __call__(self, img):

        colors = random.randint(0, self.effective_channels - 1)
        # Apply the color to the image by zeroing out the other color channel
        img = torch.cat([img] * self.num_channels, dim=0)

        for i in range(0, self.num_channels):
            if i != colors:
                img[i, :, :] *= 0.0
            else:
                img[i, :, :] *= random.uniform(0.7, 1.0)

        # for i in range(0, self.num_channels):
        #     img[i, :, :] *= random.uniform(0.7, 1.0)

        # for i in range(0, self.num_channels):
        #     img[i, :, :] *= random.randint(0, 10) * 0.1

        return img

# cosine_flag = torch.tensor(1).to(device)

def cosine_loss(x, y):
    loss = cosine_loss_fn(x, y, cosine_flag)

    return loss
