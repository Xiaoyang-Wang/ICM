import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

import numpy as np

from scipy.stats import norm

from utils import idx2onehot, gen_label_mask, sample_z, gray_scale

import matplotlib.pyplot as plt

def latent_space_traversal(generator, epoch, num_class, cluster_dim, shared_dim, num_channels, data_size, store_path, device):
    # Display a 2D manifold of the digits
    n = 20  # figure with 20x20 digits
    figure = np.zeros((num_channels, data_size * num_class * cluster_dim, data_size * n))

    grid_x = norm.ppf(np.linspace(0.01, 0.99, n))
    grid_y = norm.ppf(np.linspace(0.01, 0.99, n))

    for target in range (0, num_class):
        for dim in range(0, cluster_dim):
            for i, xi in enumerate(grid_x):
                z_c = torch.zeros(1, num_class).to(device)
                z_c[0, target] = 1.0
                z_y = torch.zeros(1, num_class * cluster_dim).to(device)
                z_y[0, target * cluster_dim + dim] = xi
                z_s = torch.zeros(1, shared_dim).to(device)

                with torch.no_grad():
                    x_decoded = generator(z_c, z_y, z_s)
                    x_decoded = x_decoded.to('cpu')

                digit = x_decoded[0].reshape(num_channels, data_size, data_size)
                row = target * cluster_dim + dim
                figure[:, row * data_size: (row + 1) * data_size,
                       i * data_size: (i + 1) * data_size] = digit

    if num_channels > 1:
        figure_list = []
        for i in range(0, num_channels):
            figure_list.append(figure[i])
        figure = np.stack(figure_list, axis=2)
    else:
        figure = figure[0]

    plt.figure(figsize=(num_class, num_class))
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(store_path + '/class_traversal_epoch_%06i.png' % (epoch))
    plt.close()

    figure = np.zeros((num_channels, data_size * num_class * cluster_dim, data_size * n))

    for dim in range (0, shared_dim):
        # figure = np.zeros((num_channels, data_size * num_class * cluster_dim, data_size * n))
        for target in range (0, num_class):
            for i, xi in enumerate(grid_x):
                z_c = torch.zeros(1, num_class).to(device)
                z_c[0, target] = 1.0
                z_y = torch.zeros(1, num_class * cluster_dim).to(device)
                z_s = torch.zeros(1, shared_dim).to(device)
                z_s[0, dim] = xi

                with torch.no_grad():
                    x_decoded = generator(z_c, z_y, z_s)
                    x_decoded = x_decoded.to('cpu')

                digit = x_decoded[0].reshape(num_channels, data_size, data_size)
                figure[:, target * data_size: (target + 1) * data_size,
                       i * data_size: (i + 1) * data_size] = digit

        if num_channels > 1:
            figure_list = []
            for i in range(0, num_channels):
                figure_list.append(figure[i])
            figure = np.stack(figure_list, axis=2)
        else:
            figure = figure[0]

        # plt.figure(figsize=(num_class, num_class))
        plt.figure()
        plt.imshow(figure, cmap='Greys_r')
        plt.savefig(store_path + '/shared_traversal_dim_%02i_epoch_%06i.png' % (dim, epoch))
        plt.close()

        figure = np.zeros((num_channels, data_size * num_class, data_size * n))

def latent_space_cluster(encoder, encoder_gray, cluster_loader, epoch, num_class, cluster_dim, shared_dim, store_path, device):
    # Translate into the latent space
    for batch_idx, (data, targets) in enumerate(cluster_loader):
        targets_cpu = targets.to('cpu')
        data , targets = data.to(device), targets.to(device)

        for dim in range(0, num_class * cluster_dim):
            with torch.no_grad():
                # one_hot_target = idx2onehot(targets_cpu, num_class).to(device)

                _, enc_gen_zc_logits, enc_gen_zy, enc_gen_zs = encoder(data.to(device))
                # enc_gray_gen_zc, _, enc_gray_gen_zy, _ = encoder_gray(torch.mean(data.to(device), 1, keepdim=True))
                enc_gray_gen_zc, _, enc_gray_gen_zy, _ = encoder_gray(gray_scale(data.to(device)))

                # mu = enc_gen_zy[:, dim:dim+1]
                mu = enc_gray_gen_zy[:, dim:dim+1]

                mu = mu.to('cpu')
                # plt.figure(figsize=(num_class, num_class))
                plt.figure()
                plt.scatter(targets_cpu, mu[:, 0], c=targets_cpu, cmap='brg')
                plt.colorbar()
                plt.savefig(store_path + '/class_cluster_dim_%02i_epoch_%06i.png' % (dim, epoch))
                plt.close()


        break

    # Translate into the latent space
    for batch_idx, (data, targets) in enumerate(cluster_loader):
        with torch.no_grad():
            # one_hot_target = idx2onehot(targets_cpu, num_class)

            data, targets = data.to(device), targets.to(device)

            _, enc_gen_zc_logits, enc_gen_zy, enc_gen_zs = encoder(data.to(device))
            # enc_gray_gen_zc, _, _, _ = encoder_gray(torch.mean(data.to(device), 1, keepdim=True))
            enc_gray_gen_zc, _, _, _ = encoder_gray(gray_scale(data.to(device)))

            mu = enc_gen_zs[:, 0:2]

            mu = mu.to('cpu')
            # plt.figure(figsize=(num_class, num_class))
            plt.figure()
            plt.scatter(mu[:, 0], mu[:, 1], c=targets.cpu(), cmap='brg')
            plt.colorbar()
            plt.savefig(store_path + '/shared_cluster_dim_%d%d_epoch_%06i.png' % (0,1,epoch))
            plt.close()
            mu = enc_gen_zs[:, 2:4]

            mu = mu.to('cpu')
            # plt.figure(figsize=(num_class, num_class))
            plt.figure()
            plt.scatter(mu[:, 0], mu[:, 1], c=targets.cpu(), cmap='brg')
            plt.colorbar()
            plt.savefig(store_path + '/shared_cluster_dim_%d%d_epoch_%06i.png' % (2,3,epoch))
            plt.close()

        break

def cycle_recon(encoder, encoder_gray, generator, cluster_loader, epoch, store_path, device):
    # Set number of examples for cycle calcs
    n_sqrt_samp = 8
    n_samp = n_sqrt_samp * n_sqrt_samp

    for batch_idx, (data, targets) in enumerate(cluster_loader):
        with torch.no_grad():
            # Save cycled and generated examples!
            data, targets = data.to(device), targets.to(device)
            data, targets = data[:n_samp], targets[:n_samp]
            _, e_zc_logits, e_zy, e_zs = encoder(data)
            # e_gray_zc, _, e_gray_zy, _ = encoder_gray(torch.mean(data, 1, keepdim=True))
            e_gray_zc, _, e_gray_zy, _ = encoder_gray(gray_scale(data))

            recon_batch = generator(e_gray_zc, e_gray_zy, e_zs)
            save_image(recon_batch[:n_samp],
                       store_path + '/cycle_reg_%06i.png' %(epoch),
                       nrow=n_sqrt_samp, normalize=True)
            save_image(data[:n_samp],
                       store_path + '/cycle_reg_origin_%06i.png' %(epoch),
                       nrow=n_sqrt_samp, normalize=True)

def gen_class(generator, cluster_loader, epoch, num_class, cluster_dim, shared_dim, store_path, device):
    ## Generate samples for specified classes
    stack_imgs = []
    for idx in range(num_class):
        # Sample specific class
        zc_samp, zc_samp_idx, zy_samp, zs_samp = sample_z(shape=32,
                                                          cluster_dim=cluster_dim,
                                                          shared_dim=shared_dim,
                                                          num_class=num_class,
                                                          fix_class=idx,
                                                          req_grad=False,
                                                          device=device)

        # Generate sample instances
        gen_imgs_samp = generator(zc_samp, zy_samp, zs_samp)

        if (len(stack_imgs) == 0):
            stack_imgs = gen_imgs_samp
        else:
            stack_imgs = torch.cat((stack_imgs, gen_imgs_samp), 0)

    # Save class-specified generated examples!
    save_image(stack_imgs,
               store_path + '/gen_class_%06i.png' %(epoch),
               nrow=32, normalize=True)
