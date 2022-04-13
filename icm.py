import argparse
import os
import numpy as np
import datetime
import random
import h5py

from torch.autograd import Variable
from torch.autograd import grad as torch_grad

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from itertools import chain as ichain

from model import Generator_CNN, Encoder_CNN, Discriminator_CNN
from utils import Color, sample_z, calc_gradient_penalty, gray_scale
from visualization import latent_space_traversal

parser = argparse.ArgumentParser(description="ClusterGAN Training Script")
parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=200, type=int, help="Number of epochs")
parser.add_argument("-b", "--batch_size", dest="batch_size", default=64, type=int, help="Batch size")
parser.add_argument("-i", "--input_dim", dest="input_dim", type=int, default=28, help="Size of image dimension")
parser.add_argument("--cluster_dim", dest="cluster_dim", default=1, type=int, help="Dimension of cluster latent space")
parser.add_argument("--shared_dim", dest="shared_dim", default=5, type=int, help="Dimension of shared latent space")
parser.add_argument("-l", "--lr", dest="learning_rate", type=float, default=0.0001, help="Learning rate")
parser.add_argument("-c", "--num_classritic", dest="num_classritic", type=int, default=5, help="Number of training steps for discriminator per iter")
parser.add_argument("-w", "--wass_flag", dest="wass_flag", action='store_true', help="Flag for Wasserstein metric")
parser.add_argument('--cuda', type=int, default=0, metavar='#',
                    help='set CUDA device number')
parser.add_argument('--store-path', default='./checkpoints', type=str, metavar='PATH',
                    help='path to store checkpoint (default: none)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--checkpoint-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument("--beta_c", dest="beta_c", type=float, default=1.0, help="beta_c")
parser.add_argument("--beta_y", dest="beta_y", type=float, default=1.0, help="beta_y")
parser.add_argument("--beta_s", dest="beta_s", type=float, default=1.0, help="beta_s")
parser.add_argument("--beta_cycle", dest="beta_cycle", type=float, default=0.05, help="beta_cycle")
parser.add_argument("--num_class", dest="num_class", type=int, default=10, help="num_class")
parser.add_argument("--num_channels", dest="num_channels", type=int, default=1, help="number of channels")
parser.add_argument("--shift_threshold", dest="shift_threshold", type=float, default=0.0, help="shift_threshold")
parser.add_argument("--split_dim", dest="split_dim", type=int, default=0, help="split_dim")
parser.add_argument("--dataset", dest="dataset", type=str, default='mnist', help="dataset")
parser.add_argument("-ha", "--his_avg", dest="his_avg", action='store_true', help="Flag for historical averaging")
parser.add_argument("-ca", "--color_avg", dest="color_avg", action='store_true', help="Flag for color averaging")

args = parser.parse_args()
print('arge: ', args)

time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

store_path = args.store_path + '_' + time
os.mkdir(store_path)
print('store_path: ', store_path)

file = open('./' + store_path + '/args.txt','w')
file.write(str(args))
file.close()

device = torch.device("cuda:%d" % args.cuda) if torch.cuda.is_available() else torch.device("cpu")
print('device: ', device)

writer = SummaryWriter(store_path + '/mnist_summary')

# Training details
n_epochs = args.n_epochs
batch_size = args.batch_size
test_batch_size = 5000
lr = args.learning_rate


b1 = 0.5
b2 = 0.9
decay = 2.5*1e-5

n_skip_iter = args.num_classritic

# Data dimensions
input_dim = args.input_dim
num_channels = args.num_channels

# Latent space info
cluster_dim = args.cluster_dim
shared_dim = args.shared_dim
num_class = args.num_class
beta_c = args.beta_c
beta_y = args.beta_y
beta_s = args.beta_s
beta_cycle = args.beta_cycle

# Wasserstein+GP metric flag
wass_metric = args.wass_flag

data_shape = (num_channels, input_dim, input_dim)

# Loss function
bce_loss = torch.nn.BCELoss()
xe_loss = torch.nn.CrossEntropyLoss()
mse_loss = torch.nn.MSELoss()
cosine_loss_fn = torch.nn.CosineEmbeddingLoss()

bce_loss.to(device)
xe_loss.to(device)
mse_loss.to(device)
cosine_loss_fn.to(device)

# Initialize generator and discriminator
generator = Generator_CNN(input_dim, cluster_dim, shared_dim, num_class, data_shape, num_channels)
encoder = Encoder_CNN(input_dim, cluster_dim, shared_dim, num_class, num_channels)
encoder_gray = Encoder_CNN(input_dim, cluster_dim, shared_dim, num_class, 1)
discriminator = Discriminator_CNN(input_dim, num_channels, wass_metric=wass_metric)

generator.to(device)
encoder.to(device)
encoder_gray.to(device)
discriminator.to(device)

# Add options for dataset
mnist_transform = transforms.Compose([transforms.ToTensor(),
                                      ])

cifar10_transform=transforms.Compose([transforms.ToTensor(),
                                      ])

print('args.dataset:\n', args.dataset)
if args.dataset == 'mnist':
    train_dataset = datasets.MNIST("../data/", train=True, download=True, transform=mnist_transform)
    test_dataset = datasets.MNIST("../data/", train=False, download=True, transform=mnist_transform)
elif args.dataset == 'fashion_mnist':
    train_dataset = datasets.FashionMNIST("../data/", train=True, download=True, transform=mnist_transform)
    test_dataset = datasets.FashionMNIST("../data/", train=False, download=True, transform=mnist_transform)
else:
    print('Unknown dataset.')
    exit()

# Configure data loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

# Test data loader
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

cluster_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4096, shuffle=True, num_workers=16)

# Optimizer
ge_chain = ichain(generator.parameters(),
                  encoder.parameters(),
                  encoder_gray.parameters())

# e_chain = ichain(encoder.parameters(),
#                  encoder_gray.parameters())

optimizer_GE = torch.optim.Adam(ge_chain, lr=lr, betas=(b1, b2), weight_decay=decay)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
# optimizer_E = torch.optim.Adam(e_chain, lr=lr, betas=(b1, b2))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2), weight_decay=decay)

# Image pool for historical averaging
image_pool = ImagePool(pool_size=n_skip_iter*50*batch_size)

# ----------
#  Training
# ----------
# Training loop
print('\nBegin training session with %i epochs...\n'%(n_epochs))

torch.autograd.set_detect_anomaly(True)

for epoch in range(n_epochs):
    train_loss_zc = 0.0
    train_loss_zy = 0.0
    train_loss_zs = 0.0
    train_loss_cycle = 0.0

    train_loss_g = 0.0
    train_loss_e = 0.0
    train_loss_d = 0.0

    correct_real = 0
    correct_fake = 0

    for i, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)

        # ---------------------------
        #  Train Generator + Encoder
        # ---------------------------

        # Step for Generator & Encoder, n_skip_iter times less than for discriminator
        if (i % n_skip_iter == 0):
            # config models
            generator.train()
            encoder.train()
            encoder_gray.train()
            discriminator.eval()

            # clear grad

            optimizer_GE.zero_grad()
            '''
            optimizer_G.zero_grad()
            '''
            # Sample random latent variables
            zc, zc_idx, zy, zs = sample_z(shape=data.shape[0],
                                          cluster_dim=cluster_dim,
                                          shared_dim=shared_dim,
                                          num_class=num_class,
                                          device=device)

            # Generate a batch of images
            gen_data = generator(zc, zy, zs)
            # Discriminator output from real and generated samples
            D_gen = discriminator(gen_data)

            # Encode the generated images
            enc_gen_zc, enc_gen_zc_logits, enc_gen_zy, enc_gen_zs = encoder(gen_data)
            enc_gray_gen_zc, enc_gray_gen_zc_logits, enc_gray_gen_zy, enc_gray_gen_zs = encoder_gray(gray_scale(gen_data))

            # Calculate losses for z_n, z_c
            # estimiting the likelikood of p(x|zn) and p(x|zc) using an encoder
            if args.color_avg:
                zc_loss = xe_loss(enc_gray_gen_zc_logits, zc_idx)
                zy_loss = mse_loss(enc_gray_gen_zy, zy)
            else:
                zc_loss = xe_loss(enc_gen_zc_logits, zc_idx)
                zy_loss = mse_loss(enc_gen_zy, zy)

            zs_loss = mse_loss(enc_gen_zs, zs)

            train_loss_zc += zc_loss.item()
            train_loss_zy += zy_loss.item()
            train_loss_zs += zs_loss.item()


            # Compute the loss according to Equation () in the paper
            if wass_metric:
                # Wasserstein GAN loss
                g_loss = torch.mean(D_gen)
                train_loss_g += g_loss.item()

                ge_loss = g_loss + beta_c * zc_loss + beta_y * zy_loss + beta_s * zs_loss
                train_loss_e += beta_c * zc_loss.item() + beta_y * zy_loss.item() + beta_s * zs_loss.item()
            else:
                # Vanilla GAN loss
                valid = Variable(torch.FloatTensor(gen_data.size(0), 1).fill_(1.0), requires_grad=False).to(device)
                g_loss = bce_loss(D_gen, valid)
                train_loss_g += g_loss.item()

                ge_loss = g_loss + beta_c * zc_loss + beta_y * zy_loss + beta_s * zs_loss
                train_loss_e += beta_c * zc_loss.item() + beta_y * zy_loss.item() + beta_s * zs_loss.item()


            ge_loss.backward(retain_graph=True)
            optimizer_GE.step()


            # Cycle through test real -> enc -> gen for consistancy
            optimizer_GE.zero_grad()
            # Encode sample real instances
            e_tzc, e_tzc_logits, e_tzy, e_tzs = encoder(data)
            e_gray_tzc, e_gray_tzc_logits, e_gray_tzy, e_gray_tzs = encoder_gray(gray_scale(data))
            # Generate sample instances from encoding
            gen_data = None
            if args.color_avg:
                gen_data = generator(e_gray_tzc, e_gray_tzy, e_tzs)
            else:
                gen_data = generator(e_tzc, e_tzy, e_tzs)

            # Compute cycle reconstruction loss
            cycle_loss = bce_loss(gen_data, data)
            train_loss_cycle += cycle_loss.item()
            img_bce_loss = beta_cycle * cycle_loss

            img_bce_loss.backward(retain_graph=True)
            optimizer_GE.step()

        # ---------------------
        #  Train Discriminator and encoder
        # ---------------------

        # config model
        generator.eval()
        encoder.eval()
        encoder_gray.eval()
        discriminator.train()

        # clear grad
        optimizer_D.zero_grad()

        # Fetch real input
        real_data = Variable(data.type(torch.FloatTensor)).to(device)

        # Sample random latent variables
        zc, zc_idx, zy, zs = sample_z(shape=data.shape[0],
                                      cluster_dim=cluster_dim,
                                      shared_dim=shared_dim,
                                      num_class=num_class,
                                      device=device)

        # Generate a batch of images
        # Note: We need to do a forward pass again to avoid gradient issue
        gen_data = generator(zc, zy, zs)

        if args.his_avg is True:
            gen_data = image_pool.query(gen_data.to('cpu')).to(device)

        # Discriminator output from real and generated samples
        D_real = discriminator(real_data)
        D_gen = discriminator(gen_data)

        # Measure discriminator's ability to classify real from generated samples
        if wass_metric:
            # Gradient penalty term
            grad_penalty = calc_gradient_penalty(discriminator, real_data, gen_data, device)
            # Wasserstein GAN loss w/gradient penalty
            d_loss = torch.mean(D_real) - torch.mean(D_gen) + grad_penalty

            train_loss_d += d_loss.item()
        else:
            # Vanilla GAN loss
            fake = Variable(torch.FloatTensor(gen_data.size(0), 1).fill_(0.0), requires_grad=False).to(device)
            valid = Variable(torch.FloatTensor(gen_data.size(0), 1).fill_(1.0), requires_grad=False).to(device)

            pred = D_real.argmax(dim=1, keepdim=True)
            correct_real += pred.eq(valid.view_as(pred)).sum().item()

            pred = D_gen.argmax(dim=1, keepdim=True)
            correct_fake += pred.eq(fake.view_as(pred)).sum().item()

            real_loss = bce_loss(D_real, valid)
            fake_loss = bce_loss(D_gen, fake)
            d_loss = (real_loss + fake_loss) / 2

            train_loss_d += d_loss.item()

        # just a normal GAN loss
        d_loss.backward()
        optimizer_D.step()


    writer.add_scalar('Training loss zc', train_loss_zc, epoch)
    writer.add_scalar('Training loss zy', train_loss_zy, epoch)
    writer.add_scalar('Training loss zs', train_loss_zs, epoch)
    writer.add_scalar('Training loss cycle', train_loss_cycle, epoch)

    writer.add_scalar('Training loss g', train_loss_g, epoch)
    writer.add_scalar('Training loss e', train_loss_e, epoch)
    writer.add_scalar('Training loss d', train_loss_d, epoch)

    print ("[Epoch %d/%d] \n"\
           "\tTraining losses: [G: %f] [E: %f] [D: %f] [zc: %f] [zy: %f] [zs: %f] [cycle: %f]" % (epoch, n_epochs,
                                                 train_loss_g, train_loss_e, train_loss_d,
                                                 train_loss_zc, train_loss_zy, train_loss_zs, train_loss_cycle)
          )

    # Generator in eval mode
    generator.eval()
    encoder.eval()
    encoder_gray.eval()

    # Save checkpoint
    if (epoch + 1) % (args.checkpoint_interval) == 0:
        generator.eval()
        encoder.eval()
        encoder_gray.eval()
        discriminator.eval()
        torch.save(generator.state_dict(), store_path + '/epoch_%d_generator.tar' % (epoch))
        torch.save(discriminator.state_dict(), store_path + '/epoch_%d_discriminator.tar' % (epoch))
        torch.save(encoder.state_dict(), store_path + '/epoch_%d_encoder.tar' % (epoch))
        torch.save(encoder_gray.state_dict(), store_path + '/epoch_%d_encoder_gray.tar' % (epoch))

        latent_space_traversal(generator, epoch, num_class, cluster_dim, shared_dim, num_channels, input_dim, store_path, device)
