import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils import initialize_weights

# Softmax function
def softmax(x):
    return F.softmax(x, dim=1)


class Reshape(nn.Module):
    """
    Class for performing a reshape as a layer in a sequential model.
    """
    def __init__(self, shape=[]):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)

    def extra_repr(self):
            # (Optional)Set the extra information about this module. You can test
            # it by printing an object of this class.
            return 'shape={}'.format(
                self.shape
            )

class VAE(nn.Module):
    def __init__(self, z_dim=10, nc=3):
        super(VAE, self).__init__()

        self.channels = nc
        self.z_dim = z_dim

        # 28x28
        self.cshape = (128, 5, 5)
        # self.cshape = (128, 8, 8)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)

        self.ishape = (128, 7, 7)
        # self.ishape = (128, 8, 8)
        self.iels_d = int(np.prod(self.ishape))

        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels, 64, 4, stride=2, bias=True),
            # nn.Conv2d(self.channels, 64, 4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, bias=True),
            # nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            Reshape(self.lshape),

            nn.Linear(self.iels, 1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, self.z_dim*2),             # B, z_dim*2
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 1024),               # B, 256
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, self.iels_d),
            nn.BatchNorm1d(self.iels_d),
            nn.LeakyReLU(0.2, inplace=True),
            Reshape(self.ishape),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, self.channels, 4, stride=2, padding=1, bias=True)
        )

    def encode(self, x):
        z = self.encoder(x)
        return z[:, :self.z_dim], z[:, self.z_dim:]

    def decode(self, z):
        return torch.sigmoid(self.decoder(z))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class Generator_CNN(nn.Module):
    """
    CNN to model the generator of a ClusterGAN
    Input is a vector from representation space of dimension z_dim
    output is a vector from image space of dimension X_dim
    """
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim, cluster_dim, shared_dim, num_class, x_shape, num_channels, verbose=False):
        super(Generator_CNN, self).__init__()

        self.name = 'generator'
        self.channels = num_channels
        self.input_dim = input_dim
        self.cluster_dim = cluster_dim
        self.shared_dim = shared_dim
        self.num_class = num_class
        self.x_shape = x_shape
        if self.input_dim is 28:
            self.ishape = (128, 7, 7) # 28x28
        elif self.input_dim is 32 or 64:
            self.ishape = (128, 8, 8) # 32x32 or 64x64
        else:
            print('Unsupported input dimension.')

        self.iels = int(np.prod(self.ishape))
        self.verbose = verbose

        layers = [
            # Fully connected layers
            torch.nn.Linear(self.num_class + self.num_class * self.cluster_dim + self.shared_dim, 1024),
            # torch.nn.Linear(self.shared_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(1024, self.iels),
            nn.BatchNorm1d(self.iels),
            nn.LeakyReLU(0.2, inplace=True),

            # Reshape to 128 x (7x7) or 128 x (8x8)
            Reshape(self.ishape),

            # Upconvolution layers
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        if self.input_dim is 64:
            layers.append(nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1, bias=True))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.ConvTranspose2d(64, self.channels, 4, stride=2, padding=1, bias=True))
        layers.append(nn.Sigmoid())

        self.model = nn.ModuleList(layers)

        initialize_weights(self)

        if self.verbose:
            print("Setting up {}...\n".format(self.name))
            print(self.model)

    def forward(self, zc, zy, zs):
        x = torch.cat((zc, zy, zs), 1)
        # z = zs

        # x = z
        for i, l in enumerate(self.model):
            x = self.model[i](x)

        # x_gen = self.model(z)
        # Reshape for output
        x_gen = x
        x_gen = x_gen.view(x_gen.size(0), *self.x_shape)
        return x_gen


class Encoder_CNN(nn.Module):
    """
    CNN to model the encoder of a ClusterGAN
    Input is vector X from image space if dimension X_dim
    Output is vector z from representation space of dimension z_dim
    """
    def __init__(self, input_dim, cluster_dim, shared_dim, num_class, num_channels, verbose=False):
        super(Encoder_CNN, self).__init__()

        self.name = 'encoder'
        self.channels = num_channels
        self.input_dim = input_dim
        self.cluster_dim = cluster_dim
        self.shared_dim = shared_dim
        self.num_class = num_class

        if self.input_dim is 28:
            self.cshape = (128, 5, 5) # 28x28
        elif self.input_dim is 32 or 64:
            self.cshape = (128, 8, 8) # 32x32 or 64x64
        else:
            print('Unsupported input dimension.')

        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.verbose = verbose

        layers = []
        if self.input_dim is 28:
            layers.append(nn.Conv2d(self.channels, 64, 4, stride=2, bias=True))
        elif self.input_dim is 32 or 64:
            layers.append(nn.Conv2d(self.channels, 64, 4, stride=2, padding=1, bias=True))

        layers.append(nn.LeakyReLU(0.2, inplace=True))

        if self.input_dim is 28:
            layers.append(nn.Conv2d(64, 128, 4, stride=2, bias=True))
        elif self.input_dim is 32:
            layers.append(nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True))
        elif self.input_dim is 64:
            layers.append(nn.Conv2d(64, 64, 4, stride=2, padding=1, bias=True))

        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if self.input_dim is 64:
            layers.append(nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(Reshape(self.lshape))
        layers.append(torch.nn.Linear(self.iels, 1024))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(torch.nn.Linear(1024, self.num_class + self.num_class * self.cluster_dim + self.shared_dim))

        self.model = nn.ModuleList(layers)

        initialize_weights(self)

        if self.verbose:
            print("Setting up {}...\n".format(self.name))
            print(self.model)

    def forward(self, in_feat):
        x = in_feat
        for i, l in enumerate(self.model):
            x = self.model[i](x)

        # z_img = self.model(in_feat)
        # Reshape for output
        z_img = x
        z = z_img.view(z_img.shape[0], -1)
        # Separate continuous and one-hot components
        zc_logits = z[:, 0:self.num_class]
        zc = softmax(zc_logits)
        zy = z[:, self.num_class:(self.cluster_dim + 1)*self.num_class]
        zs = z[:, (self.cluster_dim + 1)*self.num_class:]
        # Softmax on zc component
        return zc, zc_logits, zy, zs


class Discriminator_CNN(nn.Module):
    """
    CNN to model the discriminator of a ClusterGAN
    Input is tuple (X,z) of an image vector and its corresponding
    representation z vector. For example, if X comes from the dataset, corresponding
    z is Encoder(X), and if z is sampled from representation space, X is Generator(z)
    Output is a 1-dimensional value
    """
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim, num_channels, wass_metric=False, verbose=False):
        super(Discriminator_CNN, self).__init__()

        self.name = 'discriminator'
        self.channels = num_channels
        self.input_dim = input_dim
        if self.input_dim is 28:
            self.cshape = (128, 5, 5) # 28x28
        elif self.input_dim is 32 or 64:
            self.cshape = (128, 8, 8) # 64x64

        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.wass = wass_metric
        self.verbose = verbose

        layers = []
        if self.input_dim is 28:
            layers.append(nn.Conv2d(self.channels, 64, 4, stride=2, bias=True),)
        elif self.input_dim is 32 or 64:
            layers.append(nn.Conv2d(self.channels, 64, 4, stride=2, padding=1, bias=True))

        layers.append(nn.LeakyReLU(0.2, inplace=True))

        if self.input_dim is 28:
            layers.append(nn.Conv2d(64, 128, 4, stride=2, bias=True))
        elif self.input_dim is 32:
            layers.append(nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True))
        elif self.input_dim is 64:
            layers.append(nn.Conv2d(64, 64, 4, stride=2, padding=1, bias=True))

        layers.append(nn.LeakyReLU(0.2, inplace=True))

        if self.input_dim is 64:
            layers.append(nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(Reshape(self.lshape))
        layers.append(torch.nn.Linear(self.iels, 1024))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(torch.nn.Linear(1024, 1))

        # If NOT using Wasserstein metric, final Sigmoid
        if (not self.wass):
            layers.append(torch.nn.Sigmoid())
            # self.model = nn.Sequential(self.model, torch.nn.Sigmoid())

        self.model = nn.ModuleList(layers)

        initialize_weights(self)

        if self.verbose:
            print("Setting up {}...\n".format(self.name))
            print(self.model)

    def forward(self, img):
        # Get output
        x = img
        for i, l in enumerate(self.model):
            x = self.model[i](x)

        validity = x
        # validity = self.model(img)
        return validity
