import numpy as np
import argparse
import time
import torch
from torch.autograd import Variable

from torchvision.utils import save_image
import matplotlib.pyplot as plt
from models.Hnet_base6 import HNet
from models.Discriminitor_base import Discriminitor
# from DataView.DataView import *
from torch.optim import RMSprop
from torch.utils.data import DataLoader
from utils.loss import *
from utils.util import *

n_critic=5
sample_interval=500
latent_dim=100
# Loss weight for gradient penalty
lambda_gp = 12
noise_weight=0.05

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def train(G:HNet,D:Discriminitor,dataloader:DataLoader,epoches:int,lr_G=0.01,lr_D=0.01,b1=0.5,b2=0.999):
    if cuda:
        G.cuda()
        D.cuda()
    # Optimizers
    optimizer_G = torch.optim.Adam(G.parameters(), lr=lr_G, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=lr_D, betas=(b1, b2))
    # optimizer_G = RMSprop(G.parameters(), lr=lr_G)
    # optimizer_D = RMSprop(D.parameters(), lr=lr_D)

    gloss_list = []
    dloss_list = []
    wd=[]

    # ----------
    #  Training
    # ----------

    batches_done = 0
    for epoch in range(epoches):
        for i, (imgs, _) in enumerate(dataloader):

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.uniform(0, 1, (imgs.shape[0], latent_dim))))


            # Generate a batch of images
            fake_imgs = G(z)
            # noise = Variable(Tensor(np.random.normal(0, 1,fake_imgs.shape )))
            # fake_imgs=(1-noise_weight)*fake_imgs+noise_weight*noise

            # Real images
            real_validity = D(real_imgs)
            # Fake images
            fake_validity = D(fake_imgs)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(D, real_imgs.data, fake_imgs.data)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_imgs = G(z)
                # noise = Variable(Tensor(np.random.normal(0, 1, fake_imgs.shape)))
                # fake_imgs=(1-noise_weight)*fake_imgs+noise_weight*noise
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = D(fake_imgs)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()

                gloss_list.append(g_loss.detach().item())
                dloss_list.append(d_loss.detach().item())
                wassertain_distance = torch.mean(real_validity) - torch.mean(fake_validity)
                wd.append(wassertain_distance.detach().item())

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Wassertain distance: %f]"
                    % (epoch, epoches, i, len(dataloader), d_loss.item(), g_loss.item(), wassertain_distance.item())
                )

                if batches_done % sample_interval == 0:
                    save_image(fake_imgs.data[:16], "images/%d.jpeg" % batches_done, nrow=4, normalize=True)

                batches_done += n_critic

        if epoch%1==0:
            x = range(0, len(gloss_list))
            plt.subplot(3, 1, 1)
            plt.ylabel("g_loss")
            plt.plot(x, gloss_list)
            plt.subplot(3, 1, 2)
            plt.ylabel("d_loss")
            plt.plot(x, dloss_list)
            plt.subplot(3, 1, 3)
            plt.ylabel("wassertain_distance")
            plt.plot(x, wd)
            plt.show()
            plt.savefig("./LossCurve/loss" + str(epoch/10) + ".png")
            # gloss_list.clear()
            # dloss_list.clear()

            torch.save(G, './ModelSaved/WGAN_gp/G_gpmodel_epoch' + str(epoch) + '.pkl')
            torch.save(D, './ModelSaved/WGAN_gp/D_gpmodel_epoch' + str(epoch) + '.pkl')