"""Independent 256×256 convolutional VAE for educational experiments.

This file is a fresh portfolio implementation: no instructional TODO text,
notebook metadata, proprietary images or source dataset are redistributed.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ConvolutionalVAE(nn.Module):
    def __init__(self,latent_dim=128):
        super().__init__()
        if latent_dim<1:
            raise ValueError("latent_dim must be positive")
        self.latent_dim=latent_dim
        self.encoder=nn.Sequential(
            nn.Conv2d(3,32,4,stride=2,padding=1),nn.ReLU(),
            nn.Conv2d(32,64,4,stride=2,padding=1),nn.ReLU(),
            nn.Conv2d(64,128,4,stride=2,padding=1),nn.ReLU(),
            nn.Flatten()
        )
        n=128*32*32
        self.mu=nn.Linear(n,latent_dim)
        self.logvar=nn.Linear(n,latent_dim)
        self.decode_latent=nn.Linear(latent_dim,n)
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(128,64,4,stride=2,padding=1),nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,stride=2,padding=1),nn.ReLU(),
            nn.ConvTranspose2d(32,3,4,stride=2,padding=1),nn.Sigmoid()
        )

    def decode(self,z):
        h=self.decode_latent(z).relu().reshape(-1,128,32,32)
        return self.decoder(h)

    def forward(self,x):
        encoded=self.encoder(x)
        mu,logvar=self.mu(encoded),self.logvar(encoded)
        latent=mu+torch.randn_like(mu)*torch.exp(.5*logvar)
        return self.decode(latent),mu,logvar

    @torch.no_grad()
    def sample(self,count,device="cpu"):
        was_training=self.training
        self.eval()
        result=self.decode(torch.randn(count,self.latent_dim,device=device))
        self.train(was_training)
        return result


def vae_loss(pred,target,mu,logvar):
    reconstruction=F.binary_cross_entropy(pred,target,reduction="sum")
    kl=-0.5*torch.sum(1+logvar-mu.square()-logvar.exp())
    return reconstruction+kl,reconstruction,kl
