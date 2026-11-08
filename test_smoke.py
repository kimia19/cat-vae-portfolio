"""Offline public-edition 256px VAE shape and gradient tests."""
import unittest
import torch
from vae_network import ConvolutionalVAE,vae_loss


class PortfolioVaeTests(unittest.TestCase):
    def test_shapes_loss_and_grad(self):
        torch.set_num_threads(2)
        net=ConvolutionalVAE(latent_dim=4)
        x=torch.rand(1,3,256,256)
        y,mu,logvar=net(x)
        self.assertEqual(tuple(y.shape),(1,3,256,256))
        self.assertEqual(tuple(mu.shape),(1,4))
        total,bce,kl=vae_loss(y,x,mu,logvar)
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(bce))
        self.assertTrue(torch.isfinite(kl))
        total.backward()
        self.assertIsNotNone(net.mu.weight.grad)
        with torch.no_grad():
            sample=net.sample(2)
        self.assertEqual(tuple(sample.shape),(2,3,256,256))


if __name__=="__main__":
    unittest.main()
