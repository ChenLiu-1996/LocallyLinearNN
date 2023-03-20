from typing import Tuple

import numpy as np
import torch
from gradient_penalty import gradient_penalty
from linearity import linearity_constraint, sort_minimize_dist


def normal_init(m, mean, std):
    if isinstance(m, torch.nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class Generator(torch.nn.Module):

    def __init__(self,
                 latent_dim: int = 128,
                 num_channel: int = 3,
                 resize_conv: bool = True,
                 img_size: Tuple[int] = (64, 64)):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        assert img_size[0] == img_size[1]
        img_size = img_size[0]
        assert 2**(int(np.log2(img_size))
                   ) == img_size, '`img_size` not an integer power of 2.'
        self.img_size = img_size
        self.num_layers = int(np.log2(img_size)) - 1

        self.deconvs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        in_dim = None
        out_dim = int(latent_dim * 2**(self.num_layers // 2))

        for i in range(self.num_layers):
            if resize_conv:
                """
                Use resize-convolution instead of conv transpose.
                """
                if i == 0:
                    self.deconvs.append(
                        torch.nn.Sequential(
                            torch.nn.Upsample(scale_factor=4,
                                              mode='bicubic',
                                              align_corners=True),
                            torch.nn.Conv2d(latent_dim,
                                            out_dim,
                                            kernel_size=3,
                                            stride=1,
                                            padding='same'),
                        ))
                    self.bns.append(torch.nn.BatchNorm2d(out_dim))
                elif i < self.num_layers - 1:
                    self.deconvs.append(
                        torch.nn.Sequential(
                            torch.nn.Upsample(scale_factor=2,
                                              mode='bicubic',
                                              align_corners=True),
                            torch.nn.Conv2d(in_dim,
                                            out_dim,
                                            kernel_size=5,
                                            stride=1,
                                            padding='same'),
                        ))
                    self.bns.append(torch.nn.BatchNorm2d(out_dim))
                else:
                    self.deconvs.append(
                        torch.nn.Sequential(
                            torch.nn.Upsample(scale_factor=2,
                                              mode='bicubic',
                                              align_corners=True),
                            torch.nn.Conv2d(in_dim,
                                            num_channel,
                                            kernel_size=5,
                                            stride=1,
                                            padding='same'),
                        ))
            else:
                if i == 0:
                    self.deconvs.append(
                        torch.nn.ConvTranspose2d(latent_dim,
                                                 out_dim,
                                                 kernel_size=4,
                                                 stride=1,
                                                 padding=0))
                    self.bns.append(torch.nn.BatchNorm2d(out_dim))
                elif i < self.num_layers - 1:
                    self.deconvs.append(
                        torch.nn.ConvTranspose2d(in_dim,
                                                 out_dim,
                                                 kernel_size=4,
                                                 stride=2,
                                                 padding=1))
                    self.bns.append(torch.nn.BatchNorm2d(out_dim))
                else:
                    self.deconvs.append(
                        torch.nn.ConvTranspose2d(in_dim,
                                                 num_channel,
                                                 kernel_size=4,
                                                 stride=2,
                                                 padding=1))

            if i == 0:
                in_dim = out_dim
            elif i % 2 == 0:
                in_dim = in_dim // 2
            else:
                out_dim = out_dim // 2

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        assert len(x.shape) == 2
        x = x.view(*x.shape, 1, 1)
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                x = self.deconvs[i](x)
                x = self.bns[i](x)
                x = torch.nn.functional.relu(x)
            else:
                x = self.deconvs[i](x)
                x = torch.tanh(x)
        return x


class Discriminator(torch.nn.Module):

    def __init__(self,
                 latent_dim: int = 128,
                 num_channel: int = 3,
                 img_size: Tuple[int] = (64, 64)):
        super(Discriminator, self).__init__()
        self.latent_dim = latent_dim
        assert img_size[0] == img_size[1]
        img_size = img_size[0]
        assert 2**(int(np.log2(img_size))
                   ) == img_size, '`img_size` not an integer power of 2.'
        self.img_size = img_size
        self.num_layers = int(np.log2(img_size)) - 1

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        in_dim = latent_dim
        if self.num_layers % 2 == 0:
            out_dim = in_dim * 2
        else:
            out_dim = in_dim

        for i in range(self.num_layers):
            if i == 0:
                self.convs.append(
                    torch.nn.Sequential(
                        torch.nn.Conv2d(num_channel,
                                        out_dim,
                                        kernel_size=5,
                                        stride=1,
                                        padding='same'),
                        torch.nn.AvgPool2d(kernel_size=2)))
                self.bns.append(torch.nn.BatchNorm2d(out_dim))
            elif i < self.num_layers - 1:
                self.convs.append(
                    torch.nn.Sequential(
                        torch.nn.Conv2d(in_dim,
                                        out_dim,
                                        kernel_size=5,
                                        stride=1,
                                        padding='same'),
                        torch.nn.AvgPool2d(kernel_size=2)))
                self.bns.append(torch.nn.BatchNorm2d(out_dim))
            else:
                self.convs.append(
                    torch.nn.Sequential(
                        torch.nn.Conv2d(in_dim,
                                        1,
                                        kernel_size=3,
                                        stride=1,
                                        padding='same'),
                        torch.nn.AvgPool2d(kernel_size=4)))

            if i % 2 == 0:
                in_dim = in_dim * 2
            else:
                out_dim = out_dim * 2

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                x = self.convs[i](x)
                x = self.bns[i](x)
                x = torch.nn.functional.relu(x)
            else:
                x = self.convs[i](x)
        return x


class GAN(torch.nn.Module):

    def __init__(self,
                 learning_rate: float = 1e-4,
                 device: torch.device = torch.device('cpu'),
                 batch_size: int = 4,
                 linearity_lambda: float = 0,
                 z_dim: int = 128,
                 num_channel: int = 3,
                 img_size: Tuple[int] = (96, 96)):

        super(GAN, self).__init__()

        self.device = device
        self.linearity_lambda = linearity_lambda
        self.B = batch_size
        self.z_dim = z_dim

        self.generator = Generator(latent_dim=z_dim,
                                   resize_conv=True,
                                   num_channel=num_channel,
                                   img_size=img_size)

        self.discriminator = Discriminator(latent_dim=z_dim,
                                           num_channel=num_channel,
                                           img_size=img_size)

        self.generator.to(device)
        self.discriminator.to(device)

        self.loss_fn = torch.nn.BCELoss()
        self.ones = torch.ones(self.B, device=self.device)
        self.zeros = torch.zeros(self.B, device=self.device)
        self.opt_G = torch.optim.AdamW(self.generator.parameters(),
                                       lr=learning_rate)
        self.opt_D = torch.optim.AdamW(self.discriminator.parameters(),
                                       lr=learning_rate)

    def forward_G(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)

    def forward_D(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.discriminator(x))

    def optimize_G(self):
        z = torch.randn((self.B, self.z_dim)).to(self.device)
        x_fake = self.forward_G(z)
        y_pred_fake = self.forward_D(x_fake).view(-1)
        loss_G = self.loss_fn(y_pred_fake, self.ones)
        if self.linearity_lambda > 0:
            z_prime = torch.randn((self.B, self.z_dim)).to(self.device)
            z_prime = sort_minimize_dist(tensor_moving=z_prime, tensor_fixed=z)
            loss_G = loss_G + self.linearity_lambda * linearity_constraint(
                z, z_prime, self.generator)

        self.opt_G.zero_grad()
        loss_G.backward()
        self.opt_G.step()

    def optimize_D(self, x_real):
        x_real = x_real.to(self.device)
        with torch.no_grad():
            z = torch.randn((self.B, self.z_dim)).to(self.device)
            x_fake = self.forward_G(z)
        y_pred_real = self.forward_D(x_real).view(-1)
        y_pred_fake = self.forward_D(x_fake).view(-1)
        loss_D = self.loss_fn(y_pred_real, self.ones) + self.loss_fn(
            y_pred_fake, self.zeros)

        self.opt_D.zero_grad()
        loss_D.backward()
        self.opt_D.step()

    def train(self):
        self.generator.train()
        self.discriminator.train()

    def eval(self):
        self.generator.eval()
        self.discriminator.eval()


class WGAN(torch.nn.Module):

    def __init__(self,
                 learning_rate: float = 1e-4,
                 device: torch.device = torch.device('cpu'),
                 batch_size: int = 4,
                 linearity_lambda: float = 0,
                 D_iters_per_G_iter: int = 5,
                 grad_norm: float = 1.0,
                 z_dim: int = 128,
                 num_channel: int = 3,
                 img_size: Tuple[int] = (96, 96)):

        super(WGAN, self).__init__()

        self.device = device
        self.linearity_lambda = linearity_lambda
        self.B = batch_size
        self.z_dim = z_dim
        self.D_iters_per_G_iter = D_iters_per_G_iter
        self.grad_norm = grad_norm

        self.generator = Generator(latent_dim=z_dim,
                                   resize_conv=True,
                                   num_channel=num_channel,
                                   img_size=img_size)

        self.discriminator = Discriminator(latent_dim=z_dim,
                                           num_channel=num_channel,
                                           img_size=img_size)

        self.generator.to(device)
        self.discriminator.to(device)

        self.opt_G = torch.optim.AdamW(self.generator.parameters(),
                                       lr=learning_rate)
        self.opt_D = torch.optim.AdamW(self.discriminator.parameters(),
                                       lr=learning_rate)

    def forward_G(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)

    def forward_D(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)

    def optimize_G(self):
        z = torch.randn((self.B, self.z_dim)).to(self.device)
        x_fake = self.forward_G(z)
        y_pred_fake = self.forward_D(x_fake).view(-1)
        loss_G = -torch.mean(y_pred_fake)
        if self.linearity_lambda > 0:
            z_prime = torch.randn((self.B, self.z_dim)).to(self.device)
            z_prime = sort_minimize_dist(tensor_moving=z_prime, tensor_fixed=z)
            loss_G = loss_G + self.linearity_lambda * linearity_constraint(
                z, z_prime, self.generator)

        self.opt_G.zero_grad()
        loss_G.backward()
        self.opt_G.step()

    def optimize_D(self, x_real):
        for _ in range(self.D_iters_per_G_iter):
            x_real = x_real.to(self.device)
            with torch.no_grad():
                z = torch.randn((self.B, self.z_dim)).to(self.device)
                x_fake = self.forward_G(z)
            y_pred_real = self.forward_D(x_real).view(-1)
            y_pred_fake = self.forward_D(x_fake).view(-1)
            loss_D = torch.mean(y_pred_fake) - torch.mean(y_pred_real)

            self.opt_D.zero_grad()
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(),
                                           max_norm=self.grad_norm)
            self.opt_D.step()

    def train(self):
        self.generator.train()
        self.discriminator.train()

    def eval(self):
        self.generator.eval()
        self.discriminator.eval()


class WGANGP(torch.nn.Module):

    def __init__(self,
                 learning_rate: float = 1e-4,
                 device: torch.device = torch.device('cpu'),
                 batch_size: int = 4,
                 linearity_lambda: float = 0,
                 D_iters_per_G_iter: int = 5,
                 gp_lambda: float = 10,
                 z_dim: int = 128,
                 num_channel: int = 3,
                 img_size: Tuple[int] = (96, 96)):

        super(WGANGP, self).__init__()

        self.device = device
        self.linearity_lambda = linearity_lambda
        self.B = batch_size
        self.z_dim = z_dim
        self.gp_lambda = gp_lambda
        self.D_iters_per_G_iter = D_iters_per_G_iter

        self.generator = Generator(latent_dim=z_dim,
                                   resize_conv=True,
                                   num_channel=num_channel,
                                   img_size=img_size)

        self.discriminator = Discriminator(latent_dim=z_dim,
                                           num_channel=num_channel,
                                           img_size=img_size)

        self.generator.to(device)
        self.discriminator.to(device)

        self.opt_G = torch.optim.AdamW(self.generator.parameters(),
                                       lr=learning_rate)
        self.opt_D = torch.optim.AdamW(self.discriminator.parameters(),
                                       lr=learning_rate)

    def forward_G(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)

    def forward_D(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)

    def optimize_G(self):
        z = torch.randn((self.B, self.z_dim)).to(self.device)
        x_fake = self.forward_G(z)
        y_pred_fake = self.forward_D(x_fake).view(-1)
        loss_G = -torch.mean(y_pred_fake)
        if self.linearity_lambda > 0:
            z_prime = torch.randn((self.B, self.z_dim)).to(self.device)
            z_prime = sort_minimize_dist(tensor_moving=z_prime, tensor_fixed=z)
            loss_G = loss_G + self.linearity_lambda * linearity_constraint(
                z, z_prime, self.generator)

        self.opt_G.zero_grad()
        loss_G.backward()
        self.opt_G.step()

    def optimize_D(self, x_real):
        for _ in range(self.D_iters_per_G_iter):
            x_real = x_real.to(self.device)
            with torch.no_grad():
                z = torch.randn((self.B, self.z_dim)).to(self.device)
                x_fake = self.forward_G(z)
            y_pred_real = self.forward_D(x_real).view(-1)
            y_pred_fake = self.forward_D(x_fake).view(-1)
            gp = gradient_penalty(x_real.detach(), x_fake.detach(),
                                  self.discriminator)
            loss_D = torch.mean(y_pred_fake) - torch.mean(
                y_pred_real) + self.gp_lambda * gp

            self.opt_D.zero_grad()
            loss_D.backward()
            self.opt_D.step()

    def train(self):
        self.generator.train()
        self.discriminator.train()

    def eval(self):
        self.generator.eval()
        self.discriminator.eval()
