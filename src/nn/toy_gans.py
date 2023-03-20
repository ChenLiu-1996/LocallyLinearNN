import torch
from linearity import linearity_constraint
from torch.autograd import Variable, grad


class GAN(torch.nn.Module):

    def __init__(self,
                 learning_rate: float = 1e-4,
                 device: torch.device = torch.device('cpu'),
                 batch_size: int = 4,
                 linearity_lambda: float = 0,
                 input_dim: int = 2,
                 output_dim: int = 2,
                 hidden_dim: int = 512):
        super(GAN, self).__init__()

        self.device = device
        self.linearity_lambda = linearity_lambda
        self.B = batch_size

        self.generator = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Flatten(),
            torch.nn.Sigmoid(),
        )

        self.generator.to(device)
        self.discriminator.to(device)

        self.loss_fn = torch.nn.BCELoss()
        self.ones = torch.ones(self.B, device=self.device)
        self.zeros = torch.zeros(self.B, device=self.device)
        self.opt_G = torch.optim.AdamW(self.generator.parameters(),
                                       lr=learning_rate)
        self.opt_D = torch.optim.AdamW(self.discriminator.parameters(),
                                       lr=learning_rate)

    def forward_G(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)

    def forward_D(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)

    def optimize_G(self, real_dist_gen):
        z = torch.randn((self.B, 2)).to(self.device)
        x_fake = self.forward_G(z)
        y_pred_fake = self.forward_D(x_fake).view(-1)
        loss_G = self.loss_fn(y_pred_fake, self.ones)
        if self.linearity_lambda > 0:
            x_real_1 = torch.from_numpy(real_dist_gen.__next__()).to(
                self.device)
            x_real_2 = torch.from_numpy(real_dist_gen.__next__()).to(
                self.device)
            loss_G = loss_G + self.linearity_lambda * linearity_constraint(
                x_real_1, x_real_2, self.generator)

        self.opt_G.zero_grad()
        loss_G.backward()
        self.opt_G.step()

    def optimize_D(self, real_dist_gen):
        x_real = torch.from_numpy(real_dist_gen.__next__()).to(self.device)
        with torch.no_grad():
            z = torch.randn((self.B, 2)).to(self.device)
            x_fake = self.forward_G(z)
        y_pred_real = self.forward_D(x_real).view(-1)
        y_pred_fake = self.forward_D(x_fake).view(-1)
        loss_D = self.loss_fn(y_pred_real, self.ones) + self.loss_fn(
            y_pred_fake, self.zeros)

        self.opt_D.zero_grad()
        loss_D.backward()
        self.opt_D.step()


class WGAN(torch.nn.Module):

    def __init__(self,
                 learning_rate: float = 1e-4,
                 device: torch.device = torch.device('cpu'),
                 batch_size: int = 4,
                 linearity_lambda: float = 0,
                 D_iters_per_G_iter: int = 5,
                 input_dim: int = 2,
                 output_dim: int = 2,
                 hidden_dim: int = 512):
        super(WGAN, self).__init__()

        self.device = device
        self.linearity_lambda = linearity_lambda
        self.B = batch_size
        self.D_iters_per_G_iter = D_iters_per_G_iter

        self.generator = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Flatten(),
            torch.nn.Sigmoid(),
        )

        self.generator.to(device)
        self.discriminator.to(device)

        self.opt_G = torch.optim.AdamW(self.generator.parameters(),
                                       lr=learning_rate)
        self.opt_D = torch.optim.AdamW(self.discriminator.parameters(),
                                       lr=learning_rate)

    def forward_G(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)

    def forward_D(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)

    def optimize_G(self, real_dist_gen):
        z = torch.randn((self.B, 2)).to(self.device)
        x_fake = self.forward_G(z)
        y_pred_fake = self.forward_D(x_fake).view(-1)
        loss_G = -torch.mean(y_pred_fake)
        if self.linearity_lambda > 0:
            x_real_1 = torch.from_numpy(real_dist_gen.__next__()).to(
                self.device)
            x_real_2 = torch.from_numpy(real_dist_gen.__next__()).to(
                self.device)
            loss_G = loss_G + self.linearity_lambda * linearity_constraint(
                x_real_1, x_real_2, self.generator)

        self.opt_G.zero_grad()
        loss_G.backward()
        self.opt_G.step()

    def optimize_D(self, real_dist_gen):
        for _ in range(self.D_iters_per_G_iter):
            x_real = torch.from_numpy(real_dist_gen.__next__()).to(self.device)
            with torch.no_grad():
                z = torch.randn((self.B, 2)).to(self.device)
                x_fake = self.forward_G(z)
            y_pred_real = self.forward_D(x_real).view(-1)
            y_pred_fake = self.forward_D(x_fake).view(-1)
            loss_D = torch.mean(y_pred_fake) - torch.mean(y_pred_real)

            self.opt_D.zero_grad()
            loss_D.backward()
            self.opt_D.step()


class WGANGP(torch.nn.Module):

    def __init__(self,
                 learning_rate: float = 1e-4,
                 device: torch.device = torch.device('cpu'),
                 batch_size: int = 4,
                 linearity_lambda: float = 0,
                 D_iters_per_G_iter: int = 5,
                 gp_lambda: float = 10,
                 input_dim: int = 2,
                 output_dim: int = 2,
                 hidden_dim: int = 512):
        super(WGANGP, self).__init__()

        self.device = device
        self.linearity_lambda = linearity_lambda
        self.B = batch_size
        self.gp_lambda = gp_lambda
        self.D_iters_per_G_iter = D_iters_per_G_iter

        self.generator = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Flatten(),
            torch.nn.Sigmoid(),
        )

        self.generator.to(device)
        self.discriminator.to(device)

        self.opt_G = torch.optim.AdamW(self.generator.parameters(),
                                       lr=learning_rate)
        self.opt_D = torch.optim.AdamW(self.discriminator.parameters(),
                                       lr=learning_rate)

    def forward_G(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)

    def forward_D(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)

    def optimize_G(self, real_dist_gen):
        z = torch.randn((self.B, 2)).to(self.device)
        x_fake = self.forward_G(z)
        y_pred_fake = self.forward_D(x_fake).view(-1)
        loss_G = -torch.mean(y_pred_fake)
        if self.linearity_lambda > 0:
            x_real_1 = torch.from_numpy(real_dist_gen.__next__()).to(
                self.device)
            x_real_2 = torch.from_numpy(real_dist_gen.__next__()).to(
                self.device)
            loss_G = loss_G + self.linearity_lambda * linearity_constraint(
                x_real_1, x_real_2, self.generator)

        self.opt_G.zero_grad()
        loss_G.backward()
        self.opt_G.step()

    def optimize_D(self, real_dist_gen):
        for _ in range(self.D_iters_per_G_iter):
            x_real = torch.from_numpy(real_dist_gen.__next__()).to(self.device)
            with torch.no_grad():
                z = torch.randn((self.B, 2)).to(self.device)
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


def gradient_penalty(real_img, fake_img, D):
    """
    Adapted from
    https://github.com/EmilienDupont/wgan-gp/blob/ef82364f2a2ec452a52fbf4a739f95039ae76fe3/training.py#L73
    """
    batch_size = real_img.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand_as(real_img).to(real_img.device)
    interpolated = alpha * real_img.data + (1 - alpha) * fake_img.data
    interpolated = Variable(interpolated,
                            requires_grad=True).to(real_img.device)

    logit_interpolated = D(interpolated)

    # Calculate gradients of logit with respect to examples
    gradients = grad(outputs=logit_interpolated,
                     inputs=interpolated,
                     grad_outputs=torch.ones(logit_interpolated.size()).to(
                         real_img.device),
                     create_graph=True,
                     retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)

    # Return gradient penalty
    return ((gradients_norm - 1)**2).mean()