import torch
from gradient_penalty import gradient_penalty
from linearity import linearity_constraint, sort_minimize_dist


class GAN(torch.nn.Module):

    def __init__(self,
                 learning_rate: float = 1e-4,
                 device: torch.device = torch.device('cpu'),
                 batch_size: int = 4,
                 linearity_lambda: float = 0,
                 linearity_include_D: bool = False,
                 z_dim: int = 2,
                 output_dim: int = 2,
                 hidden_dim: int = 512):
        super(GAN, self).__init__()

        self.device = device
        self.linearity_lambda = linearity_lambda
        self.linearity_include_D = linearity_include_D
        self.B = batch_size
        self.z_dim = z_dim

        self.generator = torch.nn.Sequential(
            torch.nn.Linear(z_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(output_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Flatten(),
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

    def optimize_D(self, real_dist_gen):
        x_real = torch.from_numpy(real_dist_gen.__next__()).to(self.device)
        with torch.no_grad():
            z = torch.randn((self.B, self.z_dim)).to(self.device)
            x_fake = self.forward_G(z)
        y_pred_real = self.forward_D(x_real).view(-1)
        y_pred_fake = self.forward_D(x_fake).view(-1)
        loss_D = self.loss_fn(y_pred_real, self.ones) + self.loss_fn(
            y_pred_fake, self.zeros)
        if self.linearity_include_D:
            assert self.linearity_lambda > 0
            x_real_prime = torch.from_numpy(real_dist_gen.__next__()).to(
                self.device)
            x_real_prime = sort_minimize_dist(tensor_moving=x_real_prime,
                                              tensor_fixed=x_real)
            loss_D = loss_D + self.linearity_lambda * linearity_constraint(
                x_real, x_real_prime, self.discriminator)

        self.opt_D.zero_grad()
        loss_D.backward()
        self.opt_D.step()


class WGAN(torch.nn.Module):

    def __init__(
            self,
            learning_rate: float = 1e-4,
            device: torch.device = torch.device('cpu'),
            batch_size: int = 4,
            linearity_lambda: float = 0,
            linearity_include_D: bool = False,
            D_iters_per_G_iter: int = 5,
            grad_norm: float = 1.0,
            z_dim: int = 2,
            output_dim: int = 2,
            hidden_dim: int = 512):
        super(WGAN, self).__init__()

        self.device = device
        self.linearity_lambda = linearity_lambda
        self.linearity_include_D = linearity_include_D
        self.B = batch_size
        self.D_iters_per_G_iter = D_iters_per_G_iter
        self.grad_norm = grad_norm
        self.z_dim = z_dim

        self.generator = torch.nn.Sequential(
            torch.nn.Linear(z_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(output_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Flatten(),
        )

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

    def optimize_D(self, real_dist_gen):
        for _ in range(self.D_iters_per_G_iter):
            x_real = torch.from_numpy(real_dist_gen.__next__()).to(self.device)
            with torch.no_grad():
                z = torch.randn((self.B, self.z_dim)).to(self.device)
                x_fake = self.forward_G(z)
            y_pred_real = self.forward_D(x_real).view(-1)
            y_pred_fake = self.forward_D(x_fake).view(-1)
            loss_D = torch.mean(y_pred_fake) - torch.mean(y_pred_real)
            if self.linearity_include_D:
                assert self.linearity_lambda > 0
                x_real_prime = torch.from_numpy(real_dist_gen.__next__()).to(
                    self.device)
                x_real_prime = sort_minimize_dist(tensor_moving=x_real_prime,
                                                  tensor_fixed=x_real)
                loss_D = loss_D + self.linearity_lambda * linearity_constraint(
                    x_real, x_real_prime, self.discriminator)

            self.opt_D.zero_grad()
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(),
                                           max_norm=self.grad_norm)
            self.opt_D.step()


class WGANGP(torch.nn.Module):

    def __init__(
            self,
            learning_rate: float = 1e-4,
            device: torch.device = torch.device('cpu'),
            batch_size: int = 4,
            linearity_lambda: float = 0,
            linearity_include_D: bool = False,
            D_iters_per_G_iter: int = 5,
            gp_lambda: float = 10,
            z_dim: int = 2,
            output_dim: int = 2,
            hidden_dim: int = 512):
        super(WGANGP, self).__init__()

        self.device = device
        self.linearity_lambda = linearity_lambda
        self.linearity_include_D = linearity_include_D
        self.B = batch_size
        self.z_dim = z_dim
        self.gp_lambda = gp_lambda
        self.D_iters_per_G_iter = D_iters_per_G_iter

        self.generator = torch.nn.Sequential(
            torch.nn.Linear(z_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(output_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Flatten(),
        )

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

    def optimize_D(self, real_dist_gen):
        for _ in range(self.D_iters_per_G_iter):
            x_real = torch.from_numpy(real_dist_gen.__next__()).to(self.device)
            with torch.no_grad():
                z = torch.randn((self.B, self.z_dim)).to(self.device)
                x_fake = self.forward_G(z)
            y_pred_real = self.forward_D(x_real).view(-1)
            y_pred_fake = self.forward_D(x_fake).view(-1)
            gp = gradient_penalty(x_real.detach(), x_fake.detach(),
                                  self.discriminator)
            loss_D = torch.mean(y_pred_fake) - torch.mean(
                y_pred_real) + self.gp_lambda * gp
            if self.linearity_include_D:
                assert self.linearity_lambda > 0
                x_real_prime = torch.from_numpy(real_dist_gen.__next__()).to(
                    self.device)
                x_real_prime = sort_minimize_dist(tensor_moving=x_real_prime,
                                                  tensor_fixed=x_real)
                loss_D = loss_D + self.linearity_lambda * linearity_constraint(
                    x_real, x_real_prime, self.discriminator)

            self.opt_D.zero_grad()
            loss_D.backward()
            self.opt_D.step()
