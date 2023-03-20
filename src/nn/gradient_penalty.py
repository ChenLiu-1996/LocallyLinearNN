import torch
from torch.autograd import Variable, grad


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