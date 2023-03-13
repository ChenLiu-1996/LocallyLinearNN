import torch


def continuity_constraint(x1: torch.Tensor, x2: torch.Tensor,
                          f: torch.nn.Module) -> torch.Tensor:
    '''
    $||f(x2) - f(x1) - <\nabla_x1 f(x1), x2 - x1>||_2^2$
    '''

    # Track gradient for computing $\nabla_x1 f(x1)$
    x1_grad = x1.clone()
    x1_grad.requires_grad_(True)

    # shape: [B, d]
    f_x1_grad = f(x1_grad)

    assert len(f_x1_grad.shape) == 2
    _, d = f_x1_grad.shape

    # shape: [B, C, H, W, d]
    df_dx1 = torch.stack([
        torch.autograd.grad(outputs=f_x1_grad[:, idx],
                            inputs=x1_grad,
                            grad_outputs=torch.ones(f_x1_grad.shape[0]).to(
                                f_x1_grad.device),
                            create_graph=True,
                            retain_graph=True)[0] for idx in range(d)
    ],
                         dim=4)

    # shape: [B, d]
    inner_product = torch.sum(df_dx1 *
                              (x2 - x1)[..., None].repeat(1, 1, 1, 1, d),
                              dim=[1, 2, 3])

    # shape: [B]
    constraint = torch.norm(f(x2) - f(x1) - inner_product, p=2, dim=1)

    return torch.mean(constraint)
