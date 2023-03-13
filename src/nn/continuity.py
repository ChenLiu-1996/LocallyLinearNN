import torch


def continuity_constraint(x1: torch.Tensor, x2: torch.Tensor,
                          f: torch.nn.Module) -> torch.Tensor:
    '''
    $||f(x1) - f(x2) - <\nabla_x1 f(x1), x2 - x1>||_2^2$
    '''

    # Track gradient for computing $\nabla_x1 f(x1)$
    x1.requires_grad_(True)

    # shape: [B, d]
    f_x1 = f(x1)
    f_x2 = f(x2)

    assert len(f_x1.shape) == 2
    _, d = f_x1.shape

    # shape: [B, C, H, W, d]
    df_dx1 = torch.stack([
        torch.autograd.grad(outputs=f_x1[:, idx],
                            inputs=x1,
                            grad_outputs=torch.ones(f_x1.shape[0]).to(
                                f_x1.device),
                            create_graph=True,
                            retain_graph=True)[0] for idx in range(d)
    ],
                         dim=4)

    x1.requires_grad_(False)

    # shape: [B, d]
    inner_product = torch.sum(df_dx1 *
                              (x2 - x1)[..., None].repeat(1, 1, 1, 1, d),
                              dim=[1, 2, 3])

    # shape: [B]
    constraint = torch.norm(f_x1 - f_x2 - inner_product, p=2, dim=1)

    return torch.mean(constraint)
