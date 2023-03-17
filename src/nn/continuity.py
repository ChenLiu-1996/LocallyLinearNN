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

    inner_product = torch.zeros_like(f_x1_grad)
    for d_idx in range(d):
        # shape: [B, C, H, W]
        df_dx1 = torch.autograd.grad(
            outputs=f_x1_grad[:, d_idx],
            inputs=x1_grad,
            grad_outputs=torch.ones(f_x1_grad.shape[0]).to(f_x1_grad.device),
            create_graph=True,
            retain_graph=True)[0]

        # shape: [B, d]
        inner_product[:, d_idx] = torch.sum(df_dx1 * (x2 - x1), dim=[1, 2, 3])

    # shape: [B]
    constraint = torch.norm(f(x2).detach() - f(x1).detach() - inner_product,
                            p=2,
                            dim=1)

    return torch.mean(constraint)
