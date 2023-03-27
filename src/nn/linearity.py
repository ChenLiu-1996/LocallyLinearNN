import torch
from scipy.optimize import linear_sum_assignment


def linearity_constraint(x1: torch.Tensor, x2: torch.Tensor,
                         f: torch.nn.Module, constraint_power: int = 1) -> torch.Tensor:
    '''
    $||f(x2) - f(x1) - <\nabla_x1 f(x1), x2 - x1>||_2$

    Currently only support 2 senarios:
        f() is a classifier / point generator -> f output shape: [B, d]
        f() is an image generator -> f output shape: [B, C, H, W]
    '''

    device = x1.device

    # Track gradient for computing $\nabla_x1 f(x1)$
    x1_grad = x1.clone()
    x1_grad.requires_grad_(True)

    f_x1_grad = f(x1_grad)

    f_output_shape = len(f_x1_grad.shape)
    assert f_output_shape in [2, 4], \
        'Shape of `f` output not supported. Supported: 2 or 4. Current: %d' % f_output_shape

    if f_output_shape == 2:
        # f() is a classifer or point generator
        B, d = f_x1_grad.shape

        # shape: [B, d, x1.shape[1:]]
        df_dx1 = torch.cat([
            torch.autograd.grad(outputs=f_x1_grad[:, i],
                                inputs=x1_grad,
                                grad_outputs=torch.ones(B).to(device),
                                create_graph=True,
                                retain_graph=True)[0][:, None, ...] for i in range(d)
        ], dim=1)

        # Sum over excessive `x` dimension (after batch dimension).
        # shape: [B, d]
        inner_product = torch.sum(df_dx1 * (x2 - x1)[:, None, ...].repeat(
            1, d, *[1 for _ in range(len(x1.shape[1:]))]),
                                  dim=list(
                                      torch.arange(len(df_dx1.shape))[2:]))

    elif f_output_shape == 4:
        # f() is an image generator
        B, C, H, W = f_x1_grad.shape

        # shape: [B, C * H * W, x1.shape[1:]]
        df_dx1 = torch.cat([
            torch.autograd.grad(outputs=f_x1_grad[:, i, j, k],
                                inputs=x1_grad,
                                grad_outputs=torch.ones(B).to(device),
                                create_graph=True,
                                retain_graph=True)[0][:, None, ...] for i in range(C)
            for j in range(H) for k in range(W)
        ], dim=1)

        df_dx1 = df_dx1.reshape(B, C, H, W, *x1.shape[1:])

        # Sum over excessive `x` dimension (after batch dimension).
        # shape: [B, C, H, W]
        inner_product = torch.sum(
            df_dx1 * (x2 - x1)[:, None, None, None, ...].repeat(
                1, C, H, W, *[1 for _ in range(len(x1.shape[1:]))]),
            dim=list(torch.arange(len(df_dx1.shape))[4:]))

    # shape: [B]
    constraint = torch.norm(f(x2).detach() - f(x1).detach() - inner_product,
                            p=constraint_power,
                            dim=1)

    return torch.mean(constraint)


def sort_minimize_dist(tensor_moving: torch.Tensor,
                       tensor_fixed: torch.Tensor) -> torch.Tensor:
    '''
    Might be useful if you want to reorganize the input `x`s such that
    `x1 - x2` represents a local perturbation.
    '''

    with torch.no_grad():
        assert tensor_moving.shape == tensor_fixed.shape
        if len(tensor_moving.shape) > 2:
            B = tensor_moving.shape[0]
            # Make sure not to modify the input tensors.
            tensor_moving = tensor_moving.clone().reshape(B, -1)
            tensor_fixed = tensor_fixed.clone().reshape(B, -1)

        cost = torch.cdist(tensor_fixed, tensor_moving).cpu()
        _, col_inds = linear_sum_assignment(cost)
        tensor_moving = tensor_moving[col_inds]
    return tensor_moving
