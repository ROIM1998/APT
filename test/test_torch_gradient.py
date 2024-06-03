import torch

if __name__ == '__main__':
    # Testing the gradient correlation in a simplified setting: L = f(H), H = mWX
    W = torch.randn(768, 768, requires_grad=True)
    X = torch.randn(768, 128, requires_grad=True)
    t = torch.randn(768, 128, requires_grad=True)
    m = torch.ones(768, requires_grad=True)
    H = (m * (W @ X).T).T
    H.retain_grad()
    loss = (H - t).pow(2).sum()
    loss.backward()
    # Compare the grad calculated by autograd, the fisher information, and the sensitivity expression
    autograd_g = m.grad
    fisher_g = torch.diag(H.grad @ H.T)
    other_fisher_g = (H.grad * H).sum(dim=1)
    movement_sensitivity = (W.grad * W).sum(dim=1)
    print("Autograd equals to fisher information: ", torch.allclose(autograd_g, fisher_g))
    print("Autograd equals to movement sensitivity: ", torch.allclose(autograd_g, movement_sensitivity))
    print("Other fisher equals to movement sensitivity: ", torch.allclose(other_fisher_g, movement_sensitivity))

    # Testing the more complicated setting: L = f(H), H = BmAX, and the intermediate result P = AX
    A = torch.randn(768, 768, requires_grad=True).double()
    X = torch.randn(768, 128, requires_grad=True).double()
    B = torch.randn(8, 768, requires_grad=True).double()
    t = torch.randn(8, 128, requires_grad=True).double()
    m = torch.ones(768, requires_grad=True).double()
    m.retain_grad()
    P = (m * (A @ X).T).T
    P.retain_grad()
    H = B @ P
    H.retain_grad()
    loss = (H - t).pow(2).sum()
    loss.backward()
    # Compare the grad calculated by autograd, the fisher information, and the sensitivity expression
    autograd_g = m.grad
    fisher_g = torch.diag(B.T @ H.grad @ P.T)
    print("Autograd equals to fisher information: ", torch.allclose(autograd_g, fisher_g))

    # Representing the formula: L = f(H), H = M_p((W + s M_gb W_B M_r W_A M_ga)X + b)
    W = torch.randn(768, 768, requires_grad=True).double()
    W_A = torch.randn(8, 768, requires_grad=True).double()
    W_B = torch.randn(768, 8, requires_grad=True).double()
    X = torch.randn(768, 128, requires_grad=True).double()
    b = torch.randn(768, requires_grad=True).double()
    M_p, M_gb, M_r, M_ga = torch.diag(torch.ones(768)).double(), torch.diag(torch.ones(768)).double(), torch.diag(torch.ones(8)).double(), torch.diag(torch.ones(768)).double()
    W.retain_grad()
    W_A.retain_grad()
    W_B.retain_grad()
    b.retain_grad()
    t = torch.randn(768, 128, requires_grad=True).double()
    M_p.requires_grad = True
    M_gb.requires_grad = True
    M_r.requires_grad = True
    M_ga.requires_grad = True
    intermediate = (((W + M_gb @ W_B @ M_r @ W_A @ M_ga) @ X).T + b).T
    H = M_p @ intermediate
    H.retain_grad()
    loss = (H - t).pow(2).sum()
    loss.backward(retain_graph=True)

    # Compare the grad calculated by autograd, the fisher information, and the sensitivity expression
    autograd_g = torch.diag(M_p.grad)
    fisher_g = torch.diag(H.grad @ H.T)
    movement_sensitivity = (H.grad * H).sum(dim=1)
    autograd_mgb_g = torch.diag(M_gb.grad)
    movement_wb = (W_B.grad * W_B).sum(dim=1)
    autograd_mga_g = torch.diag(M_ga.grad)
    movement_wa = (W_A.grad * W_A).sum(dim=0)
    reverse_movement_wb = (W_B.grad * W_B).sum(dim=0)
    autograd_mr_g = torch.diag(M_r.grad)
    movement_wr = (W_A.grad * W_A).sum(dim=1)
    print("Autograd equals to fisher information: ", torch.allclose(autograd_g, fisher_g))
    print("Autograd equals to movement sensitivity: ", torch.allclose(autograd_g, movement_sensitivity))
    print("Autograd M_gb equals to movement W_B: ", torch.allclose(autograd_mgb_g, movement_wb))
    print("Autograd M_ga equals to movement W_A: ", torch.allclose(autograd_mga_g, movement_wa))
    print("Autograd M_r equals to reverse movement W_B: ", torch.allclose(autograd_mr_g, reverse_movement_wb))
    print("Autograd M_r equals to movement W_A.T: ", torch.allclose(autograd_mr_g, movement_wr))
    added_mp_g = (W.grad * W).sum(dim=1) + b.grad * b + movement_wb
    print("Autograd M_p equals to added grad: ", torch.allclose(autograd_g, added_mp_g))