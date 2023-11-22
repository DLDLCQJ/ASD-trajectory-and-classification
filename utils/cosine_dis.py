def cosine_dis(x1, x2=None, eps=1e-8):
    """
    Compute the cosine distance between two sets of vectors.
    x1 (Tensor): A tensor of vectors.
    x2 (Tensor, optional): Another tensor of vectors. If None, computes distance within x1.
    eps (float): Small value to avoid division by zero.
    """
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
