def set_seed(seed=123):
    """
    Set the seed for reproducibility across various random number generators.

    Args:
    seed (int): Seed value for random number generators.

    Returns:
    None
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Use `manual_seed_all` for multi-GPU.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
