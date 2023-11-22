class LMF(nn.Module):
    """
    Low-rank Multimodal Fusion (LMF) for combining features from multiple modalities.
    """

    def __init__(self, input_dim, output_dim, rank):
        super(LMF, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank

        # Initialize factors and fusion parameters
        self.factors = nn.ParameterList([nn.Parameter(torch.randn(input_dim + 1, rank)) for _ in range(input_dim)])
        self.fusion_weights = nn.Parameter(torch.randn(rank, output_dim))
        self.fusion_bias = nn.Parameter(torch.zeros(output_dim))

        self._init_weights()

    def _init_weights(self):
        for factor in self.factors:
            nn.init.xavier_normal_(factor)
        nn.init.xavier_normal_(self.fusion_weights)
        nn.init.zeros_(self.fusion_bias)

    def forward(self, inputs):
        batch_size = inputs[0].size(0)

        # Add bias term to inputs and apply non-linearity
        modified_inputs = [torch.cat([input, torch.ones(batch_size, 1)], dim=1) for input in inputs]
        modified_inputs = [torch.sigmoid(input) for input in modified_inputs]

        # Apply low-rank fusion
        fusion = torch.stack([torch.matmul(input, factor) for input, factor in zip(modified_inputs, self.factors)])
        fusion = torch.prod(fusion, dim=0)
        output = torch.matmul(fusion, self.fusion_weights) + self.fusion_bias

        return output
