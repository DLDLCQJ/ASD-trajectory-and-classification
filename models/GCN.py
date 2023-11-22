class GCN(nn.Module):
    def __init__(self, in_dim, lat_dim, k_order, num_layers, dropout, model_type):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout

        # Initialize layers based on model_type
        for i in range(num_layers):
            in_features = in_dim if i == 0 else lat_dim[i-1]
            out_features = lat_dim[i]
            if model_type == 'GCNConv':
                self.layers.append(GCNConv(in_features, out_features))
            elif model_type == 'ChebConv':
                self.layers.append(ChebConv(in_features, out_features, k_order))

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        return x
