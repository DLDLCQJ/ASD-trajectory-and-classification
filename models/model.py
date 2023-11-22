class GCNConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCNConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        return output + self.bias if self.bias is not None else output

class ChebConv(nn.Module):
    def __init__(self, in_features, out_features, k):
        super(ChebConv, self).__init__()
        self.K = k
        self.linear = nn.Linear(in_features * k, out_features)

    def forward(self, x, laplacian):
        cheb_x = self.transform_to_chebyshev(x, laplacian)
        return self.linear(cheb_x)

    def transform_to_chebyshev(self, x, laplacian):
        cheb_x = [x]
        if self.K > 1:
            x1 = torch.sparse.mm(laplacian, x)
            cheb_x.append(x1)

            for _ in range(2, self.K):
                x2 = 2 * torch.sparse.mm(laplacian, x1) - x
                cheb_x.append(x2)
                x, x1 = x1, x2

        return torch.cat(cheb_x, dim=-1)

class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(in_dim, out_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        return self.classifier(x)
