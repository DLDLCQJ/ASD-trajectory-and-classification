class StratifiedKFoldn(StratifiedKFold):
    def idx(self, X, y):
        np.random.seed(123)
        trval_idx = np.random.choice(np.arange(len(X)), size=int(len(y)*0.8), replace=False)
        te_idx = np.delete(np.arange(len(X)), trval_idx)
        return trval_idx, te_idx
    def split(self, X, y):
        np.random.seed(123)
        trval_idx, te_idx = self.idx(X,y)
        s = super().split(X[trval_idx],y[trval_idx])
        
        for tr_idx, val_idx in s:
            yield trval_idx, tr_idx, te_idx
