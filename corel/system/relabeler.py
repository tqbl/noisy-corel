import pandas as pd


class Relabeler:
    def __init__(self,
                 pred_path,
                 method='corel',
                 threshold=0,
                 target_fn=None,
                 **kwargs,
                 ):
        self.y_pred = pd.read_csv(pred_path, index_col=0)
        self.threshold = threshold
        self.target_fn = target_fn

        if method == 'fixed_weight':
            self.weight = kwargs['weight']
        elif method == 'corel':
            y_conf = self.y_pred.max(axis=1)
            self.weight = 1 / (1 + y_conf / kwargs['noise_rate'])

    def __call__(self, subset, index=None):
        if index is None:
            fnames = subset.tags.index
            y = pd.concat([self(subset, fname) for fname in fnames],
                          axis=1, keys=fnames).T
            return y

        target_fn = self.target_fn or subset.dataset.target
        y_true = target_fn(subset, index)
        if index not in self.y_pred.index:
            return y_true

        y_pred = self.y_pred.loc[index]
        y_true_k = y_true.argmax()
        y_pred_k = y_pred.argmax()
        if y_pred_k != y_true_k and y_pred.iloc[y_pred_k] > self.threshold:
            weight = self.weight.loc[index]
            return weight * y_true + (1 - weight) * y_pred
        return y_true
