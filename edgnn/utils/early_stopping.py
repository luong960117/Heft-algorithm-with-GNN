import numpy as np
from .io import save_checkpoint


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, save_path=''):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_path)
        elif score < self.best_score:
            self.counter += 1
            print('EarlyStopping counter:', self.counter,' out of', self.patience)
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_path)
            self.counter = 0
        return True

    def save_checkpoint(self, val_loss, model, save_path='', verbose=True):
        """
        Saves model when validation loss decrease.

        """
        if verbose:
            print('Validation loss decreased:', self.val_loss_min, ' -->',  val_loss, 'Saving model ...')
        save_checkpoint(model, save_path)
        self.val_loss_min = val_loss
