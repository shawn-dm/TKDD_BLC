import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=100, verbose=False):
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
        self.accs=0
        self.F1 = 0
        self.F2 = 0
        self.F3 = 0
        self.F4 = 0
        self.P1 = 0
        self.P2 = 0
        self.P3 = 0
        self.P4 = 0
        self.R1 = 0
        self.R2 = 0
        self.R3 = 0
        self.R4 = 0
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, accs,F1,F2,F3,F4,P1,P2,P3,P4,R1,R2,R3,R4,model,modelname,str):

        score = accs

        if self.best_score is None:
            self.best_score = score
            self.accs = accs
            self.F1 = F1
            self.F2 = F2
            self.F3 = F3
            self.F4 = F4
            self.P1 = P1
            self.P2 = P2
            self.P3 = P3
            self.P4 = P4
            self.R1 = R1
            self.R2 = R2
            self.R3 = R3
            self.R4 = R4
            self.save_checkpoint(val_loss, model,modelname,str)
        elif score < self.best_score:
            self.counter += 1
            # print('EarlyStopping counter: {} out of {}'.format(self.counter,self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                print("BEST Accuracy: {:.4f}|NR F1: {:.4f} Precision: {:.4f} Recall: {:.4f} |FR  F1: {:.4f}  Precision: {:.4f} Recall: {:.4f}|TR F1: {:.4f}  Precision: {:.4f} Recall: {:.4f}|UR F1: {:.4f} Precision: {:.4f} Recall: {:.4f}"
                      .format(self.accs,self.F1,self.P1,self.R1,self.F2,self.P2,self.R2,self.F3,self.P3,self.R3,self.F4,self.P4,self.R4))
        else:
            self.best_score = score
            self.accs = accs
            self.F1 = F1
            self.F2 = F2
            self.F3 = F3
            self.F4 = F4
            self.P1 = P1
            self.P2 = P2
            self.P3 = P3
            self.P4 = P4
            self.R1 = R1
            self.R2 = R2
            self.R3 = R3
            self.R4 = R4
            self.save_checkpoint(val_loss, model,modelname,str)
            self.counter = 0

    def save_checkpoint(self, val_loss, model,modelname,str):
        '''Saves model when validation loss decrease.'''
        # if self.verbose:
        #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_loss_min,val_loss))
        torch.save(model.state_dict(),modelname+str+'.m')
        self.val_loss_min = val_loss
