import torch
from torch import Tensor

def raw_pitch_accuracy(y_pred, y, threshold=50, only_voiced=False):
    # checks if predicted pitch is within a threshold cent of the ground truth.
    # result is an array of ints with value either 0 or 1 for each frame
    f0_est_cents: Tensor = (y_pred['f0'] + 1).log2_()
    f0_cents: Tensor = (y['f0'] + 1).log2_() # adding +1 to prevent log(0)
    return_array = (torch.abs(f0_est_cents - f0_cents) < (threshold / 1200)).int()
    
    if only_voiced:
        return_array[y_pred['is_voiced'] < 0.5] = -1
        return_array[y['is_voiced'] < 0.5] = -1
        return_array = return_array[~torch.any(return_array == -1, dim=1)]
    return return_array

