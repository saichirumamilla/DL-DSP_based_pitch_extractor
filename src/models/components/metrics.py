from typing import Tuple

import csv
import torch

from torchmetrics import Metric

class RCAMetric(Metric):
    num_correct_frames_voiced: int
    num_total_frames_voiced: int
    per_utterance_rca: list
    per_utterance_diff: list
    
    def __init__(self, threshold=50, test_mode=True):
        super().__init__()
        self.test_mode = test_mode
        self.add_state('num_correct_frames_voiced', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('num_total_frames_voiced', default=torch.tensor(0), dist_reduce_fx='sum')
        self.threshold = threshold
        self.per_utterance_rca = [] 

    def hz_to_cent(self, f_hz: torch.Tensor):
        reference_frequency = 62.5
        return 1200 * torch.log2(f_hz / reference_frequency)

    def compute(self)-> torch.Tensor:
        rca = self.num_correct_frames_voiced / self.num_total_frames_voiced
        return rca
    
    def ret_per_utterance_rca(self):
        return torch.Tensor(self.per_utterance_rca)
    
    def compute_rca_per_frame(
            self,
            y_pred: torch.Tensor,
            y: torch.Tensor,
            probs: torch.Tensor,
            return_type: type = torch.Tensor
        ) -> Tuple[list | torch.Tensor, torch.Tensor]:
        mask = (probs > 0)
        y_pred_cent = self.hz_to_cent(y_pred)
        y_cent = self.hz_to_cent(y)
        diff = torch.abs(y_pred_cent - y_cent)
        correct_frames = (diff < self.threshold)
        correct_frames = torch.where(mask, correct_frames, False)
        if return_type == torch.Tensor:
            return correct_frames, mask
        elif return_type == list:
            return list(correct_frames.cpu().numpy()), mask.cpu().numpy()
        
    def update(self, y_pred: torch.Tensor, y: torch.Tensor, probs: torch.Tensor)-> list:
        '''
            y_pred (torch.Tensor): Predicted pitch values in Hz
            y (torch.Tensor): Ground truth pitch values in Hz
            probs (torch.Tensor): Voicing probabilities
        '''
        correct_frames, mask = self.compute_rca_per_frame(y_pred, y, probs, return_type=torch.Tensor)

        self.num_correct_frames_voiced += torch.sum(correct_frames) # Add the number of correct frames to the total
        self.num_total_frames_voiced += mask.long().sum()  # Add the total number of frames to the total
        
        if self.test_mode:
            # Compute the RCA per utterance
            correct_per_utterance = torch.sum(correct_frames, dim=1) / correct_frames[0].numel()
            self.per_utterance_rca += list(correct_per_utterance)




    





