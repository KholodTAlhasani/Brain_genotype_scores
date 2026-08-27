"""
Loss function of the Brain_SNP_NN.
has the parameter:
initial_weights: calculated in the main script, it is the initial weights of the classes,
dynamic_tau: if True it will use CDB loss, if False it will use the normal CrossEntropyLoss.

Created by Kholod Alhasani
"""
import torch
from torch import nn

class MaskedLoss(nn.Module):
    def __init__(self, class_weights=None):
        super(MaskedLoss, self).__init__()
        self.class_weights = class_weights
        self.loss_func = nn.CrossEntropyLoss(reduction='none')  # Default loss without weights
    
    def forward(self, output, target, mask, task_id=None):
        device = output.device
        target = target.to(device)
        mask = mask.to(device)
        
        valid = (target >= 0) & (mask.bool()) # Combine both filters: valid label + mask
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device)
        valid_output = output[valid]  # Apply the valid_indices to both the output and target tensors
        valid_target = target[valid]

        # Select the appropriate weights for the current task
        if self.class_weights is not None and task_id is not None and task_id in self.class_weights:
            current_weights = self.class_weights[task_id].to(device)
            loss = nn.CrossEntropyLoss(weight=current_weights, reduction='none')(valid_output, valid_target)
        else:
            # Calculate the cross-entropy loss without reduction
            loss = self.loss_func(valid_output, valid_target)  # Use the default loss function if no weights are specified

        return loss