"""PartitionData: the torch Dataset that slices a run of data into KF stacks.

STUB (reconstructed): pymindy.training.utils imports ``PartitionData`` from
pymindy.dataset, but the class itself was not carried into this copy of the
repo.  Its contract is fixed by how train_*_data_MSE consumes the loader:

    for stack in train_loader:          # stack: (batch, stack_len, n_cols)
        data_ind_stack = stack[:, :, -1].type(torch.int32)   # time index
        stack          = stack[:, :, :-1]                    # observations

so one dataset item must be a float tensor of shape
``(kf_step + rec_step, n_cols)`` whose rows are consecutive timepoints and
whose last column is the row's index into the full run (the caller appends
that column before training; it is used to look up Gamma_ind_hist).

``stack_interval`` is read here as the stride *between* stacks -- each stack is
contiguous in time (as a Kalman filter over one-step dynamics requires) and
successive stacks start ``stack_interval`` samples apart.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class PartitionData(Dataset):
    def __init__(self, data, stack_interval, kf_step, rec_step):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        self.data = data.type(torch.float)
        self.stack_interval = int(stack_interval)
        self.stack_len = int(kf_step) + int(rec_step)

        n_time = self.data.shape[0]
        if n_time < self.stack_len:
            raise ValueError(
                f"data has {n_time} timepoints, fewer than the "
                f"{self.stack_len} needed for one kf_step+rec_step stack")
        self._len = (n_time - self.stack_len) // self.stack_interval + 1

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if idx < 0:
            idx += self._len
        if not 0 <= idx < self._len:
            raise IndexError(idx)
        start = idx * self.stack_interval
        return self.data[start:start + self.stack_len, :]
