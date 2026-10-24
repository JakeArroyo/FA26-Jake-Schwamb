"""ExtendedKalmanFilterHMM: the EKF with HMM-modulated state dynamics.

STUB (reconstructed): backprop_kalman_filter.HMMINDyBPKF subclasses this, but
the module was not carried into this copy of the repo.  Identical to
ExtendedKalmanFilter except that the state map is modulation-state dependent,
so ``predict``, ``predict_with_cov`` and ``smooth`` each take the current
Gamma (modulation) index.  The observation map does not depend on it.

Signatures are fixed by the call sites in HMMINDyBPKF: predict(x, ind),
predict_with_cov(x, P, ind), update(x_pred, y_pred, y, P_pred) and
smooth(x_next, x_upd, x_pred_next, P_next, P_upd, P_pred_next, ind).
"""

import torch

from pymindy.training.extended_kalman_filter import ExtendedKalmanFilter


class ExtendedKalmanFilterHMM(ExtendedKalmanFilter):
    def _func_F(self, x, mod_ind):
        return self.core.func_F(x, mod_ind)

    def _func_dFdx(self, x, mod_ind):
        return self.core.func_dFdx(x, mod_ind)

    def predict(self, x, mod_ind):
        return self._func_F(x, mod_ind)

    def predict_with_cov(self, x, P, mod_ind):
        x_pred = self._func_F(x, mod_ind)
        A = self._func_dFdx(x, mod_ind)
        P_pred = A @ P @ A.transpose(-1, -2) + self.core.Q
        return x_pred, P_pred

    def smooth(self, x_smooth_next, x_update, x_pred_next,
               P_smooth_next, P_update, P_pred_next, mod_ind):
        A = self._func_dFdx(x_update, mod_ind)
        M = P_update @ A.transpose(-1, -2)
        J = torch.linalg.solve(P_pred_next.transpose(-1, -2), M.transpose(-1, -2)).transpose(-1, -2)
        x_smooth = x_update + J @ (x_smooth_next - x_pred_next)
        P_smooth = P_update + J @ (P_smooth_next - P_pred_next) @ J.transpose(-1, -2)
        return x_smooth, P_smooth
