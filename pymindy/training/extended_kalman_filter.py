"""ExtendedKalmanFilter: EKF predict/update/smooth over an AbstractMINDyCore.

STUB (reconstructed): backprop_kalman_filter.MINDyBPKF subclasses this, but
the module was not carried into this copy of the repo.  The method set and
signatures are fixed by the call sites in MINDyBPKF.cul_error_MSE:
``predict``, ``predict_with_cov``, ``update``, ``_func_H``, ``self.core``.

These are the textbook EKF equations with the core supplying the nonlinear
state map (``func_F``/``func_dFdx``), the observation map
(``func_H``/``func_dHdx``) and the noise covariances (``Q``/``R``).

Everything runs inside torch.func.vmap and torch.compile with gradients
flowing through the core's parameters, so: no in-place writes, no explicit
matrix inverse (torch.linalg.solve is vmap- and autograd-friendly), and no
data-dependent control flow.
"""

import torch


class ExtendedKalmanFilter:
    def __init__(self, core):
        self.core = core

    @property
    def device(self):
        return self.core.device

    # --- core maps (thin wrappers; the *BPKF subclasses call _func_H) ---
    def _func_F(self, x):
        return self.core.func_F(x)

    def _func_H(self, x):
        return self.core.func_H(x)

    def _func_dFdx(self, x):
        return self.core.func_dFdx(x)

    def _func_dHdx(self, x):
        return self.core.func_dHdx(x)

    # --- EKF steps ---
    def predict(self, x):
        """State prediction only (used by the free-simulation / rec_step stage)."""
        return self._func_F(x)

    def predict_with_cov(self, x, P):
        """State + covariance prediction.  P_pred = A P A' + Q."""
        x_pred = self._func_F(x)
        A = self._func_dFdx(x)
        P_pred = A @ P @ A.transpose(-1, -2) + self.core.Q
        return x_pred, P_pred

    def update(self, x_pred, y_pred, y, P_pred):
        """Measurement update.  K = P_pred H' (H P_pred H' + R)^-1."""
        H = self._func_dHdx(x_pred)
        PHt = P_pred @ H.transpose(-1, -2)
        S = H @ PHt + self.core.R
        # K = PHt @ S^-1, solved as (S' \ PHt')' to avoid forming an inverse.
        K = torch.linalg.solve(S.transpose(-1, -2), PHt.transpose(-1, -2)).transpose(-1, -2)
        x = x_pred + K @ (y - y_pred)
        eye = torch.eye(P_pred.shape[-1], dtype=P_pred.dtype, device=P_pred.device)
        P = (eye - K @ H) @ P_pred
        return x, P

    def smooth(self, x_smooth_next, x_update, x_pred_next,
               P_smooth_next, P_update, P_pred_next):
        """One Rauch-Tung-Striebel backward step.  J = P_update A' P_pred_next^-1."""
        A = self._func_dFdx(x_update)
        M = P_update @ A.transpose(-1, -2)
        J = torch.linalg.solve(P_pred_next.transpose(-1, -2), M.transpose(-1, -2)).transpose(-1, -2)
        x_smooth = x_update + J @ (x_smooth_next - x_pred_next)
        P_smooth = P_update + J @ (P_smooth_next - P_pred_next) @ J.transpose(-1, -2)
        return x_smooth, P_smooth
