import torch

from pymindy.training.extended_kalman_filter import ExtendedKalmanFilter
from pymindy.training.extended_kalman_filter_hmm import ExtendedKalmanFilterHMM


class MINDyBPKF(ExtendedKalmanFilter):
    """MINDyBPKF is a class that extends the ExtendedKalmanFilter class. It computes the culmulative error of the Kalman filter and forward simulation stages.

    Attributes:
    -----------
    core : torch.nn.Module
        The core pyMINDy model for neural dynamics.
    warm_up_step : int
        The number of steps for the warm-up stage. (not culmulating error for backpropagation)
    kf_step : int
        The number of steps for the Kalman filter stage. (It includes the warm-up stage)
    rec_step : int
        The number of steps for the forward simulation stage.

    """

    def __init__(self, core, warm_up_step, kf_step, rec_step):
        super().__init__(core)
        self.warm_up_step = warm_up_step
        self.kf_step = kf_step
        self.rec_step = rec_step

    def batch_cul_error_MSE(self, batch, X0_est, P0):
        """ This function uses the vmap function to apply the cul_error_MSE function to each batch.

        Parameters:
        -----------
        batch : torch.Tensor
            The batch of data.
        X0_est : torch.Tensor
            The initial state estimate.
        P0 : torch.Tensor
            The initial covariance matrix.

        Returns:
        --------
        torch.Tensor
            The culmulative error of the Kalman filter and forward simulation stages (excluding the warm-up stage). Each element represents the culmulative error of an element in the batch.
        """
        return torch.func.vmap(self.cul_error_MSE)(batch, X0_est, P0)

    @torch.compile
    def cul_error_MSE(self, Y, X0_est, P0):
        """ This function computes the culmulative error of the Kalman filter and forward simulation stages.

        Parameters:
        -----------
        Y : torch.Tensor
            The data.
        X0_est : torch.Tensor
            The initial state estimate.
        P0 : torch.Tensor
            The initial covariance matrix.

        Returns:
        --------
        torch.Tensor
            A scalar that represents the culmulative error of the Kalman filter and forward simulation stages (excluding the warm-up stage).
        """

        x = X0_est
        P = P0
        i = 0
        loss = 0
        loss_func = torch.nn.MSELoss(reduction="sum")
        for _ in range(self.kf_step):
            x_pred, P_pred = self.predict_with_cov(x, P)
            y_pred= self._func_H(x_pred)
            x, P = self.update(x_pred, y_pred, Y[i, :], P_pred)

            if i >= self.warm_up_step:
                loss += loss_func(y_pred, Y[i, :])
            i += 1
        for _ in range(self.rec_step):
            x_pred= self.predict(x)
            y_pred= self._func_H(x_pred)
            if i >= self.warm_up_step:
                loss += loss_func(y_pred, Y[i, :])
            i += 1
        return loss / (self.kf_step + self.rec_step - self.warm_up_step)

class HMMINDyBPKF(ExtendedKalmanFilterHMM):
    """HMMINDyBPKF is a class that extends the ExtendedKalmanFilter class. It computes the culmulative error of the Kalman filter and forward simulation stages.

    Attributes:
    -----------
    core : torch.nn.Module
        The core pyMINDy model for neural dynamics.
    warm_up_step : int
        The number of steps for the warm-up stage. (not culmulating error for backpropagation)
    kf_step : int
        The number of steps for the Kalman filter stage. (It includes the warm-up stage)
    rec_step : int
        The number of steps for the forward simulation stage.

    """

    def __init__(self, core, warm_up_step, kf_step, rec_step):
        super().__init__(core)
        self.warm_up_step = warm_up_step
        self.kf_step = kf_step
        self.rec_step = rec_step

    def batch_cul_error_MSE(self, batch, X0_est, P0, Gamma_ind_stack):
        """ This function uses the vmap function to apply the cul_error_MSE function to each batch.

        Parameters:
        -----------
        batch : torch.Tensor
            The batch of data.
        X0_est : torch.Tensor
            The initial state estimate.
        P0 : torch.Tensor
            The initial covariance matrix.
        Gamma_ind_stack : torch.Tensor
            The Gamma (modulation matrix) index for the batch of data

        Returns:
        --------
        torch.Tensor
            The culmulative error of the Kalman filter and forward simulation stages (excluding the warm-up stage). Each element represents the culmulative error of an element in the batch.
        """
        return torch.func.vmap(self.cul_error_MSE)(batch, X0_est, P0, Gamma_ind_stack)

    @torch.compile
    def cul_error_MSE(self, Y, X0_est, P0, Gamma_ind_stack):
        """ This function computes the culmulative error of the Kalman filter and forward simulation stages.

        Parameters:
        -----------
        Y : torch.Tensor
            The data.
        X0_est : torch.Tensor
            The initial state estimate.
        P0 : torch.Tensor
            The initial covariance matrix.
        Gamma_ind_stack : torch.Tensor
            The Gamma (modulation matrix) index for the batch of data

        Returns:
        --------
        torch.Tensor
            A scalar that represents the culmulative error of the Kalman filter and forward simulation stages (excluding the warm-up stage).
        """
        x = X0_est
        P = P0
        i = 0
        loss = 0
        loss_func = torch.nn.MSELoss(reduction="sum")
        for _ in range(self.kf_step):
            x_pred, P_pred = self.predict_with_cov(x, P, Gamma_ind_stack[i])
            y_pred= self._func_H(x_pred)
            x, P = self.update(x_pred, y_pred, Y[i, :], P_pred)

            if i >= self.warm_up_step:
                loss += loss_func(y_pred, Y[i, :])
            i += 1
        for _ in range(self.rec_step):
            x_pred= self.predict(x, Gamma_ind[i])
            y_pred= self._func_H(x_pred)
            if i >= self.warm_up_step:
                loss += loss_func(y_pred, Y[i, :])
            i += 1
        return loss / (self.kf_step + self.rec_step - self.warm_up_step)

    def warm_up_kalman_smooth(self, data, X_est, P0_est, Gamma_ind_hist):
        x = X_est
        P = P0_est
        i = 0
        Gamma_offset = 0
        x_pred_hist = torch.zeros((X_est.shape[0], X_est.shape[1], self.warm_up_step),device=self.core.device)
        P_pred_hist = torch.zeros((X_est.shape[0], X_est.shape[1], X_est.shape[1], self.warm_up_step),device=self.core.device)
        x_smooth = torch.zeros((X_est.shape[0], X_est.shape[1]))
        P_smooth = torch.zeros((X_est.shape[0], X_est.shape[1], X_est.shape[1]))
        print(Gamma_ind_hist.shape)
        
        for _ in range(self.warm_up_step):
            x_pred, P_pred = torch.func.vmap(self.predict_with_cov)(x,P,Gamma_ind_hist[Gamma_offset:,:])
            if i > 0:
                x_pred = x_pred[:-1,:]
                P_pred = P_pred[:-1,:]
                Gamma_offset += 1
            y_pred = torch.func.vmap(self._func_H)(x_pred)
            x, P = torch.func.vmap(self.update)(x_pred, y_pred, data[i:,:], P_pred)
            x_pred_hist[i:,:,i] = x
            P_pred_hist[i:,:,:,i] = P
            i += 1

        #print(x_pred_hist.shape)
        x_prev = x
        P_prev = P
        i -= 1
        for _ in range(self.warm_up_step,1,-1):
            x, P = torch.func.vmap(self.smooth)(x_prev, x_pred_hist[i:,:,i-1], x_pred_hist[i:,:,i], P_prev, P_pred_hist[i:,:,:,i-1], P_pred_hist[i:,:,:,i], Gamma_ind_hist[i:,:])
            if i > 0:
                x_prev = torch.vstack([x, x_prev[-(self.warm_up_step-i),:]])
                P_prev = torch.vstack([P, P_prev[-(self.warm_up_step-i),:,:][None,:,:]])
            else:
                x_prev = x
                P_prev = P
            x_smooth[i-1:,:] = x_prev
            P_smooth[i-1:,:,:] = P_prev
            i -= 1
        return x_smooth, P_smooth

    def kf_full_data(self, data, X0, P0, Gamma_ind_hist):
        T = data.shape[0]
        #x_pred = torch.zeros((X0.shape[0],T)).to(self.device)
        #P_pred = torch.zeros((P0.shape[0],P0.shape[1],T)).to(self.device)
        #x_update = torch.zeros((X0.shape[0],T)).to(self.device)
        #P_update = torch.zeros((P0.shape[0],P0.shape[1],T)).to(self.device)
        post_resid = torch.zeros((T,)).to(self.device)
        #x_smooth = torch.zeros((X0.shape[0],T)).to(self.device)
        #P_smooth = torch.zeros((P0.shape[0],P0.shape[1],T)).to(self.device)
        #x_pred[:,0] = X0
        x_pred = X0
        #P_pred[:,:,0] = P0
        P_pred = P0
        y_pred = self._func_H(X0)
        #x_update[:,0], P_update[:,:,0] = self.update(x_pred[:,0],y_pred,data[0,:],P0)
        x_update, P_update = self.update(x_pred,y_pred,data[0,:],P0)
        y_update = self._func_H(x_update)
        post_resid[0] = (data[0,:] - y_update)[None,:] @ (data[0,:] - y_update)[:,None]

        # Forward Pass
        for t in range(1,T):
            #x_pred[:,t], P_pred[:,:,t] = self.predict_with_cov(x_update[:,t-1],P_update[:,:,t-1],Gamma_ind_hist[t-1])
            x_pred, P_pred = self.predict_with_cov(x_update,P_update,Gamma_ind_hist[t-1])
            y_pred = self._func_H(x_pred)
            #x_update[:,t], P_update[:,:,t] = self.update(x_pred[:,t],y_pred,data[t,:],P_pred[:,:,t])
            x_update, P_update = self.update(x_pred,y_pred,data[t,:],P_pred)
            y_update = self._func_H(x_update)
            post_resid[t] = (data[t,:] - y_update)[None,:] @ (data[t,:] - y_update)[:,None]
            
        # Backward Pass
        #x_smooth[:,-1] = x_update[:,T-1]
        #P_smooth[:,:,-1] = P_update[:,:,T-1]
        #for t in range(T-2,-1,-1):
        #    x_smooth[:,t], P_smooth[:,:,t] = self.smooth(x_smooth[:,t+1],x_update[:,t],x_pred[:,t+1],P_smooth[:,:,t+1],P_update[:,:,t],P_pred[:,:,t+1],Gamma_ind_hist[t])
        x_smooth = 0
        P_smooth = 0
        
        return x_smooth, P_smooth, post_resid