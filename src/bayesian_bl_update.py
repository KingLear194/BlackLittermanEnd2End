import torch

from aux_utilities import create_idx_list
from train_utils import sub_cov


class BayesBLupdate:
    
    def __init__(self, 
                 tau_view = 0.2,
                 n = (3,3),
                 device = 'cpu', 
                 epsilon = 1e-4
                 ):
        
        self.n = n
        self.device = device
        self.tau_view = torch.tensor(tau_view).to(self.device)
        self.epsilon = epsilon
        
    def set_n(self, n):
        self.n = n
        
    def produce_update_one_block(self,mu_b, cov_b, P, views):
        """
    Compute the BL-updated mean and covariance estimation given view weights and estimated return for one asset block

    Parameters:
        mu_b (torch.Tensor): Mean benchmark return, a 1d-array (tensor) with length equal to n_assets.
        cov_b (torch.Tensor): Benchmark covariance matrix, a symmetric matrix of size n_assets x n_assets (tensor).
        P (torch.Tensor): Portfolio weights matrix of dimension n_assets x n_views (tensor).
        views (torch.Tensor): Mean view returns, a 1d-array (tensor) with length equal to n_views.

    Returns:
        mu_bl (torch.Tensor): The BL expected return vector, a 1d-array (tensor) with length equal to n_assets.
        cov_bl (torch.Tensor): The BL covariance matrix, a symmetric matrix of size n_assets x n_assets (tensor).
        """
        
        sigma_view = torch.matmul(torch.matmul(P.T, cov_b),P)
        try:
            conf_view_inv = torch.linalg.inv(self.tau_view * sigma_view) 
        except:
            conf_view_inv = torch.linalg.pinv(self.tau_view * sigma_view, hermitian = True) 
        try:
            conf_bm_inv = torch.linalg.inv(cov_b)
        except:
            print("I have to use p_inverse for conf_bm_inv")
            eps = self.epsilon*torch.eye(cov_b.shape[0]).to(self.device)
            conf_bm_inv = torch.linalg.pinv(cov_b+eps, hermitian = True)
        try:
            cov_bl = torch.linalg.inv(P @ conf_view_inv @ P.T + conf_bm_inv)
        except:
            print("I have to use p_inverse for cov_bl")
            cov_bl = torch.linalg.pinv(P @ conf_view_inv @ P.T + conf_bm_inv, hermitian = True)
        
        mu_bl = cov_bl @ (P @ conf_view_inv @ views.flatten() + conf_bm_inv @ mu_b)

        return mu_bl, cov_bl
    
    def produce_update(self, mu_b, cov_b, P, views):
        """
        P has shape total_nr_assets x nr_views
        mu_b has shape total_nr_assets
        cov_b has shape total_nr_assets x total_nr_assets and is in block matrix form
        views has shape nr_dates x nr_views
        """
        
        if len(self.n)==1:
            self.updates = [self.produce_update_one_block(mu_b, cov_b, P, views)]
            return
        
        mus = torch.split(mu_b, self.n)
        covs = [sub_cov(cov_b, idx) for idx in create_idx_list(self.n)]
        P_list = torch.split(P, self.n, dim=0)
        views_list = torch.split(views,1,dim=0) 
        self.updates = [self.produce_update_one_block(mus[i], covs[i], P_list[i], views_list[i]) 
                   for i in range(len(self.n))]
                
    def get_BLupdate(self):
        return torch.cat([update[0] for update in self.updates], dim=0), \
                    torch.block_diag(*[update[1] for update in self.updates]) 
    