import gpytorch
import torch

class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, nout):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([nout]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([nout])),
            batch_shape=torch.Size([nout])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, nout, rank=None):
        '''
        Parameters
        ----------
        nout : int
            number of outputs
        '''
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)

        if rank is None:
            rank = nout

        self.mean_module = gpytorch.means.MultitaskMean(
            # gpytorch.means.ConstantMean(), num_tasks=nout
            gpytorch.means.ZeroMean(), num_tasks=nout
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]), num_tasks=nout, rank=rank
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

class IndependentGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        '''
        Parameters
        ----------
        nout : int
            number of outputs
        '''
        super(IndependentGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu = 1.5, ard_num_dims=train_x.shape[1]))
        # self.covar_module = gpytorch.kernels.MaternKernel(nu = 1.5, ard_num_dims=train_x.shape[1])
        # self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
