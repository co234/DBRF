import torch
from torch.nn import Module, Linear, ReLU, Dropout,Sigmoid, Softmax
from mlp_vae import *
import torch.nn.init as init

# SAMME DISCRIMINATOR STRUCTURE AS FACTORVAE
class Discriminator(nn.Module):
    def __init__(self, z_dim):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 50),
            nn.LeakyReLU(0.1, True),
            nn.Linear(50, 50),
            nn.LeakyReLU(0.1, True),
            nn.Linear(50, 50),
            nn.LeakyReLU(0.1, True),
            nn.Linear(50, 2)
        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def forward(self, z):
        return self.net(z).squeeze()



def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


if __name__ == "__main__":
    b = torch.distributions.Uniform(low=0,high=1).sample((2000,1))
    z = torch.distributions.Normal(1,1).sample((2000,20))
    print(b.size(0))

    latent = torch.cat([z, b], dim=1)


    d = Discriminator(latent.shape[1])
    d_ = d(latent)
    print(d_)
    x1 = d_[:,:1]
    x2 = d_[:,1:]
    print(x1)
    print(x2)
    print(x1.data[:,:1].numpy())
    print(x1.clone().detach())
    print(x2.clone().detach())
    # (d_[:, :1] - d_[:, 1:]).mean()