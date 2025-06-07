import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv, GATConv
from torch_geometric.nn import global_add_pool


# Decoders 
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        mlp_layers = [nn.Linear(latent_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers-2)]
        mlp_layers.append(nn.Linear(hidden_dim, 2*n_nodes*(n_nodes-1)//2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.relu(self.mlp[i](x))
        
        x = self.mlp[self.n_layers-1](x)
        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=1, hard=True)[:,:,0]

        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:,idx[0],idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)
        return adj

class ConditionalDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes, cond_dim):
        super(ConditionalDecoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        # First layer takes latent_dim + cond_dim as input
        mlp_layers = [nn.Linear(latent_dim + cond_dim, hidden_dim)]
        mlp_layers += [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 2)]
        mlp_layers.append(nn.Linear(hidden_dim, 2 * n_nodes * (n_nodes - 1) // 2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, cond):
        x = torch.cat([x, cond], dim=1)
        for i in range(self.n_layers - 1):
            x = self.relu(self.mlp[i](x))

        x = self.mlp[self.n_layers - 1](x)
        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=1, hard=True)[:, :, 0]

        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:, idx[0], idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)
        return adj

#Encoders 
class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            ))                        
        for layer in range(n_layers-1):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            )) 

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)

        out = global_add_pool(x, data.batch)
        out = self.bn(out)
        out = self.fc(out)
        return out

# GAT encoder
class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, heads=4, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.heads = heads

        self.convs = nn.ModuleList()

        # First GAT layer
        self.convs.append(
            GATConv(
                in_channels=input_dim,
                out_channels=hidden_dim // heads,
                heads=heads,
                concat=True
            )
        )

        # Intermediate GAT layers
        for _ in range(n_layers - 1):
            self.convs.append(
                GATConv(
                    in_channels=hidden_dim,  
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    concat=True
                )
            )

        # Project to latent_dim in the final step
        self.fc = nn.Linear(hidden_dim, latent_dim)

        # Batch normalization
        self.bn = nn.BatchNorm1d(latent_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.leaky_relu(x, 0.2)
            x = F.dropout(x, self.dropout, training=self.training)

        # Global pooling
        out = global_add_pool(x, data.batch)  # Shape: [batch_size, hidden_dim]

        # Project to latent space
        out = self.fc(out)  # Shape: [batch_size, latent_dim]

        # Apply batch normalization
        out = self.bn(out)

        return out

# Autoencoders 

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes):
        super(AutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        self.encoder = GIN(input_dim, hidden_dim_enc, latent_dim, n_layers_enc)
        
        self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)

    def forward(self, data):
        x_g = self.encoder(data)
        adj = self.decoder(x_g) 
        return adj



    def encode(self, data):
        x_g = self.encoder(data)
        return x_g

    def decode(self, x_g):
        adj = self.decoder(x_g)
        return adj
    def decode_mu(self, mu):
       adj = self.decoder(mu)
       return adj

    def loss_function(self, data):
        x_g  = self.encoder(data)
        adj = self.decoder(x_g)
        loss = F.binary_cross_entropy(adj, data.A, reduction='mean')
        return loss, loss, torch.tensor(0.)


# Autoencoder with Conditioning in Decoder
class ConditionalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes, cond_dim):
        super(ConditionalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        self.encoder = GIN(input_dim, hidden_dim_enc, latent_dim, n_layers_enc)
        self.decoder = ConditionalDecoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes, cond_dim)

    def forward(self, data):
        x_g = self.encoder(data)  # Encode graph
        cond = data.stats  # Conditioning vector of size (batch_size, cond_dim)
        adj = self.decoder(x_g, cond)  # Decode graph with conditioning
        return adj

    def encode(self, data):
        x_g = self.encoder(data)
        return x_g

    def decode(self, x_g, cond):
        adj = self.decoder(x_g, cond)
        return adj

    def decode_mu(self, mu, cond):
        adj = self.decoder(mu, cond)
        return adj

    def loss_function(self, data):
        x_g = self.encoder(data)
        cond = data.stats  # Conditioning vector

        adj = self.decoder(x_g, cond)
        loss = F.binary_cross_entropy(adj, data.A, reduction='mean')
        return loss, loss, torch.tensor(0.)


# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes):
        super(VariationalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        self.encoder = GIN(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)

    def forward(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)
        return adj

    def encode(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        return x_g

    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, mu, logvar):
       x_g = self.reparameterize(mu, logvar)
       adj = self.decoder(x_g)
       return adj

    def decode_mu(self, mu):
       adj = self.decoder(mu)
       return adj

    def loss_function(self, data, beta=0.05):
        x_g  = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)
        
        recon = F.l1_loss(adj, data.A, reduction='mean')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + beta*kld

        return loss, recon, kld

