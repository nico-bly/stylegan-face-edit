import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import generate_nb
import matplotlib.pyplot as plt
import torch
import numpy as np
import dnnlib
import legacy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import importlib

import projector_nb
import numpy as np
import torch
import os
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler



############## BETA VAE WITH PCA LOSS

network = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
with dnnlib.util.open_url(network) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)

def generate_samples(G, num_samples):
    z_samples = np.random.RandomState().randn(num_samples, 512).astype(np.float32) 
    
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples.cpu().numpy()
    w_samples = w_samples[:,0,:]

    return w_samples


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # Output mean and logvar
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mean, logvar = h[:, :latent_dim], h[:, latent_dim:]
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decode(z)
        return x_recon, mean, logvar, z

def loss_function(recon_x, x, mean, logvar, z, alpha=1.0):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    
    # Bimodal prior
    prior_mean1 = torch.zeros_like(mean)
    prior_mean2 = torch.zeros_like(mean)
    prior_var1 = torch.ones_like(logvar)
    prior_var2 = torch.ones_like(logvar)
    
    # Log probabilities
    log_p1 = -0.5 * torch.sum((z - prior_mean1)**2 / prior_var1 + torch.log(prior_var1), dim=1)
    log_p2 = -0.5 * torch.sum((z - prior_mean2)**2 / prior_var2 + torch.log(prior_var2), dim=1)
    
    # Log-sum-exp trick to avoid numerical instability
    max_log_p = torch.max(log_p1, log_p2)
    log_prior = max_log_p + torch.log(torch.exp(log_p1 - max_log_p) + torch.exp(log_p2 - max_log_p))
    
    # Ensure no negative values are passed to torch.log
    if torch.any(logvar <= 0):
        print("Logvar contains non-positive values")
    
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
    kl_loss = torch.sum(kl_loss - log_prior)
    
    return recon_loss + alpha * kl_loss

# Parameters
input_dim = 512
hidden_dim = 512
latent_dim = 512
batch_size = 16
epochs = 15
learning_rate = 1e-2


w_samples_ae = generate_samples(G, 100000)

# preprocessing pipeline: standard scaler and then MinMax sclaer (value 0,1)
preprocessing_pipeline = Pipeline([
    ('minmax', MinMaxScaler()),
    ('standard', StandardScaler())])

w_samples_scaled = preprocessing_pipeline.fit_transform(w_samples_ae)

# Create DataLoader
dataset = TensorDataset(torch.tensor(w_samples_scaled, dtype=torch.float32))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model and optimizer

vae = VAE(input_dim, hidden_dim, latent_dim).to(device)
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Training loop
for epoch in range(epochs):
    vae.train()
    train_loss = 0
    for data in dataloader:
        x = data[0].to(device)
        optimizer.zero_grad()
        x_recon, mean, logvar, z = vae(x)
        loss = loss_function(x_recon, x, mean, logvar, z)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(dataloader)
    print(f'Epoch {epoch + 1}, Loss: {train_loss:.4f}')
    #scheduler.step(train_loss)

print("Training complete.")
torch.save(vae.state_dict(), 'vae_bimodal_model.pth')
