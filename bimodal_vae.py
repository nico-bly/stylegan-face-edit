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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.optim import Adam
import projector_nb
import numpy as np
import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau


batch_size = 16

network = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with dnnlib.util.open_url(network) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)
        
def generate_samples(G, num_samples):
    z_samples = np.random.RandomState().randn(num_samples, 512).astype(np.float32) 
    
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples.cpu().numpy()
    w_samples = w_samples[:,0,:]

    return w_samples


w_samples_ae = generate_samples(G, 100000)

# Define your data preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ('standard', StandardScaler()),
    ('minmax', MinMaxScaler())
])


w_samples_scaled = preprocessing_pipeline.fit_transform(w_samples_ae)
w_samples_scaled = np.clip(w_samples_scaled, 0, 1)

print(np.min(w_samples_scaled))
print(np.max(w_samples_scaled))
dataset = TensorDataset(torch.tensor(w_samples_scaled, dtype=torch.float32))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

input_dim = 512
hidden_dim = 256
latent_dim = 100
batch_size = 16
epochs = 10
learning_rate = 1e-3


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim, hidden_dim ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim , hidden_dim ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim, latent_dim * 2)  # 2 for mean and variance
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim , hidden_dim ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim , hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

        
    def encode(self, x):
        h = self.encoder(x)
        mean, log_var = torch.chunk(h, 2, dim=1)
        return mean, log_var
    
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        x_recon = self.decode(z)
        return x_recon, mean, log_var

# Define loss function
def loss_function(recon_x, x, mean, log_var):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return BCE + KLD


vae = VAE(input_dim, hidden_dim, latent_dim).to(device)
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)


for epoch in range(epochs):
    for data in dataloader:
        x = data[0].to(device)
        x_recon, mean, log_var = vae(x)

        loss = loss_function(x_recon, x, mean, log_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step(loss)
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

print("Training complete.")

torch.save(vae.state_dict(), 'vae_model.pth')