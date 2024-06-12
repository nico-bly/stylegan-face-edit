
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
importlib.reload(projector_nb)
import numpy as np
import torch
import os
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt



def generate_samples(G, num_samples):
    z_samples = np.random.RandomState().randn(num_samples, 512).astype(np.float32) 
    
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples.cpu().numpy()
    w_samples = w_samples[:,0,:]

    return w_samples

def get_pca(samples):
    
    scaler = StandardScaler()
    w_samples = scaler.fit_transform(samples)  

    pca = PCA(n_components=100)
    pca.fit(w_samples)

    explained_variance = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance)

    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center',
        label='Individual Explained Variance')
    plt.step(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, where='mid',
         label='Cumulative Explained Variance')

    plt.xlabel('Principal Component Index')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    return pca , scaler


def move_according_to_component(G, device, pca, outdir, directions,  layers, num_steps, save_video=True, title_vid = 'proj', random_seed=None):
    os.makedirs(outdir, exist_ok=True)

    if random_seed is not None:
        np.random.seed(random_seed)

    explained_variance = pca.explained_variance_
    V = pca.components_.T
   
    # Generate a random z vector
    #z_sample = np.random.RandomState().randn(1, 512).astype(np.float32)  # select a random z vector
    z_sample = np.random.RandomState(random_seed).randn(1, 512).astype(np.float32) 
    w_sample = G.mapping(torch.from_numpy(z_sample).to(device), None)

    # Generate the original image
    img_original = G.synthesis(w_sample, noise_mode='const')
    img_original = (img_original + 1) * (255 / 2)
    img_original = img_original.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    '''
    plt.imshow(img_original)
    plt.title('Original image')
    plt.axis('off')
    plt.show()
    '''

    # Convert w_sample to numpy and select the second element in the latent vector (w space)
    w_sample = w_sample.cpu().numpy()
    w_sample = w_sample[:, 1, :]  
    w_sample = scaler.transform(w_sample)

    img_list = []
    w_list = []

    # Initialize x
    x = np.zeros((V.shape[1], 1))
    X =  np.tile(x, (1, num_steps))

    # We use X to store the variations according to the selected dimension in one matrix 
    # as we dont move the same way on each direction

    for v in directions:
        sigma = np.sqrt(explained_variance[v])
        start_value = -2* sigma
        end_value = 2* sigma
        arrays_between = np.linspace(start_value, end_value, num_steps + 2)[1:-1]
        X[v] = arrays_between
    
   
    # each line of X correspond to a direction, the columns represent the range from -2 sigma
    # to 2 sigma
    # So the vector x of the article is one of the column of X
    
    # X.shape[0] is the number of componets
    # X.shape[1] is the number of steps from -2 sigma to 2 sigma
    
    for i in range(X.shape[1]):
        x_tmp = X[:,i]
        movement = V @ x_tmp
        movement = np.expand_dims(movement, axis=1)
      
        w_new = w_sample.T + movement
        w_new = w_new.T
        w_new = scaler.inverse_transform(w_new)
        W = np.tile(scaler.inverse_transform(w_sample), (1, 18, 1))
        W[:,layers,:] = w_new
        w_list.append(W)

    '''
    for scalar in tqdm(arrays_between):
        w_pca = pca.transform(w_sample)
        w_pca[:, component_index] += scalar
        w_new_single = pca.inverse_transform(w_pca)

        w_new = np.tile(w_new_single[:, np.newaxis, :], (1, 18, 1))
        w_list.append(w_new)
    '''
    video = imageio.get_writer(f'{outdir}/{title_vid}.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
    for w_modified in w_list:
            w_modified_tensor = torch.from_numpy(w_modified).to(device)
            synth_image = G.synthesis(w_modified_tensor, noise_mode='const')
            synth_image = (synth_image + 1) * (255 / 2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            img_list.append(synth_image)
        

    for synthetic_image in img_list:
        video.append_data(np.concatenate([img_original, synthetic_image], axis=1))

    if save_video: 
        print("Editing video ...")
        os.makedirs(outdir, exist_ok=True)
        print(f'Saving optimization progress video "{outdir}/{title_vid}.mp4"')
        
        video.close()
        print("Video saved")
    
    return img_list, X