## Modification of StyleGAN2-ADA &mdash;

This project aims to explore the latent space of stylegan model to edit face images

### Unsupervised methods for analysis

- PCA

The PCA gives a first insight of the structure of the latent space, and we learn that it is quite organized.
When moving along a direction, there is a modification of characteristics such as gender, pose, glasses, hair
Stylegan2 structure of latent space enables a layer wise modification to isolate some attriubutes.

Problem: some features are entangled: if we modify gender, this modifies the hair lenght for instance

Can other unsupervised methods allow disantanglement?

- KPCA with a RBF Kernel

In some specific cases allow some disantanglement

- Variational auto encoder

An other way to capture more complex relationship in the latent space is to modify it with an autoencoder

### Supervised method: InterfaceGAN

Generation of 50k latent code and classifciation of resulting images on the presence of different attributes on the generated image.

SVM on these attributesto find separation in the latent space and moving on the normal vector of this hyperplan
