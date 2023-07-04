import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
#other for PDiags:
import math
import numpy as np
import scipy
import matplotlib.animation as animation

## install and import the persistence diagram constructor:
!pip install git+https://github.com/HuPBA/fast_zero_dimensional_persistence_diagrams
import zero_persistence_diagram

## next 3 fctns for topological loss:
def get_indices_from_condensed_index(condensed_index, number_of_points):
    b = 1 - (2 * number_of_points)
    i = int((-b - math.sqrt(b ** 2 - 8 * condensed_index)) // 2)
    j = condensed_index + i * (b + i + 2) // 2 + 1
    return i, j

def compute_persistence_diagram(point_cloud):
    # Compute euclidean distance matrix for points in the point cloud
    distance_matrix = torch.cdist(point_cloud, point_cloud, p=2)
    # Compute the persistence diagram without backprop
    with torch.no_grad():
        condensed_distance_matrix = scipy.spatial.distance.squareform(distance_matrix.detach(), checks=False)
        pd, condensed_pairs = zero_persistence_diagram.zero_persistence_diagram_by_single_linkage_algorithm(
            condensed_distance_matrix)
    pairs = torch.tensor([get_indices_from_condensed_index(condensed_index, point_cloud.shape[0]) for condensed_index in
                          condensed_pairs])
    # Filter the distance matrix to have the pairs we want
    pd_according_to_pairs = distance_matrix[pairs[:, 0], pairs[:, 1]]
    return pd_according_to_pairs

def loss_topo(point_cloud):
    pd_according_to_pairs = compute_persistence_diagram(point_cloud)
    # Compute the loss. In this case, we minimise the total persistence of the 
    # persistence diagram.
    return torch.sum(pd_according_to_pairs)

## FCGAN:
class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64
image_dim = 28 * 28 * 1  # 784
batch_size = 32
num_epochs = 50

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()
w_topo = 0.5
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        opt_disc.zero_grad()
        opt_gen.zero_grad()
        
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        output = disc(fake).view(-1)
        
        ##ADDED: topological loss:
        noise_for_pd = torch.randn(100, z_dim).to(device)
        point_cloud = gen(noise_for_pd).to(device) ## 100 points of z_dim(=64) dimensions
        
        lossG = criterion(output, torch.ones_like(output)) + w_topo * loss_topo(point_cloud)
        gen.zero_grad()
        
        lossG.backward()
        opt_gen.step()

        if batch_idx % 500 == 0:
            print("epoch", epoch)

            ### new:
            with torch.no_grad():
              fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
              fake = fake.to(device)
              grid_img = torchvision.utils.make_grid(fake[:32], nrow=8, normalize=True)  # Create a grid of images
              # Convert the grid tensor to numpy array and transpose the dimensions
              grid_img = grid_img.cpu().numpy().transpose((1, 2, 0))

              # Display the grid of images
              plt.figure(figsize=(10, 10))
              plt.imshow(grid_img)
              plt.axis('off')
              plt.show()

              step += 1
