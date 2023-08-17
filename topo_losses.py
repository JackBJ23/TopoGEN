# -*- coding: utf-8 -*-
"""topo_losses.ipynb

Original file is located at
    https://colab.research.google.com/drive/1X-kZOdGD0bSEEBOFHcxysUyCiwUF-3eM
"""

# ripser_parallel: https://giotto-ai.github.io/giotto-ph/build/html/modules/ripser_parallel.html
# bottleneck dist: https://persim.scikit-tda.org/en/latest/notebooks/distances.html

!pip3 install giotto-ph
import numpy as np
from gph import ripser_parallel

!pip install ipython

from IPython.display import Image  # to display images
import sys
!{sys.executable} -m pip install giotto-tda

# here comes our protagonist!
from gph import ripser_parallel

# Import utils
import numpy as np
from gtda.homology._utils import _postprocess_diagrams

# To generate dataset
from sklearn import datasets

# Plotting
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from plotly import graph_objects as go
from gtda.plotting import plot_diagram, plot_point_cloud

!pip3 install Cython ripser tadasets
!pip3 install persim #provides matching
#used this: https://persim.scikit-tda.org/en/latest/notebooks/distances.html
import persim
import tadasets
import ripser
#other packages:
import math
import scipy
import torch
import random

"""Bottleneck0 loss fctn: loss_topo0, with auxiliary loss in push0 when dist is dgm2 matched to diagonal.  """

#euclidean dist for torch tensors:
def dist(point1, point2):
    return torch.sqrt(torch.sum((point2 - point1)**2))

#euclidean dist for numpy points:
def dist_np(point1, point2):
    return np.sqrt(np.sum((point2 - point1)**2))

#supremum dist for torch tensors:
def dist_sup_tc(b1, d1, b2, d2):
    # Calculate the sup norm between points (b1, d1) and (b2, d2)
    return torch.max(torch.abs(b1 - b2), torch.abs(d1 - d2))

#auxiliary loss when d(D,D0) (in deg0) only depends on D0 (so gradients are 0):
def push0(point_cloud): #code of compute_persistence_diagram here, but the sum is *-1 so backprop pushes points out from the diag
   # Compute the persistence diagram without backprop
    with torch.no_grad():
        ##convert points for computing PD:
        points_np = point_cloud.numpy()
        # get PD with generators:
        dgm = ripser_parallel(points_np, maxdim=0, return_generators=True)

    p1, p2 = point_cloud[dgm['gens'][0][0][1]], point_cloud[dgm['gens'][0][0][2]]
    loss = -dist(p1, p2) #dropped here the /2. (real dist to diag is dist(p1, p2)/2.)
    for i in range(1, len(dgm['gens'][0])):
      p1, p2 = point_cloud[dgm['gens'][0][i][1]], point_cloud[dgm['gens'][0][i][2]] #pt (0,d) with d=dist(p1,p2) (euclidean dist)
      loss += -dist(p1, p2) #dist to diagonal of (0,d) is dist(p1, p2)/2.

    return loss

def loss_topo0(point_cloud, dgm2): #changed so it directly returns the loss
    with torch.no_grad():
        ##convert points for computing PD:
        points_np = point_cloud.numpy()
        dgm = ripser_parallel(points_np, maxdim=0, return_generators=True)
        distance_bottleneck, matching = persim.bottleneck(dgm['dgms'][0][:-1], dgm2['dgms'][0][:-1], matching=True)
        #find the pair that gives the max distance:
        index = np.argmax(matching[:, 2])
        i, j = int(matching[index][0]), int(matching[index][1]) #i, j: the i-th and j-th point of the dgm1, dgm2 respectively, that give the bottleneck dist.
        # (if the largest dist is point<->diagonal: i or j is -1)
        #i is the i-th pt in dgm and j is the j-th pt in dgm2 which give the bottleneck dist (i.e. it is the largest dim)
        #for the loss, need to know what is the point i (learnable), i=(distmatrix[xi,yi],distmatrix[ai,bi]) in the distance matrix for some 4 indices
        #but gen[0]
        # i is the index of a point of the PD. but (gens[i][1], gens[i][2]) is the pair of vertices of the point cloud that correspond to the point i=(0,d), with d=dist(gens[i][1]-gens[i][2])
        #get the 2 points that give the distance of the i-th pt in dgm in the 1st diagram and compute the loss:
    if i>=0:
      point1_dgm1 = point_cloud[dgm['gens'][0][i][1]]
      point2_dgm1 = point_cloud[dgm['gens'][0][i][2]]

    if i>=0 and j>=0:
      new_bdist = torch.abs(dist(point1_dgm1, point2_dgm1) - dgm2['dgms'][0][j][1])
    else:
      if i==-1: #so the j-th point from dgm2 is matched to the diagonal -> backprop through loss would give 0 -> goal: make points further from diag
        #new_bdist = torch.abs(dist(point1_dgm2, point2_dgm2) - 0.)/2
        new_bdist = push0(point_cloud)
      else: #then  j==-1, so the i-th point from dgm1 is matched to the diagonal
        new_bdist = dist(point1_dgm1, point2_dgm1)/2.

    loss = new_bdist

    return loss

"""Bottleneck1 loss fctn: loss_topo1, with auxiliary loss in push1 when dist is dgm2 matched to diagonal"""

#auxiliary loss when d(D,D0) (in deg1) only depends on D0 (so gradients are 0):
def push1(point_cloud):
    # Compute the persistence diagram without backprop
    with torch.no_grad():
        ##convert points for computing PD:
        points_np = point_cloud.numpy()
        # get PD with generators:
        dgm = ripser_parallel(points_np, maxdim=1, return_generators=True)

    p1, p2, p3, p4 = point_cloud[dgm['gens'][1][0][0][0]], point_cloud[dgm['gens'][1][0][0][1]], point_cloud[dgm['gens'][1][0][0][2]], point_cloud[dgm['gens'][1][0][0][3]]
    #get the dist to diagonal:
    loss = -(dist(p3, p4)-dist(p2, p1))/2.

    for i in range(1, len(dgm['gens'][1])):
      p1, p2, p3, p4 = point_cloud[dgm['gens'][1][0][i][0]], point_cloud[dgm['gens'][1][0][i][1]], point_cloud[dgm['gens'][1][0][i][2]], point_cloud[dgm['gens'][1][0][i][3]]
      #get the dist to diagonal:
      loss += -(dist(p3, p4)-dist(p2, p1))/2.

    return loss

def loss_topo1(point_cloud, point_cloud2, dgm2):
    with torch.no_grad():
        ##convert points for computing PD:
        points_np = point_cloud.numpy()
        dgm = ripser_parallel(points_np, maxdim=1, return_generators=True)

        distance_bottleneck, matching = persim.bottleneck(dgm['dgms'][1], dgm2['dgms'][1], matching=True)

        #find the pair that gives the max distance:
        index = np.argmax(matching[:, 2])
        i, j = int(matching[index][0]), int(matching[index][1])

        #i is the i-th pt in dgm and j is the j-th pt in dgm2 which give the bottleneck dist (i.e. it is the largest dim)
        #for the loss, need to know what is the point i (learnable), i=(distmatrix[xi,yi],distmatrix[ai,bi]) in the distance matrix for some 4 indices
        # i is the index of a point of the PD. but (gens[i][1], gens[i][2]) is the pair of vertices of the point cloud that correspond to the point i=(0,d), with d=dist(gens[i][1]-gens[i][2])

    #get the 2 points that give the distance of the i-th pt in dgm in the 1st diagram:
    #if i>0, then the pt of dgm1 is off-diag:
    if i>=0:
      point0_dgm1 = point_cloud[dgm['gens'][1][0][i][0]]
      point1_dgm1 = point_cloud[dgm['gens'][1][0][i][1]]
      point2_dgm1 = point_cloud[dgm['gens'][1][0][i][2]]
      point3_dgm1 = point_cloud[dgm['gens'][1][0][i][3]]
      birth_dgm1 = dist(point0_dgm1, point1_dgm1)
      death_dgm1 = dist(point2_dgm1, point3_dgm1)
    #get the 2 points that give the distance of the j-th pt in dgm in the 2nd diagram:
    if j>=0:
      point0_dgm2 = point_cloud2[dgm2['gens'][1][0][j][0]]
      point1_dgm2 = point_cloud2[dgm2['gens'][1][0][j][1]]
      point2_dgm2 = point_cloud2[dgm2['gens'][1][0][j][2]]
      point3_dgm2 = point_cloud2[dgm2['gens'][1][0][j][3]]
      birth_dgm2 = dist(point0_dgm2, point1_dgm2)
      death_dgm2 = dist(point2_dgm2, point3_dgm2)

    if i>=0 and j>=0:
      new_bdist = dist_sup_tc(birth_dgm1, death_dgm1, birth_dgm2, death_dgm2)
    else:
      if i==-1: #so the j-th point from dgm2 is matched to the diagonal
        new_bdist = push1(point_cloud)
      else: #then j==-1, so the i-th point from dgm1 is matched to the diagonal
        new_bdist = (death_dgm1 - birth_dgm1)/2.

    loss = new_bdist

    return loss

"""Loss based on dsigma0:"""

def dist_2(a, b, c, d):
    return (a - c)**2 + (b - d)**2

#return ksigma:
def ksigma(point_cloud, point_cloud2):
    sigma = 0.01
    # Compute the persistence diagram without backprop
    with torch.no_grad():
        ##convert points for computing PD:
        points_np = point_cloud.numpy()
        points_np2 = point_cloud2.numpy()
        # get PD with generators:
        dgm = ripser_parallel(points_np, maxdim=0, return_generators=True)
        dgm2 = ripser_parallel(points_np2, maxdim=0, return_generators=True)

    ksigma = 0
    ## use formula for k_sigma from paper (https://arxiv.org/pdf/1412.6821.pdf):
    for i in range(len(dgm['gens'][0])):
        # pt in dgm: (0,d), d=dist(p1,p2)
        p1, p2 = point_cloud[dgm['gens'][0][i][1]], point_cloud[dgm['gens'][0][i][2]]
        d1 = dist(p1, p2)
        for j in range(len(dgm2['gens'][0])):
           #pt in dgm2: (0,d), d=dist(q1,q2)
           q1, q2 = point_cloud2[dgm2['gens'][0][j][1]], point_cloud2[dgm2['gens'][0][j][2]]
           d2 = dist(q1, q2)
           ksigma += torch.exp(-dist_2(0, d1, 0, d2)/(8*sigma)) - torch.exp(-dist_2(0, d1, d2, 0)/(8*sigma))

    ksigma *= 1/(8*3.141592*sigma)
    return ksigma

#return pseudo-distance that comes ksigma and squared, dsigma**2:
def dsigma(point_cloud, point_cloud2):
    k11 = ksigma(point_cloud, point_cloud)
    #k22 = ksigma(point_cloud2, point_cloud2)
    k12 = ksigma(point_cloud, point_cloud2)
    #return k11 + k22 - 2*k12
    return k11 - 2*k12 #no need of k22 since no backpropagation through it (fixed point cloud)

"""Loss with dsigma1 (degree 1 of diagrams):"""

#the only diff from ksigma (deg0) is how to take the pt of dgms (b,d) wrt the pts of the point clouds, for the backpropagation
def ksigma1(point_cloud, point_cloud2):
    sigma = 0.01
    # Compute the persistence diagram without backprop
    with torch.no_grad():
        ##convert points for computing PD:
        points_np = point_cloud.numpy()
        points_np2 = point_cloud2.numpy()
        # get PD with generators:
        dgm = ripser_parallel(points_np, maxdim=0, return_generators=True)
        dgm2 = ripser_parallel(points_np2, maxdim=0, return_generators=True)

    ksigma1 = 0
    ## use formula for k_sigma from paper (https://arxiv.org/pdf/1412.6821.pdf):
    for i in range(len(dgm['gens'][1])):
        # pt in dgm: (b1,d1), with b1, d1 = dist(p2, p1), dist(dist(p3, p4)
        p1, p2, p3, p4 = point_cloud[dgm['gens'][1][0][i][0]], point_cloud[dgm['gens'][1][0][i][1]], point_cloud[dgm['gens'][1][0][i][2]], point_cloud[dgm['gens'][1][0][i][3]]
        b1 = dist(p1,p2)
        d1 = dist(p3,p4)

        for j in range(len(dgm2['gens'][1])):
          #pt in dgm2: (b2,d2)
          q1, q2, q3, q4 = point_cloud2[dgm2['gens'][1][0][j][0]], point_cloud2[dgm2['gens'][1][0][j][1]], point_cloud2[dgm2['gens'][1][0][j][2]], point_cloud2[dgm2['gens'][1][0][j][3]]
          b2 = dist(q1,q2)
          d2 = dist(q3,q4)

          ksigma1 += torch.exp(-dist_2(b1, d1, b2, d2)/(8*sigma)) - torch.exp(-dist_2(b1, d1, d2, b2)/(8*sigma))

    ksigma1 *= 1/(8*3.141592*sigma)
    return ksigma1

def dsigma1(point_cloud, point_cloud2):
    k11 = ksigma1(point_cloud, point_cloud)
    #k22 = ksigma(point_cloud2, point_cloud2)
    k12 = ksigma1(point_cloud, point_cloud2)
    #return k11 + k22 - 2*k12
    return k11 - 2*k12 #no need of k22 since no backpropagation through it (fixed point cloud)

"""Function for plotting the gif of evolution of pt cloud during training:"""

!pip install pillow

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from IPython.display import Image as IPImage

def plot_pc_gif(point_cloud):
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(point_cloud[:, 0], point_cloud[:, 1], s=10, c='b')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Point Cloud')
    plt.xlim(0, 30)  # Adjust the limits as per your point cloud data
    plt.ylim(-30, 30)  # Adjust the limits as per your point cloud data
    plt.close(fig)
    return fig

def generate_gif(point_clouds, name):
    # Create a list of figures for each point cloud
    figures = [plot_pc_gif(point_cloud) for point_cloud in point_clouds]

    # Save each figure as an image and store them in a list
    images = []
    for idx, fig in enumerate(figures):
        fig.savefig(f'point_cloud_{idx}.png', dpi=80)
        images.append(Image.open(f'point_cloud_{idx}.png'))

    # Save the images as a GIF
    images[0].save(name, save_all=True, append_images=images[1:], duration=200, loop=0)

    # Display the GIF
    IPImage(name)

#to use it:
#generate_gif(point_clouds)

"""To display the gif:"""

from IPython.display import Image as Image_IPython, display

def display_gif(filename):
    display(Image_IPython(filename))

# Provide the filename of the generated GIF
gif_filename = 'point_clouds_evolution.gif'

# Display the GIF:
#display_gif(gif_filename)

"""Example of use:

Test 1: Train a pt cloud with l_topo of degree 0: Here the learnable pt cloud begins with 5 clusters, and the reference one has 3 clusters. It learns to have these 3 clusters.
"""

#!apt-get install -y imagemagick

"""First, generate a snythetic ground truth point cloud with dgm:"""

##create point_cloud2 and dgm2:
point_cloud2 = np.array([[5.,5.], [10., 10.], [20.0, 6.0]])

# Plot the point cloud
fig = go.Figure(plot_point_cloud(point_cloud2))
fig.show()

dgm2 = ripser_parallel(point_cloud2, maxdim=0, return_generators=True)
dgm_gtda = _postprocess_diagrams([dgm2["dgms"]], "ripser", (0, ), np.inf, True)[0]
fig = go.Figure(plot_diagram(dgm_gtda, homology_dimensions=(0, )))
fig.show()

point_cloud2 = torch.tensor(point_cloud2, dtype=torch.float32)

def optimize_point_cloud(number_of_iterations, number_of_points):
    point_clouds = []
    losses = []

    ## added for creating manually the initial point cloud:
    number_of_points = 60
    newdata = np.zeros((number_of_points,2))
    r1 = 0.5
    for i in range(number_of_points):
      newdata[i][0] = random.uniform(-r1, r1)
      newdata[i][1] = random.uniform(-r1, r1)

    for i in range(10):
      newdata[i+10][0] = random.uniform(-r1, r1)+10.
      newdata[i+10][1] = random.uniform(-r1, r1)
    for i in range(10):
      newdata[i+20][0] = random.uniform(-r1, r1)
      newdata[i+20][1] = random.uniform(-r1, r1)+20
    for i in range(10):
      newdata[i+30][0] = random.uniform(-r1, r1)+30
      newdata[i+30][1] = random.uniform(-r1, r1)+30
    for i in range(20):
      newdata[i+40][0] = random.uniform(-r1, r1)+10
      newdata[i+40][1] = random.uniform(-r1, r1)-25

    point_cloud = torch.tensor(newdata, dtype=torch.float32, requires_grad=False)

    # plot the PD initial:
    dgm = ripser_parallel(newdata, maxdim=1, return_generators=True)
    dgm_gtda = _postprocess_diagrams([dgm["dgms"]], "ripser", (0,), np.inf, True)[0]
    fig = go.Figure(plot_diagram(dgm_gtda, homology_dimensions=(0,)))
    fig.show()

    point_cloud.requires_grad = True
    point_clouds.append(np.copy(point_cloud.detach().numpy()))
    optimizer = torch.optim.Adam([point_cloud], lr=0.05)

    #plot initial point cloud:
    fig = go.Figure(plot_point_cloud(point_clouds[-1]))
    fig.show()

    for i in range(number_of_iterations):

        loss = loss_topo0(point_cloud, dgm2)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (i<1000 and i%200==0): point_clouds.append(np.copy(point_cloud.detach().numpy()))
        if (i%1000==0): point_clouds.append(np.copy(point_cloud.detach().numpy()))

        if (i+1) % 1000 == 0:
            print(f"Iteration {i + 1}/{number_of_iterations}, Loss: {loss.item()}")
            fig = go.Figure(plot_point_cloud(point_clouds[-1]))
            fig.show()

            # get PD of latest point cloud and plot it:
            dgm = ripser_parallel(point_clouds[-1], maxdim=1, return_generators=True)
            dgm_gtda = _postprocess_diagrams([dgm["dgms"]], "ripser", (0,), np.inf, True)[0]
            fig = go.Figure(plot_diagram(dgm_gtda, homology_dimensions=(0,)))
            fig.show()

            with torch.no_grad(): losses.append(loss.item())

    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel("Iteration (/1000)")
    plt.ylabel("Loss")
    plt.show()

    return point_clouds

point_clouds = optimize_point_cloud(50000, 100)
generate_gif(point_clouds, 'pc_evolution_test1.gif')
display_gif('pc_evolution_test1.gif')

