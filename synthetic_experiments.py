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

from IPython.display import Image as Image_IPython, display

def display_gif(filename):
    display(Image_IPython(filename))

# Provide the filename of the generated GIF
gif_filename = 'point_clouds_evolution.gif'

# Display the GIF:
#display_gif(gif_filename)

'''
Test: minimize difference between the 2 density fctns:
'''

def density(point_cloud, dgm, x):
  tot = 0
  sigma = 0.5
  pers = 20.

  with torch.no_grad():
    for i in range(len(dgm['dgms'][0])-1): tot += (dgm['dgms'][0][i][1] / pers) ** 4

  density_x = 0 #density at x
  for i in range(len(dgm['dgms'][0])-1):
    p1, p2 = point_cloud[dgm['gens'][0][i][1]], point_cloud[dgm['gens'][0][i][2]] #pt (0,d) with d=dist(p1,p2) (euclidean dist)
    d = dist(p1, p2) #pt of pt cloud is (0,d)
    density_x += (d/pers)**4 * torch.exp(-((d-x)/sigma)**2)

  return density_x / tot

def loss_density(point_cloud, point_cloud2, dgm2):

  xs = np.linspace(0.,35.,100)

  #make dgm of point_cloud:
  with torch.no_grad():
        ##convert points for computing PD:
        points_np = point_cloud.numpy()
        dgm = ripser_parallel(points_np, maxdim=0, return_generators=True)

  loss = 0
  ## compute difference between both functions in 100 pts (those given by xs)
  dens=[]
  dens0=[]
  for x in xs:
    dx = density(point_cloud, dgm, x)
    d0x = density(point_cloud2, dgm2, x)
    loss += (dx - d0x)**2
    with torch.no_grad():
      dens.append(dx.detach().numpy())
      dens0.append(d0x.detach().numpy())

  plt.plot(xs, dens, color='blue')
  plt.plot(xs, dens0, color='red')
  plt.show()

  return loss

'''
Test:
'''

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

#def optimize_point_cloud(number_of_iterations, number_of_points):
number_of_iterations = 10000
point_clouds = []
losses = []

## added for creating manually the initial point cloud:
number_of_points = 40
newdata = np.zeros((number_of_points,2))
r1 = 0.5
for i in range(number_of_points):
  newdata[i][0] = random.uniform(-r1, r1)
  newdata[i][1] = random.uniform(-r1, r1)
for i in range(10,20):
  newdata[i][0] = random.uniform(-r1, r1)
  newdata[i][1] = random.uniform(-r1, r1)+20
for i in range(20,30):
  newdata[i][0] = random.uniform(-r1, r1)+30
  newdata[i][1] = random.uniform(-r1, r1)+30
for i in range(30,40):
  newdata[i][0] = random.uniform(-r1, r1)+10
  newdata[i][1] = random.uniform(-r1, r1)-25

point_cloud = torch.tensor(newdata, dtype=torch.float32, requires_grad=False)

# plot the PD initial:
dgm = ripser_parallel(newdata, maxdim=1, return_generators=True)
dgm_gtda = _postprocess_diagrams([dgm["dgms"]], "ripser", (0,), np.inf, True)[0]
fig = go.Figure(plot_diagram(dgm_gtda, homology_dimensions=(0,)))
fig.show()

point_cloud.requires_grad = True
point_clouds.append(np.copy(point_cloud.detach().numpy()))
optimizer = torch.optim.Adam([point_cloud], lr=0.1)

#plot initial point cloud:
fig = go.Figure(plot_point_cloud(point_clouds[-1]))
fig.show()

for i in range(number_of_iterations):

    loss = loss_density(point_cloud, point_cloud2, dgm2)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (i%1000==0): point_clouds.append(np.copy(point_cloud.detach().numpy()))

    if (i+1) % 50 == 0:
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

generate_gif(point_clouds, 'pc_evolution_test4.gif')
display_gif('pc_evolution_test4.gif')
