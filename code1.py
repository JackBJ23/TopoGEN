import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
#other for PDiags:
import math
import numpy as np
import scipy
import matplotlib.animation as animation

print("hello")
a = 3
a = a+1
print("Result:", a)
b = 2
print("Number: ", b)

# Generate random image data
image = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)

# Create a figure and axes
fig, ax = plt.subplots()

# Display the image
ax.imshow(image)

# Remove the axis labels
ax.axis('off')

# Show the plot
plt.show()
