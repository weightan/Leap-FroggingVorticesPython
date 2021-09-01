
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

N = 500
imsized = 10


def cMap1():

    v = 10
    k = 256

    vals = np.ones((k, 4))
    vals[:, 0] = np.array([(i % v)/v for i in range(k)])
    vals[:, 1] = np.array([((i + 5) % v)/v for i in range(k)])
    vals[:, 2] = np.array([((i + 7) % v)/v for i in range(k)])
    newcmp = ListedColormap(vals)

    return newcmp

def cMap2():
    colors = [(234/255, 230/255, 202/255),
              (114/255, 0, 0),
              (234/255, 230/255, 202/255),
              (114/255, 0, 0),
              (234/255, 230/255, 202/255),
              (114/255, 0, 0),
              (30/255, 23/255, 20/255),
              (234/255, 230/255, 202/255),
              (114/255, 0, 0),
              (30/255, 23/255, 20/255),
              (234/255, 230/255, 202/255),
              (30/255, 23/255, 20/255),
              (114/255, 0, 0)]  # R -> G -> B

    cmap = LinearSegmentedColormap.from_list('my_list', colors, N=40)
    return cmap

def display(mesh):

    cmap = 'twilight_r'
    #cmap = cMap2()

    plt.figure(num = None, figsize=(imsized, imsized), dpi=300)

    plt.axis('off')

    #plot = plt.imshow(mesh, cmap = cmap, interpolation='lanczos' )
    plot = plt.imshow(mesh, cmap = cmap, interpolation='lanczos')
    ####

    filenameImage = f'test{N}_{cmap}.png'

    plt.savefig(filenameImage, bbox_inches = 'tight')

    ####

    plt.show()
    plt.close()



if __name__ == '__main__':

    mesh = np.load('ArrNP500_300_0.7_0.1_(-1-0.5j)_(-1+0.2j)_(-0.5-0.5j)_(-0.5+0.3j).npy')

    display(mesh)