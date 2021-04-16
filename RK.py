
import matplotlib.pyplot as plt
import math
import numpy as np
import cmath

rcore = 0.1

List_Strengh_Vortices = np.array([1.0, -1.0, 1.0, -1.0])
List_of_Vort = np.array([-1.0 - 0.5j, -1.0 + 0.5j, -0.5 - 0.5j, -0.5 + 0.5j])

N = 100
steps = 100
tmax = 0.79


dt = tmax / steps


def RungeUpdateVortInTime(listofz):

    #listofz - list of vortices

    k1 = dt * value_of_interaction_matrix(listofz)
    k2 = dt * value_of_interaction_matrix(listofz + k1/2)
    k3 = dt * value_of_interaction_matrix(listofz + k2/2)
    k4 = dt * value_of_interaction_matrix(listofz + k3)

    #return list [z1 n+1, z2 n+1, z3 n+1, z4 n+1]

    return (k1 + 2*k2 + 2*k3 + k4)/6 + listofz

def functionValue(K, z, z0):

    r2 = abs(z - z0)**2

    temp =  K * (z0 - z) / r2 * (1 - cmath.exp((-r2)/(rcore**2))) * complex(0, 1)

    #temp =  K * (z0 - z) / (r2 * (1 - cmath.exp((-r2)/(rcore**2)))) * complex(0, 1)

    return temp

def value_of_interaction_matrix(Zlist):

    M = len(Zlist)

    #matrixValues = np.zeros((M, M), dtype=np.complex128)
    matrixSum = np.zeros((M), dtype=np.complex128)

    for i in range(M):
        for j in range(M):
            if i != j:
                matrixSum[i]  += functionValue(List_Strengh_Vortices[j], Zlist[i], Zlist[j])

    return matrixSum

def caclulateMatrixVortexesInTime():

    M = len(List_of_Vort)

    matrixVortInTime = np.zeros((steps +1, M), dtype=np.complex128)

    matrixVortInTime[0, :] = List_of_Vort

    #print(matrixVortInTime)

    for i in range(1, steps+1):

        matrixVortInTime[i, :] = RungeUpdateVortInTime(matrixVortInTime[i - 1, :])

    #print(matrixVortInTime[1, :])
    #print(matrixVortInTime[steps - 1, :])
    #print(matrixVortInTime[steps, :])
    

    return matrixVortInTime

def evalOnTimeGrid(z, listofZ):
    #listofZ - list of vortices

    temp = 0

    for i in range(len(List_of_Vort)):

        temp += functionValue(List_Strengh_Vortices[i], z, listofZ[i])

    #return complex value
    return temp

def RungeEvalInDim(z, listofZ):
    #z - poit on the grid, x + iy, y[-2.5, 2.5], x[-2.5, 2.5] or somethinng like that
    #listofZ  = listofZ[::-1]

    k1 = dt * evalOnTimeGrid(z,        listofZ)
    k2 = dt * evalOnTimeGrid(z + k1/2, listofZ)
    k3 = dt * evalOnTimeGrid(z + k2/2, listofZ)
    k4 = dt * evalOnTimeGrid(z + k3,   listofZ)

    #returned complex value to display
    return (k1 + 2*k2 + 2*k3 + k4)/6 


def display(mesh):

    cmap = 'plasma'

    plt.figure(num = None, figsize=(10, 10), dpi=300)

    plt.axis('off')

    #plot = plt.imshow(mesh, cmap = cmap, interpolation='lanczos' )
    plot = plt.imshow(mesh, cmap = cmap, interpolation='lanczos')
    ####

    filenameImage = f'test{N}_{steps}_{tmax}_{rcore}_{cmap}.png'

    plt.savefig(filenameImage, bbox_inches = 'tight')

    ####

    plt.show()
    plt.close()




def  test():
    print('0')

def  run():
    # {y, -1.125, 1.125, 2.25/n}, {x, -0.35, 1.9, 2.25/n}
    #ListDensityPlot[Map[Sign[Im[#]]Arg[#] &, image, {2}]

    VortMesh = caclulateMatrixVortexesInTime()


    n = N
    mesh = np.zeros((n, n))

    xm = -2
    ym = -2

    arrX = np.linspace(xm, xm + 4, num = n)
    arrY = np.linspace(ym, ym + 4, num = n)

    

    value = 0

    ik = 0
    jk = 0

    for i in range(n):
        for j in range(n):

            value = complex(arrX[i], arrY[j])

            for t in range(steps, -1, -1):
                value -= RungeEvalInDim(value, VortMesh[t, :])

            mesh[j, i] = cmath.phase(value) * math.copysign(1, value.imag) 
            #mesh[j, i] = value

    #mesh = np.rot90(mesh, k = 3)
    print(mesh)
    #mesh = np.rot90(np.flip(mesh), k = 3)

    filenameArr = f'ArrNP{N}_{steps}_{tmax}_{rcore}'
    np.save(filenameArr, mesh)

    print('done')

    display(mesh)









if __name__ == '__main__':
    run()


