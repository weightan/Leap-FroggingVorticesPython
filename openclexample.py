
import matplotlib.pyplot as plt

import time

import numpy as np

import pyopencl as cl

import math

import cmath

imsized = 10 #4 - 10



List_Strengh_Vortexes = np.array([1.0, -1.0, 1.0, -1.0], dtype = np.float64)
List_of_Vort = np.array([-1.0 - 0.1j, -1.0 + 0.1j, -0.5 - 0.5j, -0.5 + 0.5j], dtype=np.complex128)

kvortexes = len(List_of_Vort)
rcore = 0.1

tmax = 0.7

steps = 70

dt = tmax/steps
actSteps = steps

actualT = steps*dt
print(actualT)


w = 500
h = 500


def calc_frame_opencl(q, vort, vortstrength = List_Strengh_Vortexes):
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    output = np.empty(q.shape, dtype=np.complex128)

    #vort = np.array([1.0 + 1j,1.0, 100.0,100.0, 1000.0,1000.0, 100.0,1.0], dtype=np.complex128)
    #vortstrength = np.array([-1, 1 , -1 , 1], dtype=np.uint16)

    mf = cl.mem_flags
    q_opencl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = q)

    output_opencl = cl.Buffer(ctx, mf.WRITE_ONLY, output.nbytes)

    vort4_opencl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vort)

    vortstrength_opencl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vortstrength)
    
    prg = cl.Program(ctx,
        
        """
    #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

    
    constant  double rcore = 0.1;
    

    #define vortN 4
    
    double2 fValue( double K, double2 Z, double2 Z0){

        double r2 =  pow(   sqrt( pow((Z.x - Z0.x), 2) +  pow((Z.y - Z0.y), 2)  ),      2);
        double constM = K*(   1.0 - exp(  (-r2)/(pow(rcore, 2)) )   ) / r2  ;
        double Xk = - constM*(Z0.y - Z.y);
        double Yk =   constM*(Z0.x - Z.x);
        double2 T1 = (double2) (Xk, Yk);

        return(T1);
    }



    __kernel void runge(__global double2 *q,
                        __global double2 *output,
                        __global double2 *vortlist,
                        __global double  *vortstrength,
                        double const dt)
    {
        int gid = get_global_id(0);

        

        double2 k1 = (double2) (0, 0);

        for (int iter = 0; iter < vortN; iter++){
        
            k1 += fValue( vortstrength[iter], q[gid], vortlist[iter] );
        }

        k1 *= dt;

       

        double2 k2 = (double2) (0, 0);

        for (int iter = 0; iter < vortN; iter++){
        
            k2 += fValue( vortstrength[iter], q[gid] + k1 /2, vortlist[iter] );

        }

        k2 *= dt;

        

        double2 k3 = (double2) (0, 0);

        for (int iter = 0; iter < vortN; iter++){
        
            k3 += fValue( vortstrength[iter], q[gid] + k2 /2, vortlist[iter] );

        }

        k3 *= dt;

        
        

        double2 k4 = (double2) (0, 0);

        for (int iter = 0; iter < vortN; iter++){
        
            k3 += fValue( vortstrength[iter], q[gid] + k3, vortlist[iter] );

        }

        k4 *= dt;

        

        output[gid] =  q[gid] - ((k1 + 2*k2 + 2*k3 + k4)/6) ;
    }
    """,
    ).build()

    prg.runge(
        queue, output.shape, None, q_opencl, output_opencl, vort4_opencl, vortstrength_opencl, np.float64(dt)

    )

    cl.enqueue_copy(queue, output, output_opencl).wait()

    return output



def display(mesh):

    #imsized = 10

    cmap = 'copper_r'

    plt.figure(num = None, figsize=(imsized, imsized), dpi=300)

    plt.axis('off')

    #plot = plt.imshow(mesh, cmap = cmap, interpolation='lanczos' )
    plot = plt.imshow(mesh, cmap = cmap, interpolation='lanczos')
    ####

    filenameImage = f'test{h}_{w}_{dt}_{actSteps}_{actualT}_{cmap}_{List_of_Vort[0]}_{List_of_Vort[1]}_{List_of_Vort[2]}_{List_of_Vort[3]}.png'

    plt.savefig(filenameImage, bbox_inches = 'tight')

    ####

    plt.show()
    plt.close()


def caclulateMatrixVortexesInTime():

    matrixVortInTime = np.zeros((steps +1, kvortexes), dtype=np.complex128)

    matrixVortInTime[0, :] = List_of_Vort

    for i in range(1, steps + 1):

        matrixVortInTime[i, :] = RungeUpdateVortInTime(matrixVortInTime[i - 1, :])

    return matrixVortInTime


def RungeUpdateVortInTime(listofz):

    #listofz - list of vortices

    k1 = dt * value_of_interaction_matrix(listofz)
    k2 = dt * value_of_interaction_matrix(listofz + k1/2)
    k3 = dt * value_of_interaction_matrix(listofz + k2/2)
    k4 = dt * value_of_interaction_matrix(listofz + k3)

    #return list [z1 n+1, z2 n+1, z3 n+1, z4 n+1]

    return (k1 + 2*k2 + 2*k3 + k4)/6 + listofz

def value_of_interaction_matrix(Zlist):

    #matrixValues = np.zeros((kvortexes, kvortexes), dtype=np.complex128)
    matrixSum = np.zeros((kvortexes), dtype=np.complex128)

    for i in range(kvortexes):
        for j in range(kvortexes):
            if i != j:
                matrixSum[i]  += functionValue(List_Strengh_Vortexes[j], Zlist[i], Zlist[j])

    return matrixSum

def functionValue(K, z, z0):

    r2 = abs(z - z0)**2

    temp =  K * (z0 - z) / r2 * (1 - math.exp((-r2)/(rcore**2))) * complex(0, 1)

    #temp =  K * (z0 - z) / (r2 * (1 - cmath.exp((-r2)/(rcore**2)))) * complex(0, 1)

    return temp


class FluidSimulation:
    def draw(self, x1, x2, y1, y2):
        xx = np.linspace(x1, x2, num = w)
        yy = np.linspace(y2, y1, num = h) * 1j
        q = np.ravel(xx + yy[:, np.newaxis]).astype(np.complex128)
        #print(q)
        start_main = time.time()

        VortMesh = caclulateMatrixVortexesInTime()
        #print(VortMesh)

        
        output = q

        for i in range(steps, 0, -1):
            #output -=  calc_frame_opencl(output, VortMesh [i, :] )
            output =  calc_frame_opencl(output, VortMesh [i, :] )


        end_main = time.time()

        secs = end_main - start_main
        print(f"Main took: {secs}")

        self.mandel = np.zeros((h*w))

        for index, z in np.ndenumerate(output):
            #self.mandel[index] = cmath.phase(z) 
            self.mandel[index] = cmath.phase(z) * math.copysign(1, z.imag) 

        self.mandel = np.reshape(self.mandel, (h,w))
       

    def create_image(self):
        posx = 0
        posy = 0
        r = 2.2

        self.draw(posx - r, posx + r, posy - r, posy + r)
        
        display(self.mandel)





if __name__ == "__main__":
    test = FluidSimulation()
    test.create_image()