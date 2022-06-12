'''
To generate supplemental points for FCC Rodrigues zone
To solve the problem that point density is lower near FZ boundary than inside
Points are in FZ, but far away enough from the origin
'''

import numpy as np
from math import tan, pi
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from itertools import combinations
from rotations import ro2eu

# 60 degree around [111], eight triangular facets orthogonal to the <111> directions at a distance of tan(π/6) (=√3-1) from the origin
a = 3-2*2**0.5 
# 90 degree around [001], six octagonal facets orthogonal to the <100> directions, at a distance of tan(π/8) (=√2-1) from the origin
b = 2**0.5-1 

t1 = np.array([[-b,-b,-b,1],
                [-a,-b,-b,1],
                [-b,-a,-b,1],
                [-b,-b,-a,1]])
t2 = np.array([[-b,-b,b,1],
                [-a,-b,b,1],
                [-b,-a,b,1],
                [-b,-b,a,1]])
t3 = np.array([[-b,b,-b,1],
                [-a,b,-b,1],
                [-b,a,-b,1],
                [-b,b,-a,1]])
t4 = np.array([[b,-b,-b,1],
                [a,-b,-b,1],
                [b,-a,-b,1],
                [b,-b,-a,1]])
t5 = np.array([[-b,b,b,1],
                [-a,b,b,1],
                [-b,a,b,1],
                [-b,b,a,1]])
t6 = np.array([[b,-b,b,1],
                [a,-b,b,1],
                [b,-a,b,1],
                [b,-b,a,1]])
t7 = np.array([[b,b,-b,1],
                [a,b,-b,1],
                [b,a,-b,1],
                [b,b,-a,1]])
t8 = np.array([[b,b,b,1],
                [a,b,b,1],
                [b,a,b,1],
                [b,b,a,1]])
# for i in [t1,t2,t3,t4,t5,t6,t7,t8]:
#     print(np.linalg.det(i))
sign = [-1,1,1,1,-1,-1,-1,1]


def insideFZ(point):
    '''
    Test whether a point is inside FCC FZ

    Input:
    point: arrays of point coordinates, (n,4)

    Return:
    arrays of booleans, (n,)
    '''
    n = len(point)
    result = np.ones((n,),dtype=bool)

    for i in range(n):
        p = point[i]
        for (j,s) in zip([t1,t2,t3,t4,t5,t6,t7,t8],sign):
            count = 0
            for k in range(4):
                temp = deepcopy(j)
                temp[k,0:3] = p
                if np.linalg.det(temp)*s < 0.:
                    break
                else:
                    count += 1
            if count == 4:
                result[i] = False
                break
    
    return result

def triangle_mesh(point):
    '''
    do 1-turn triangular mesh
    Input:
    point: array of points, (n*3)

    Return:
    mid-points of any two in points, (m*3)
    '''

    n = len(point)
    result = np.zeros((1,3))
    for i in combinations(range(n),2):
        temp = (point[i[0]] + point[i[1]])*0.5
        result = np.concatenate((result, np.array([temp])),axis=0)
    return result[1:]

def EMsampleRFZ_supplement(n=1000, d=a):
    '''
    generate random points in FCC FZ
    distance from the origin is larger than d

    Input:
    n: number of points
    d: distance

    Return:
    points that meet requirements, (m,4)
    (m <= n)
    '''

    temp_n = 4*n
    while True:
        points = 2*b*np.random.random((temp_n,3))-b
        temp_points = points[np.linalg.norm(points,axis=-1)>=d]
        if len(temp_points) >= n:
            points = temp_points[:n]
            break
        else:
            temp_n *= 2
    
    return points[insideFZ(points)]


def EMsampleRFZ_supplement2(n=10, m=3, l=1, c=0.05):
    '''
    generate uniform meshgrid points in FCC FZ
    1. base is a shell of points on FZ boundaries
    2. each batch is coefficient*base

    Input:
    n: int, number of points along the edge of a octagonal facet
    m: int, number of epochs in triangular meshgrid on triangular facet (suggest <= 5)
    l: int, number of batches
    c: float, difference between coefficients of layers nearby

    Output
    array of points, (k*3)
    '''

    # meshgrid on six octagonal facets
    r = np.linspace(-b, b, n)
    x,y = np.meshgrid(r,r)
    x = x.flatten()
    y = y.flatten()
    temp = b*np.ones_like(x)

    points_1 = np.concatenate((np.array([x,y,temp]).T,
                                np.array([x,y,-temp]).T,
                                np.array([x,temp,y]).T,
                                np.array([x,-temp,y]).T,
                                np.array([temp,x,y]).T,
                                np.array([-temp,x,y]).T),axis=0)
    
    # meshgrid on eight triangular facets
    points_2 = np.zeros((1,3))
    for i in [t1,t2,t3,t4,t5,t6,t7,t8]:
        temp = i[1:,:3]
        for epoch in range(m):
            temp = np.unique(np.concatenate((temp,triangle_mesh(temp)),axis=0),axis=0)
        points_2 = np.concatenate((points_2,temp),axis=0)
    points_2 = points_2[1:]


    coeff = 0.99
    points = coeff*np.unique(np.concatenate((points_1,points_2),axis=0),axis=0)
    points = points[insideFZ(points)]

    
    result = np.zeros((1,3))
    for i in range(l):
        result = np.concatenate((result,(1-i*c)*points),axis=0)
    return result[1:]


def plotFZ():
    fig = plt.figure()
    ax = plt.gca(projection='3d')

    for i in [t1,t2,t3,t4,t5,t6,t7,t8]:
        for j in combinations(range(1,4),2):
            ax.plot(i[j,0], i[j,1], i[j,2], c='black')
        for j in range(1,4):
            temp = deepcopy(i[j,0:3])
            temp[np.abs(temp)==a] *= -1
            ax.plot((i[j,0],temp[0]),(i[j,1],temp[1]),(i[j,2],temp[2]), c='black')
    
    # points = EMsampleRFZ_supplement(1000, d=0.38)
    # ax.scatter(points[:,0],points[:,1],points[:,2], c='red')
    points = EMsampleRFZ_supplement2(n=10, m=3, l=3, c=0.05)
    # To color different shells
    n = len(points)
    ax.scatter(points[:int(n/3),0],points[:int(n/3),1],points[:int(n/3),2], c='red')
    ax.scatter(points[int(n/3):2*int(n/3),0],points[int(n/3):2*int(n/3),1],points[int(n/3):2*int(n/3),2], c='green')
    ax.scatter(points[2*int(n/3):3*int(n/3),0],points[2*int(n/3):3*int(n/3),1],points[2*int(n/3):3*int(n/3),2], c='blue')
    print(len(points))
    # ax.scatter(points[:,0],points[:,1],points[:,2], c='red')
    ax.tick_params(axis = 'both', which = 'major', labelsize = 17)
    # plt.show()
    plt.savefig('test_2.png', dpi=600)
    return

def points_output(file, n=10, m=3, l=1, c=0.05):
    '''
    write supplementary points into an existing orientation file generated by EMsampleRFZ
    '''
    
    points = EMsampleRFZ_supplement2(n, m, l, c)
    length = np.sqrt(np.sum(np.power(points,2),axis=-1)).reshape((-1,1))
    points = np.concatenate((points,length), axis=-1)

    f = open(file, 'a')
    
    for i in range(len(points)):
        euler = 180./pi*ro2eu(points[i])
        w = ' '*5+str(euler[0])+' '*5+str(euler[1])+' '*4+str(euler[2])+'\n'
        f.write(w)

    f.close()
    print(len(points))
    return

if __name__ == '__main__':  
    plotFZ()
    file = 'Ni_euler_extended.txt'
    points_output(file, n=30, m=5, l=6, c=0.03)
    pass
