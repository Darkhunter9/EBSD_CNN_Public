import numpy as np
from math import sin, cos, pi

def quaternion_multiply(a,b):
    '''
    quaternion multiplication

    input: a,b: 2darray of quaternions, n*4
    output: multiplcation: 2darray, n*4
    '''

    result = np.zeros_like(a,dtype='float32')
    result[:,[0]] = np.sum(np.multiply(np.multiply(a,b),np.array([1,-1,-1,-1])),axis=1).reshape((-1,1))
    result[:,1:] = np.multiply(b[:,1:],a[:,[0]]) + np.multiply(a[:,1:],b[:,[0]]) + np.cross(a[:,1:],b[:,1:])

    return result

def disorientation(a,b):
    '''
    only works for cubic material
    point group higher than 432

    input: a,b: 2darray of unit quaternions, n*4
    output: disorientation: 2darray, n*1
    '''

    b_star = np.multiply(b,np.array([1.,-1.,-1.,-1.]))
    misorientation = -1*np.sort(-1*np.abs(quaternion_multiply(a,b_star)), axis=1)

    # 1st element: misorientation[:,[0]]
    # 2nd element: 0.5*np.sum(misorientation,axis=1).reshape((-1,1))
    # 3rd element: 0.5**0.5*np.sum(misorientation[:,[0,1]],axis=1).reshape((-1,1))
    temp = np.concatenate((misorientation[:,[0]], 
                        0.5*np.sum(misorientation,axis=1).reshape((-1,1)),  
                        0.5**0.5*np.sum(misorientation[:,[0,1]],axis=1).reshape((-1,1))), 
                        axis=1)
    
    return 360/pi*np.arccos(np.clip(np.max(temp,axis=1).reshape((-1,1)),-1,1))


if __name__ == '__main__':
    print('test of quaternion multiplication')
    a = np.array([[1,2,3,4],[1,2,3,4]])
    b = np.array([[1,2,3,4],[5,6,7,8]])
    print(quaternion_multiply(a,b))

    print('test of disorientation')
    a = np.array([[cos(30/180/pi),0,0,sin(30/180/pi)],[cos(60/180/pi),0,0.5**0.5*sin(60/180/pi),0.5**0.5*sin(60/180/pi)]])
    b = np.array([[cos(60/180/pi),0,0.5**0.5*sin(60/180/pi),0.5**0.5*sin(60/180/pi)],[cos(30/180/pi),0,0,sin(30/180/pi)]])
    print(disorientation(a,b))