import numpy as np
from math import pi

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

def misorientation(a,b):
    '''
    works for all structures

    input: a,b: 2darray of unit quaternions, n*4
    output: misorientation angle (in unit of degree): 2darray, n*1
    '''

    b_star = np.multiply(b,np.array([1.,-1.,-1.,-1.]))
    result = quaternion_multiply(a,b_star)
    
    # return result
    return 360/pi*np.arccos(np.clip(result[:,:1],-1,1))

if __name__ == '__main__':
    pass