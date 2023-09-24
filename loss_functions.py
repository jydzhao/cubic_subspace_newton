###################################################
### Common Loss Functions and their derivatives ###
###################################################

# Authors: Jonas Kohler and Aurelien Lucchi, 2017


from math import log
from math import sqrt

import random
import time
import os

import numpy as np
from scipy import linalg
from sklearn.utils.extmath import randomized_svd

# Return the loss as a numpy array

def robust_lin_regression_loss(w, X, Y):
    P = X.dot(w)
    l = np.average([eta(Y[i] - P[i]) for i in range(len(Y))])
    return l

def non_linear_square_loss_nonconvex(w, X , Y, alpha=1e-3, beta=1):
    n = X.shape[0]
    d = X.shape[1]
    P = X.dot(w)
    z = phi(P)
    l = 0.5 * np.average([(Y[i] - z[i]) ** 2 for i in range(len(Y))])
    l = l + alpha*np.dot(beta*w**2,1/(1+beta*w**2))
    return l

def square_loss(w, X , Y, alpha=1e-3):
    n = X.shape[0]
    d = X.shape[1]
    P = X.dot(w)
    l = 0.5 * np.average([(Y[i] - P[i]) ** 2 for i in range(len(Y))])
    l = l + 0.5 * alpha * (np.linalg.norm(w) ** 2)
    return l

def square_loss_nonconvex(w, X , Y, alpha=1e-3, beta=1):
    n = X.shape[0]
    d = X.shape[1]
    P = X.dot(w)
    l = 0.5 * np.average([(Y[i] - P[i]) ** 2 for i in range(len(Y))])
    l = l + alpha*np.dot(beta*w**2,1/(1+beta*w**2))
    return l

def hinge_loss(w, X, Y, alpha=1e-3):
    n = X.shape[0]
    d = X.shape[1]
    P = [np.dot(w, X[i]) for i in range(n)]  # prediction <w, x>
    l = np.sum([max(0, 1 - Y[i] * P[i]) for i in range(len(Y))]) / n
    l = l + 0.5 * alpha * (np.linalg.norm(w) ** 2)
    return l

def logistic_loss(w, X, Y, alpha=1e-3):
    n = X.shape[0]
    d = X.shape[1]
    z = X.dot(w) 
    l= - (np.dot(log_phi(z),Y)+np.dot(np.ones(n)-Y,one_minus_log_phi(z)))/n
    l = l + 0.5 *  alpha * (np.linalg.norm(w) ** 2)
    return l

def logistic_loss_nonconvex(w,X,Y,alpha=1e-3,beta=1):
    n = X.shape[0]
    d = X.shape[1]
    z = X.dot(w)  # prediction <w, x>
    h = phi(z)
#     print('sum(np.log(h))', sum(np.log(h)))
#     print('(np.dot(np.log(h),Y)', np.dot(np.log(h),Y))
    l= - (np.dot(np.log(h),Y)+np.dot(np.ones(n)-Y,np.log(np.ones(n)-h)))/n
    l= l + alpha*np.dot(beta*w**2,1/(1+beta*w**2))
    return l

def softmax_loss(w,X,ground_truth,alpha=1e-3,n_classes=None):
    assert (n_classes is not None),"Please specify number of classes as n_classes for softmax regression"
    n = X.shape[0]
    d = X.shape[1]
    w=np.matrix(w.reshape(n_classes,d).T)
    z=np.dot(X,w) #activation of each i for class c
    z-=np.max(z,axis=1)  # model is overparametrized. allows to subtract maximum to prevent overflow for each datapoint!
    h = np.exp(z)
    P= h/np.sum(h,axis = 1) #gives matrix with with [P]i,j = probability of sample i to be in class j (n x nC)
    error=np.multiply(ground_truth, np.log(P)) 
    l = -(np.sum(error) / n)
    l += 0.5*alpha*(np.sum(np.multiply(w,w))) #weight decay
    return l 

def monkey_loss(w,X,Y):
    l = w[0] ** 3 - 3 * w[0] * w[1] ** 2
    return l

def rosenbrock_loss(w,X,Y):
    l = (1. - w[0]) ** 2 + 100. * (w[1] - w[0] ** 2) ** 2
    return l

def non_convex_coercive_loss(w,X,Y):
    l = 0.5 * w[0] ** 2 + 0.25 * w[1] ** 4 - 0.5 * w[1] ** 2 # this can be found in Nesterov’s Book: Introductory Lectures on Convex Optimization
    return l

def rastrigin_loss(w):
    A = 10
    l = A * len(w) + np.sum(w**2 - A * np.cos(2 * np.pi * w))
    return l

def quartic_regularized_loss(w,A,lam):
    l = 1/2 * w.T @ A @ w + lam/12 * np.sum(w**4)
    return l

# Return the gradient as a numpy array

def robust_lin_regression_gradient(w, X, Y):
    P = X.dot(w)
    grad = np.average([grad_eta(Y[i] - P[i])*X[i,:] for i in range(len(Y))], axis=0)
    return grad

def square_loss_gradient(w, X, Y, alpha=1e-3):
    n = X.shape[0]
    d = X.shape[1]
    grad = (-X.T.dot(Y)+np.dot(X.T,X.dot(w)))/n
    grad = grad + alpha * w
    return grad

def square_loss_nonconvex_gradient(w, X, Y, alpha=1e-3, beta=1):
    n = X.shape[0]
    d = X.shape[1]
    grad = (-X.T.dot(Y)+np.dot(X.T,X.dot(w)))/n
    grad = grad + alpha*np.multiply(2*beta*w,(1+beta*w**2)**(-2))
    return grad

def non_linear_square_loss_nonconvex_gradient(w, X, Y, alpha=1e-3, beta=1):
    n = X.shape[0]
    d = X.shape[1]
    P = X.dot(w)
    z = phi(P)
    zz = phi(P)*(1-phi(P)) # phi'(P)
#     grad = (-zz*(X.T.dot(Y)-np.dot(X.T,z)))/n
    grad = -np.dot(X.T, zz*(Y-z))/n
    grad = grad + alpha*np.multiply(2*beta*w,(1+beta*w**2)**(-2))
    return grad

def logistic_loss_gradient(w, X, Y, alpha=1e-3):
    n = X.shape[0]
    d = X.shape[1]
    z = X.dot(w)  
    h = phi(z)
    grad= X.T.dot(h-Y)/n
    grad = grad + alpha * w
    return grad

def logistic_loss_nonconvex_gradient(w, X, Y, alpha=1e-3, beta=1):
    n = X.shape[0]
    d = X.shape[1]
    z = X.dot(w)   # prediction <w, x>
    h = phi(z)
    grad= X.T.dot(h-Y)/n
    grad = grad + alpha*np.multiply(2*beta*w,(1+beta*w**2)**(-2))
    return grad

def softmax_loss_gradient(w, X, ground_truth, alpha=1e-3,n_classes=None):
    assert (n_classes is not None), "Please specify number of classes as n_classes for softmax regression"
    
    n = X.shape[0]
    d = X.shape[1]
    w=np.matrix(w.reshape(n_classes,d).T)
    z=np.dot(X,w) 
    z-=np.max(z,axis=1)  
    h = np.exp(z)
    P= h/np.sum(h,axis = 1) 

    grad = -np.dot(X.T, (ground_truth - P))

    grad = grad / n  + alpha* w
    grad = np.array(grad)
    grad = grad.flatten(('F'))
    return grad
        
def monkey_gradient(w,X,Y):
    grad = np.array([3 * (w[0] ** 2 - w[1] ** 2), -6 * w[0] * w[1]])
    return grad

def rosenbrock_gradient(w,X,Y):
    grad = np.array([-2 + 2. * w[0] - 400 * w[0] * w[1] + 400 * w[0] ** 3, 200. * w[1] - 200 * w[0] ** 2])
    return grad

def non_convex_coercive_gradient(w,X,Y):
    grad = np.array([w[0], w[1] ** 3 - w[1]])
    return grad

def rastrigin_gradient(w):
    A = 10
    grad = np.array(2 * w + 2 * np.pi * A * np.sin(2 * np.pi * w))
    return grad

def quartic_regularized_gradient(w,A,lam,indices):
    # gradient of the quartic regularized loss f(x)=1/2 x^T*A*x + lam*||x||^4 evaluated on the coordinate subspace of dimension S indicated by <indices>.
    # A is the full dxd matrix, while x is a S-dimensional vector, with S << d 
    grad = A[np.ix_(indices-1,indices-1)] @ x + lam/3 * x**3
    return grad

# Return the Hessian matrix as a 2d numpy array
def robust_lin_regression_hessian(w, X, Y):
    P = X.dot(w)
    H = np.average([hess_eta(Y[i] - P[i])*np.outer(X[i,:], X[i,:]) for i in range(len(Y))], axis=0)
    return H

def square_loss_hessian( w, X, Y, alpha=1e-3):
    n = X.shape[0]
    d = X.shape[1]
    H = np.dot(X.T, X) / n + (alpha * np.eye(d, d))
    return H

def square_loss_nonconvex_hessian( w, X, Y, alpha=1e-3, beta=1):
    n = X.shape[0]
    d = X.shape[1]
    H = np.dot(X.T, X) / n + alpha * np.eye(d,d)*np.multiply(2*beta-6*beta**2*w**2,(beta*w**2+1)**(-3))
    return H
            
def non_linear_square_loss_nonconvex_hessian(w, X, Y, alpha=1e-3, beta=1):
    n = X.shape[0]
    d = X.shape[1]
    P = X.dot(w)
    z = phi(P)
    zz = phi(P)*(1-phi(P)) # phi'(P)
    zzz = phi(P) - 3*phi(P)**2 + 2*phi(P)**3 # phi''(P)
    H = np.average([(zz[i]**2 - (Y[i] - z[i])*zzz[i]) * np.outer(X[i,:],X[i,:])  for i in range(len(Y))],axis=0) \
        + alpha * np.eye(d,d)*np.multiply(2*beta-6*beta**2*w**2,(beta*w**2+1)**(-3))
            
    return H

def logistic_loss_hessian( w, X, Y, alpha=1e-3):
    n = X.shape[0]
    d = X.shape[1]
    z= X.dot(w)
    q=phi(z)
    h= np.array(q*(1-phi(z)))
    H = np.dot(np.transpose(X),h[:, np.newaxis]* X) / n
    H = H + alpha * np.eye(d, d) 
    return H 

def logistic_loss_nonconvex_hessian( w, X, Y, alpha=1e-3,beta=1):
    n = X.shape[0]
    d = X.shape[1]
    z= X.dot(w)
    q=phi(z)
    h= q*(1-phi(z))
    H = np.dot(np.transpose(X),h[:, np.newaxis]* X) / n  
    H = H + alpha * np.eye(d,d)*np.multiply(2*beta-6*beta**2*w**2,(beta*w**2+1)**(-3))
    return H

def softmax_loss_hessian( w, X, Y, alpha=1e-3,n_classes=None):
    assert (n_classes is not None),"Please specify number of classes as n_classes for softmax regression"

    n = X.shape[0]
    d = X.shape[1]
    w=np.matrix(w.reshape(n_classes,d).T)
    z=np.dot(X,w) 
    z-=np.max(z,axis=1)  
    h = np.exp(z)
    P= h/np.sum(h,axis = 1) 
    H=np.zeros([d*n_classes,d*n_classes])
    for c in range(n_classes):
        for k in range (n_classes):

            if c==k:
                D=np.diag(np.multiply(P[:,c],1-P[:,c]).A1)
                Hcc = np.dot(np.dot(np.transpose(X), D), X) 
            else:
                D=np.diag(-np.multiply(P[:,c],P[:,k]).A1)
                Hck = np.dot(np.dot(np.transpose(X), D), X) 
                H[c*d:(c+1)*d,k*d:(k+1)*d]=Hck
                H[k*d:(k+1)*d,c*d:(c+1)*d,]=Hck

    H = H/n + alpha*np.eye(d*n_classes,d*n_classes) 
    return H
        
def monkey_hessian(w,X,Y):
    H = np.array([[6 * w[0], -6 * w[1]], [-6 * w[1], -6 * w[0]]])
    return H

def rosenbrock_hessian(w,X,Y):
    H = np.array([[2 - 400 * w[1] + 1200 * w[0] ** 2, -400 * w[0]], [-400 * w[0], 200]])
    return H

def non_convex_coercive_hessian(w,X,Y):
    H = np.array([[1, 0], [0, 3 * w[1] ** 2 - 1]])
    return H

def rastrigin_hessian(w):
    A = 10
    H = np.diag(2 + 4 * np.pi ** 2 * A * np.cos(2 * np.pi * x))
    return H

def quartic_regularized_hessian(W,A,lam,indices):
    # Hessian of the quartic regularized loss f(x)=1/2 x^T*A*x + lam*||x||^4 evaluated on the coordinate subspace of dimension S indicated by <indices>.
    # A is the full dxd matrix, while x is a S-dimensional vector, with S << d 
    H = np.array(A[np.ix_(indices-1,indices-1)] + lam * np.diag(x**2))
    return H

# Return the Hessian-vector product as a numpy array
def robust_lin_regression_Hv(w, X, Y, v):
    P = X.dot(w)
    Hv = np.average([hess_eta(Y[i] - P[i])* X[i,:] * np.dot(X[i,:],v)  for i in range(len(Y))], axis=0)
    return Hv

def square_loss_Hv(w,X, Y, v,alpha=1e-3): 
    n = X.shape[0]
    d = X.shape[1]
    Xv=np.dot(X,v)
    Hv=np.dot(X.T,Xv)/n + alpha * v
    return Hv

def square_loss_nonconvex_Hv(w,X, Y, v,alpha=1e-3, beta=1): 
    n = X.shape[0]
    d = X.shape[1]
    Xv=np.dot(X,v)
    Hv=np.dot(X.T,Xv)/n + alpha *np.multiply(np.multiply(2*beta-6*beta**2*w**2,(beta*w**2+1)**(-3)), v)
    return Hv

def non_linear_square_loss_nonconvex_Hv(w, X, Y, v, alpha=1e-3, beta=1):
    n = X.shape[0]
    d = X.shape[1]
    P = X.dot(w)
    z = phi(P)
    zz = phi(P)*(1-phi(P)) # phi'(P)
    zzz = phi(P) - 3*phi(P)**2 + 2*phi(P)**3 # phi''(P)
    Hv = np.average([(zz[i]**2 - (Y[i] - z[i])*zzz[i]) * np.outer(X[i,:],X[i,:])  for i in range(len(Y))],axis=0) @ v \
       + alpha *np.multiply(np.multiply(2*beta-6*beta**2*w**2,(beta*w**2+1)**(-3)), v)
            
    return Hv

def logistic_loss_Hv(w,X, Y, v,alpha=1e-3): 
    n = X.shape[0]
    d = X.shape[1]
    _z=X.dot(w)
    _z = phi(-_z)
    d_binary = _z * (1 - _z)
    wa = d_binary * X.dot(v)
    Hv = X.T.dot(wa)/n
    out = Hv + alpha * v
    return out

def logistic_loss_nonconvex_Hv(w, X, Y, v,alpha=1e-3,beta=1): 
    n = X.shape[0]
    d = X.shape[1]
    _z=X.dot(w)
    _z = phi(-_z)
    d_binary = _z * (1 - _z)
    wa = d_binary * X.dot(v)
    Hv = X.T.dot(wa)/n
    out = Hv + alpha *np.multiply(np.multiply(2*beta-6*beta**2*w**2,(beta*w**2+1)**(-3)), v)
    return out

def softmax_loss_Hv(w, X, ground_truth, v, alpha=1e-30,n_classes=None):
    assert (n_classes is not None),"Please specify number of classes as n_classes for softmax regression"
    n = X.shape[0]
    d = X.shape[1]
    
    w_multi=np.matrix(w.reshape(n_classes,d).T)
    z_multi=np.dot(X,w_multi) #activation of each i for class c
    z_multi-=np.max(z_multi,axis=1)  # model is overparametrized. allows to subtract maximum to prevent overflow. 
    h_multi = np.exp(z_multi)
    P_multi= np.array(h_multi/np.sum(h_multi,axis = 1)) #gives matrix with with [P]i,j = probability of sample i to be in class j (n x nC)

    v = v.reshape(n_classes, -1)

    r_yhat = np.dot(X, v.T)
    r_yhat += (-P_multi * r_yhat).sum(axis=1)[:, np.newaxis]
    r_yhat *= P_multi
    hessProd = np.zeros((n_classes, d))
    hessProd[:, :d] = np.dot(r_yhat.T, X)/n
    hessProd[:, :d] += v * alpha
    return hessProd.ravel()

def monkey_Hv(w,X,Y,v):
    H=monkey_hessian(w,X,Y)
    return np.dot(H,v)

def rosenbrock_Hv(w,X,Y,v):
    H=rosenbrock_hessian(w,X,Y)
    return np.dot(H,v)

def non_convex_coercive_Hv(w,X,Y,v):
    H=non_convex_coercive_hessian(w,X,Y)
    return np.dot(H,v)

######## Auxiliary Functions: robust Sigmoid, log(sigmoid) and 1-log(sigmoid) computations ########

def phi(t): #Author: Fabian Pedregosa
    # logistic function returns 1 / (1 + exp(-t))
    idx = t > 0
    out = np.empty(t.size, dtype=float)
    out[idx] = 1. / (1 + np.exp(-t[idx]))
    exp_t = np.exp(t[~idx])
    out[~idx] = exp_t / (1. + exp_t)
    return out

def log_phi(t):
    # log(Sigmoid): log(1 / (1 + exp(-t)))
    idx = t>0
    out = np.empty(t.size, dtype=float)
    out[idx]=-np.log(1+np.exp(-t[idx]))
    out[~idx]= t[~idx]-np.log(1+np.exp(t[~idx]))
    return out

def one_minus_log_phi(t):
    # log(1-Sigmoid): log(1-1 / (1 + exp(-t)))
    idx = t>0
    out = np.empty(t.size, dtype=float)
    out[idx]= -t[idx]-np.log(1+np.exp(-t[idx]))
    out[~idx]=-np.log(1+np.exp(t[~idx]))
    return out

def eta(t): 
    # eta(t) = log(t^2/2 + 1)
    out = np.log(t**2/2 + 1)
    return out

def grad_eta(t):
    # eta'(t) = 2t/(t^2 + 2)
    out = 2*t/(t**2 + 2)
    return out

def hess_eta(t):
    # eta''(t) = (4-2t^2)/(t^2 + 2)^2
    out = (4 - 2*t**2)/((t**2 + 2)**2)
    return out