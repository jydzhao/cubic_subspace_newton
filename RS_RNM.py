import numpy as np
from scipy.sparse.linalg import eigsh
from collections import defaultdict
from datetime import datetime
from matplotlib import pyplot as plt

def f_quadratic(H, x):
    return 1/2 * x.T @ H @ x

def grad_quadratic(H, x):
    return H @ x

def Hess_quadratic(H, x):
    return H

def sample_gaussian(s,n):
    P = np.random.normal(loc=0,scale=1/s,size=(s,n))
    return P


def RS_RNM(solver, loss, grad_f, hess_vec, Hess_f, X, Y, x0, tolerance=1e-4, tau=100, max_iter=1000, H_0=1.0, line_search=True, schedule='constant', log_every=10, trace=True, log_Hessian=False, verbose_level=0):

    # Constants taken from the original works of RS-RNM in Fuji et al. 
    c1 = 2
    c2 = 1
    gamma = 0.5
    alpha = 0.3
    beta = 0.5

    history = defaultdict(list) if trace else None
    xk = np.copy(x0)
    n = x0.shape[0]

    print(f'Running RS-RNM with s={tau}, n={n}')

    start_timestamp = datetime.now()

    for it in range(max_iter):

        Pk = sample_gaussian(tau,n) # sample random Gaussian

        fk = loss(xk,X,Y) # compute full loss
        gk = grad_f(xk, X, Y, np.arange(n)) # compute full gradient
        gk_norm = np.linalg.norm(gk)
        Hk = Hess_f(xk, X, Y, np.arange(n)) # compute full Hessian (n x n)

        if verbose_level >= 1 and it % 100 == 0:
            print(f'iter: {it}, loss: ', fk, ' grad norm: ', gk_norm)
        
        if gk_norm < tolerance:
            status = 'success'
            print('Success!')
            break

        if trace and it%log_every==0:
            history['w_k'].append(xk.copy())
            # history['grad_S'].append(np.linalg.norm(grad_k_S))
            history['iter'].append(it)
            history['grad_est'].append(gk_norm)
            # history['grad'].append(np.linalg.norm(grad_k))
            history['func_full'].append(fk)
            # history['func_S'].append(func_S_k)
            history['time'].append((datetime.now() - start_timestamp).total_seconds())

        Sk = Pk @ Hk @ Pk.T # compute the projection of the Hessian onto random subspace (s x s)
        
        try:
            lam_min, _ = eigsh(Sk, k=1, which='SA') # calculate smallest eigenvalue of projected Hessian
        except:
            lam_min = np.linalg.eigvalsh(Sk)[0]
            print('scipy.sparse.linalg.eigsh failed. Smallest eigenvalue computed with np.linalg.eigvalsh: ', lam_min)
        Lam_k = max(0, -lam_min)

        eta_k = c1*Lam_k + c2*gk_norm**gamma

        Mk = Sk + eta_k * np.eye(tau) # regularized, projected Hessian

        dk = -Pk.T @ (np.linalg.inv(Mk) @ (Pk @ gk)) # search direction

        # Armijo's rule
        max_iter_Armijo = 100
        lk = 0
        for it_Armijo in range(max_iter_Armijo):
            beta_lk = beta**lk

            if ( fk - loss(xk +  beta_lk * dk,X,Y) >= -alpha * beta_lk * gk.T @ dk):
                tk = beta_lk
                break
            else:
                lk += 1
        
            if it_Armijo == max_iter_Armijo-1:
                print('Armijos line search failed!')
                tk = 0
                break
        
        xk = xk + tk * dk
        
        

    status = 'iterations_exceeded'

    return xk, status, history
        


def main():

    n = 1500
    A = np.random.normal(size=(n,n))

    Q, R = np.linalg.qr(A)

    H = 1/2 * A @ A.T
    print(f'cond(H) = {np.linalg.cond(H)}, lam_max(H) = {np.linalg.eigvalsh(H)[-1]}, lam_min(H) = {np.linalg.eigvalsh(H)[0]}')

    f = lambda x: f_quadratic(H,x)
    grad_f = lambda x: grad_quadratic(H,x)
    Hess_f = lambda x: Hess_quadratic(H,x)

    s = 100
    gamma = 0.5
    c1 = 2
    c2 = 1
    alpha = 0.3
    beta = 0.5
    
    x0 = np.random.standard_normal(size=(n,))

    history, _, _ = RS_RNM(x0, f, grad_f, Hess_f, s, gamma, c1, c2, alpha, beta, max_iter=1000, verbose_level=1)

    plt.figure()
    plt.subplot(221)
    plt.semilogy(history['iter'], history['loss'])
    plt.xlabel('Iter k')
    plt.ylabel('Loss')
    plt.subplot(222)
    plt.semilogy(history['time'], history['loss'])
    plt.xlabel('Time s')
    plt.ylabel('Loss')
    plt.subplot(223)
    plt.semilogy(history['iter'], history['grad_norm'])
    plt.xlabel('Iter k')
    plt.ylabel('Grad norm')
    plt.subplot(224)
    plt.semilogy(history['time'], history['grad_norm'])
    plt.xlabel('Time s')
    plt.ylabel('Grad norm')

    plt.show()
    
if __name__ == '__main__':

    main()


