from __future__ import print_function
import math
import numpy as np
import scipy
from collections import defaultdict
from datetime import datetime
from scipy.special import expit
from scipy.special import logsumexp, softmax
from utils import *


# Coordinate methods for minimizing the log-sum-exp function:
#       func(x) = mu log sum_{i=1}^m exp( (<a_i, x> - b_i) / mu )
#   a_1, ..., a_m are rows of (m x n) matrix A.
#   b is given (m x 1) vector.
#   mu is a scalar value.


def generate_logsumexp(n=100, mu=0.05, seed=31415, replication_factor=1):
    """Generates random problem."""

    np.random.seed(seed)

    m = 6 * replication_factor * n
    A = np.random.rand(m, n) * 2 - 1
    b = np.random.rand(m) * 2 - 1

    #replicate data
    A = np.repeat(A, replication_factor, axis=1)

    print(A.shape)
    print(b.shape)

    # Compute gradient at zero.
    g = A.T.dot(softmax( -1.0 / mu * b))
    # Rotate function to have f'(0) = 0.
    A -= g
    # x_star = np.zeros(n*replication_factor)
    x_star = np.zeros(n*replication_factor)
    f_star = mu * logsumexp(1.0 / mu * (A.dot(x_star) - b))

    return (A, b, x_star, f_star)

def generate_logsumexp_w_covariance_matrix(n=100, mu=0.05, seed=31415, replication_factor=1, cov_mat=None):
    """Generates random problem."""

    np.random.seed(seed)

    m = 6 * replication_factor * n

    mean = np.zeros(m)

    if cov_mat is None:
        Sig = np.random.randn(m,m) 
        cov_mat = Sig @ Sig.T/2

    
    A = np.random.multivariate_normal(mean, cov_mat, size=(n))
    A = A.T
    b = np.random.multivariate_normal(mean, cov_mat)

    #replicate data
    A = np.repeat(A, replication_factor, axis=1)

    print('corr_value: %.4f' % corr_value(A))
    print(A.shape)
    print(b.shape)

    # Compute gradient at zero.
    g = A.T.dot(softmax( -1.0 / mu * b))
    # Rotate function to have f'(0) = 0.
    A -= g
    # x_star = np.zeros(n*replication_factor)
    x_star = 5 * np.ones(n*replication_factor)
    f_star = mu * logsumexp(1.0 / mu * (A.dot(x_star) - b))

    return (A, b, x_star, f_star)


def coordinate_gradient_method(solver, loss, grad, hess_vec, hessian, A, b, x_0, tolerance, tau=1,
                               max_iter=10000, H_0=1.0, line_search=True, 
                               trace=True, schedule='constant',scale_lin=1.0, scale_quad=1.0, c=1.0, exp=0.05, eps_1=1e-2, eps_2=1e-2, jump_iter=1, jump_coord=1, 
                               verbose_level=0, log_every=100):


    history = defaultdict(list) if trace else None
    start_timestamp = datetime.now()
    x_k = np.copy(x_0)
    L_k = H_0
    n = x_k.shape[0]
    num_coord = 0 # total number of evaluated coordinates squared
    
    # mu_inv = 1.0 / mu
    # Ax_k = A.dot(x_k)
    # a_k = mu_inv * (Ax_k - b)
    func_k = loss(x_k,A,b)
    # pi_k = softmax(a_k)
    # # The whole gradient can be computed as:
    grad_k = grad(x_k, A, b, np.arange(n))    
    grad_est = grad_k
    tolerance_passed = False

    for k in range(max_iter + 1):

        if trace and k%log_every==0:
            history['w_k'].append(x_k.copy())
            history['L'].append(L_k)
            # history['grad_S'].append(np.linalg.norm(grad_k_S))
            history['grad_est'].append(np.linalg.norm(grad_est))
            history['func_full'].append(func_k)
            history['time'].append(
                (datetime.now() - start_timestamp).total_seconds())
            history['num_coord'].append(num_coord)
            # history['norm_s_k_squared'].append(np.linalg.norm(h)**2)

        # for debugging
        if verbose_level >= 1 and k % 100 == 0:
            print('iter: ', k)
            start_t_iter = datetime.now()

        # Choose randomly a subset of coordinates.
        if schedule == 'constant':
            tau_schedule = tau
        elif schedule == 'linear':
            tau_schedule = min(int(np.floor(tau+scale_lin*k)),len(x_0))
        elif schedule == 'quadratic':
            tau_schedule = min(int(np.floor(tau+scale_quad*k**2)),len(x_0))
        elif schedule == 'exponential':
            tau_schedule = min(int(np.floor(tau+c*np.exp(exp*k))),len(x_0))
        elif schedule == 'jump':
            if k <= jump_iter:
                tau_schedule = tau
            else:
                tau_schedule = jump_coord
        else:
            print('Unknown schedule type. Using constant coordinate schedule.')
            tau_schedule = tau

        num_coord += tau_schedule

        # grad_k = grad(x_k, A, b, np.arange(n)) # calculate full gradient to check convergence 

        # Choose randomly a subset of coordinates.
        S = np.random.choice(n, tau, replace=False)
        # A_S = A[:, S]
        grad_k_S = grad(x_k, A, b, S)

        alpha = 0.9
        if k == 0:
            grad_est = grad_k_S
        else:
            grad_est = alpha*grad_k_S + (1-alpha)*grad_est
        
        # Perform line search.

        # h = - 1e-8* grad_k_S
        # x_tmp = x_k.copy()
        # x_tmp[S] += h
        # func_T = loss(x_tmp,A,b)
        
        L_k = 1
        line_search_max_iters = 40
        for i in range(line_search_max_iters + 1):

            # The Gradient Step.
            h = -1.0 / L_k * grad_k_S
            x_tmp = x_k.copy()
            x_tmp[S] += h
            func_T = loss(x_tmp,A,b)
            # a_T = a_k + mu_inv * A_S.dot(h) 
            # func_T = mu * logsumexp(a_T) + lam * np.sum(x_k**2/(1+x_k**2))

            if not line_search:
                break

            if i == line_search_max_iters:
                print('W: line_search_max_iters reached.', flush=True)
                break

            if func_k - func_T >= 0.5 / L_k * grad_k_S.dot(grad_k_S):
                break

            L_k *= 2

        if line_search:
            L_k *= 0.5

        # Update the current point.
        x_k[S] += h
        # a_k = a_T
        func_k = func_T
        # pi_k = softmax(a_k)

        if trace and k%log_every==0:
            history['norm_s_k'].append(np.linalg.norm(h))

        #         if func_k - f_star <= tolerance:
        if np.linalg.norm(grad_est) <= tolerance:
            status = 'success'
            if tolerance_passed == False:
                tolerance_iter = k
            tolerance_passed = True
            
            if k - tolerance_iter >= 150:
                break

        if k == max_iter:
            status = 'iterations_exceeded'
            break

        if verbose_level >= 1 and k%100==0:
            print('func_k: ', func_k)
            print('grad_norm_est: ', np.linalg.norm(grad_est))

        if verbose_level >= 1 and k%100 ==0:
            print('time per iteration:', (datetime.now() - start_t_iter).total_seconds())

        if verbose_level >= 2 and k%20==0:
            print('func_T: ', func_T)
            print('iter: ', k)            
            # print('norm(grad_k): ', np.linalg.norm(grad_k))
            print('norm(grad_k_S): ', np.linalg.norm(grad_k_S))
#               
            print('||h||: ', np.linalg.norm(h))
            print('norm(w_k)', np.linalg.norm(x_k))
            print('norm(tmp_w_k): ', np.linalg.norm(x_tmp))
            print('L_k: ', L_k)

    return x_k, status, history


def cubic_newton_step_ncg(matvec, g, H, x_0, tol=1e-8, 
                          max_iters=10000, trace=False, 
                          rel_stop_cond=False):
    """
    Nonlinear Conjugate Gradients for minimizing the function:
        f(x) = <g, x> + 1/2 * <Ax, x> + H/3 * ||x||^3
    matvec function computes Ah product.
    """
    alpha = 0.05

    l2_norm_sqr = lambda x: x.dot(x)
    dual_norm_sqr = lambda x: x.dot(x)
    to_dual = lambda x: x
    

    results = defaultdict(list)

    n = g.shape[0]
    x_k = np.copy(x_0)
    x_k_norm = l2_norm_sqr(x_k) ** 0.5
    A_x_k = matvec(x_k)

    f_k = g.dot(x_k) + 0.5 * A_x_k.dot(x_k) + H * x_k_norm ** 3 / 3.0
    g_k = g + A_x_k + H * x_k_norm * to_dual(x_k)
    g_k_sqr_norm = dual_norm_sqr(g_k)
    g_0_sqr_norm = g_k_sqr_norm

    p_k = g_k

    if trace:   
        results['func'].append(f_k)
        results['grad_sqr_norm'].append(g_k_sqr_norm)    
        
    for k in range(max_iters):
        
        if k % n == 0:
            # restart every n iterations.
            p_k = g_k
    
        # Exact line search, minimizing g(h) = f(x_k - h p_k).
        A_p_k = matvec(p_k) # A*p_k
        A_pk_pk = A_p_k.dot(p_k) # p_k^T * A * p_k
        A_pk_xk = A_p_k.dot(x_k) # p_k^T * A * x_k
        pk_pk = to_dual(p_k).dot(p_k) # ||p_k||^2
        xk_pk = to_dual(x_k).dot(p_k) # p_k^T * x_k
        g_p_k = g.dot(p_k) # g^T * p_k
        
        if H < 1e-9: 
            # Quadratic function, exact minimum.
            h_k = (A_pk_xk + g_p_k) / A_pk_pk 
        else:
            h = 1.0
            # 1-D Newton method.
            EPS = 1e-14
            for i in range(20):
                r = l2_norm_sqr(h * p_k - x_k) ** 0.5
                # first derivative g'(h)
                g_G =  - g_p_k - A_pk_xk \
                       + h * (A_pk_pk + H * r * pk_pk) \
                       - H * r * xk_pk
                if np.abs(g_G) < EPS:
                    break
                # second derivative g''(h)
                g_H = A_pk_pk + H * r * pk_pk \
                        + H / r * (h * pk_pk - xk_pk) ** 2
                h = h - g_G / g_H
            
            h_k = h    
        
        # new iterate
        T = x_k - h_k * p_k 
        
#         T1 = x_k - h_k * p_k
#         T1_norm = l2_norm_sqr(T1) ** 0.5
#         A_T1 = matvec(T1)
#         f_T1 = g.dot(T1) + 0.5 * A_T1.dot(T1) + H * T1_norm ** 3 / 3.0
                
#         T2 = x_k - alpha * g_k
#         T2_norm = l2_norm_sqr(T2) ** 0.5
#         A_T2 = matvec(T2)
#         f_T2 = g.dot(T2) + 0.5 * A_T2.dot(T2) + H * T2_norm ** 3 / 3.0
        
#         T = T1 if f_T1 <= f_T2 else T2


        # calculate the gradient at the new iterate
        T_norm = l2_norm_sqr(T) ** 0.5
        A_T = matvec(T)
        f_T =  g.dot(T) + 0.5 * A_T.dot(T) + H * T_norm ** 3 / 3.0
        g_T = g + A_T + H * T_norm * to_dual(T) 
        
        # Dai-Yuan update rule.
        beta_k = g_T.dot(g_T) / (g_T - g_k).dot(p_k)
  
        # update the conjugate direction, iterate, loss, gradient and their norms
        p_k = g_T - beta_k * p_k
        x_k = T
        f_k = f_T
        g_k = g_T
        g_k_sqr_norm = dual_norm_sqr(g_k)
        x_k_norm = T_norm
        
        if trace:
            results['func'].append(f_k)   
            results['grad_sqr_norm'].append(g_k_sqr_norm)

        if rel_stop_cond:
            if g_k_sqr_norm <= tol * g_0_sqr_norm:
                return x_k, f_k, "success", results
        else:
            if g_k_sqr_norm <= tol:
                return x_k, f_k, "success", results
            
    return x_k, f_k, "iterations_exceeded", results


def coordinate_cubic_newton_old(A, b, mu, lam, x_0, tolerance, f_star, tau=1,
                            max_iter=10000, H_0=1.0, line_search=True, 
                            trace=True, seed=31415, schedule='constant',scale_lin=1.0, scale_quad=1.0, c=1.0, exp=0.05):
    
    np.random.seed(seed)
    history = defaultdict(list) if trace else None
    start_timestamp = datetime.now()
    x_k = np.copy(x_0)
    H_k = H_0
    n = x_k.shape[0]
    
    mu_inv = 1.0 / mu
    Ax_k = A.dot(x_k)
    a_k = mu_inv * (Ax_k - b)
    func_k = mu * logsumexp(a_k) + lam * np.sum(x_k**2/(1+x_k**2))
    pi_k = softmax(a_k)
    # The whole gradient can be computed as:
    grad_k = A.T.dot(pi_k) + 2*lam*x_k/(1+x_k**2)**2 
    # The Hessian is:
    #   hess_k = mu_inv * A.T.dot(A * pi_k.reshape(-1, 1)) \
#                   - np.outer(grad_k, grad_k.T) + np.diag(( 1 - 3 * x_k[S]**2 )/( 1 + x_k[S]**2 )**3)

    for k in range(max_iter + 1):

        if trace:
            history['grad'].append(np.linalg.norm(grad_k))
            history['func'].append(func_k)
            history['time'].append(
                (datetime.now() - start_timestamp).total_seconds())
            history['H'].append(H_k)

#         if func_k - f_star <= tolerance:
        if np.linalg.norm(grad_k) <= tolerance:
            status = 'success'
            break

        if k == max_iter:
            status = 'iterations_exceeded'
            break

        # Choose randomly a subset of coordinates.
        if schedule == 'constant':
            tau_schedule = tau
        elif schedule == 'linear':
            tau_schedule = min(int(np.floor(tau+scale_lin*k)),len(x_0))
        elif schedule == 'quadratic':
            tau_schedule = min(int(np.floor(tau+scale_quad*k**2)),len(x_0))
        elif schedule == 'exponential':
            tau_schedule = min(int(np.floor(tau+c*np.exp(exp*k))),len(x_0))
        else:
            print('Unknown schedule type. Using constant coordiante scheudule.')
            tau_schedule = tau
        
#         print('k = ', k, 'tau = ',tau_schedule)
            
            
        S = np.random.choice(n, tau_schedule, replace=False)
        A_S = A[:, S]
        grad_k_S = A_S.T.dot(pi_k) + 2*lam*x_k[S]/(1+x_k[S]**2)**2
        
        grad_k = A.T.dot(pi_k) + 2*lam*x_k/(1+x_k**2)**2 # calculate full gradient to check convergence 
        
        hess_vec = lambda h: mu_inv * (
            A_S.T.dot(pi_k * A_S.dot(h)) - grad_k_S.dot(h) * grad_k_S) + np.diag(( 1 - 3 * x_k[S]**2 )/( 1 + x_k[S]**2 )**3).dot(h) 

        # Perform line search.
        line_search_max_iters = 40
#         print('H_k: ,', H_k)
        for i in range(line_search_max_iters + 1):

            # The Cubic Newton Step.
            h, m_h, step_status, step_res = \
                cubic_newton_step_ncg(hess_vec, grad_k_S, 2 * H_k, 
                                      np.zeros(tau_schedule))
#             print('m_h: ', m_h)
            
            if step_status != 'success':
                print(('W: cubic newton step status: %s ' % step_status),
                      flush=True)
                
            a_T = a_k + mu_inv * A[:, S].dot(h)
            func_T = mu * logsumexp(a_T) + lam * np.sum(x_k**2/(1+x_k**2))

            if not line_search:
                break

            if i == line_search_max_iters:
                print('W: line_search_max_iters reached.', flush=True)
                break
                
            

            if func_k - func_T >= -m_h:
                break

            H_k *= 2

        if line_search:
            H_k *= 0.5
            if H_k <= 1e-10:
                H_k = 1e-10

        # Update the current point.
        x_k[S] += h
        a_k = a_T
        func_k = func_T
        pi_k = softmax(a_k)
        

    return x_k, status, history

