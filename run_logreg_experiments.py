from __future__ import print_function
import os
import numpy as np
from datetime import datetime
import math
import scipy
from collections import defaultdict
import yaml

from subproblem_solvers import *
from logsumexp_coordinate_methods import *

from sklearn.datasets import load_svmlight_file
from loss_functions import *
import loss_functions
from plotting_logreg import * 
from utils import *

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")


def coordinate_cubic_newton_new(solver, loss, grad, hess_vec, hessian, X, Y, w_0, 
                                tolerance, tau=1,
                                max_iter=10000, H_0=1.0, line_search=True, 
                                trace=True, schedule='constant', c=1.0, exp=0.05, eps_1=1e-2, eps_2=1e-2,
                                verbose_level=0, log_every=100, beta=0.2):

    history = defaultdict(list) if trace else None
    start_timestamp = datetime.now()
    w_k = np.copy(w_0)
    H_k = H_0
    n = w_k.shape[0]
    num_coord = 0 # (#coord^2 + #coord)
    
    lambda_k = 0
    eta_1 = 0.05
    eta_2 = 0.9 
    gamma_1 = 1.5 
    gamma_2 = 2. 
    

    func_k = loss(w_k,X,Y)
    func_S_k = np.copy(func_k)
    
    

    # The whole gradient can be computed as:
    grad_k = grad(w_k, X, Y, np.arange(n))
    grad_est = grad_k #use the full gradient as a first approximation to the gradient estimation
    alpha = 0.8
    # beta = 0.2


    if schedule == 'adaptive':
        hessian_F_est = np.linalg.norm(hessian(w_k, X, Y, np.arange(n)),2)

    tolerance_passed = False

       
    for k in range(max_iter + 1):
        
        if verbose_level >= 1 and k%100 ==0:
            start_t_log = datetime.now()

        if trace and k%log_every==0:
            history['w_k'].append(w_k.copy())
            # history['grad_S'].append(np.linalg.norm(grad_k_S))
            history['grad_est'].append(np.linalg.norm(grad_est))
            # history['grad'].append(np.linalg.norm(grad_k))
            history['func_full'].append(func_k)
            # history['func_S'].append(func_S_k)
            history['time'].append(
                (datetime.now() - start_timestamp).total_seconds())
            history['num_coord'].append(num_coord)
            # history['norm_s_k'].append(np.linalg.norm(h))
            # history['norm_s_k_squared'].append(np.linalg.norm(h)**2)
            history['H'].append(H_k)
            
        if verbose_level >= 1 and k%100 ==0:
            print('time for logging:', (datetime.now() - start_t_log).total_seconds())

        # for debugging
        if verbose_level >= 1 and k % 1 == 0:
            print('iter: ', k)
            start_t_iter = datetime.now()

        if verbose_level >= 1 and k%100 ==0:
            start_t_calc_sched = datetime.now()
        # Choose randomly a subset of coordinates.
        if schedule == 'constant':
            tau_schedule = tau
        elif schedule == 'exponential':
            tau_schedule = min(int(np.floor(tau+c*np.exp(exp*k))),len(w_0))
        elif schedule == 'adaptive':
            # c = 10000
            if k == 0:
                hn = 1

            eps_1 =  1e4* c * hn**2
            # eps_2 = 10000000 * c * hn
            eps_2 =  1e16   * c * hn**2
            print('eps_1^2:', eps_1**2)
            print('eps_2:', eps_2)
            # print('grad_est: ', np.linalg.norm(grad_est))
            # print('Hess_norm_est: ', np.linalg.norm(hessian_F_est))
            adapt_sched_term1 = 1-((eps_1)/(np.linalg.norm(grad_est)**2))
            adapt_sched_term2 = np.sqrt(1 - (eps_2 / (hessian_F_est**2)))
            print('term_1: ', adapt_sched_term1)
            print('term_2^2: ', 1 - eps_2/(hessian_F_est**2))
            # print('h^4/grad^2: ', hn**4/(np.linalg.norm(grad_est)**2))
            if k == 0:
                tau_schedule = min(max(int(len(w_0)*max(adapt_sched_term1, 
                                                        adapt_sched_term2)
                                        ), 
                                        1),
                                    len(w_0)
                                    )
            else:
                tau_schedule = int((1-beta)*tau_schedule + beta*min(max(int(len(w_0)*max(adapt_sched_term1, 
                                                                                        adapt_sched_term2)
                                                                            ), 
                                                                        1),
                                                                        len(w_0)
                                                                    ))
            print('tau(S_%d) = %d' %(k, tau_schedule))
        else:
            print('Unknown schedule type. Using constant coordinate schedule.')
            tau_schedule = tau

        if verbose_level >= 1 and k%100 ==0:
            print('time calc sched.:', (datetime.now() - start_t_calc_sched).total_seconds())
                    
        num_coord += tau_schedule**2 + tau_schedule
        
        # grad_k = grad(w_k, X, Y, np.arange(n)) # calculate full gradient to check convergence 
          
        S = np.random.choice(n, tau_schedule, replace=False)
        X_S = X[:, S]
        
        if verbose_level >= 1 and k%100 ==0:
            start_t_calc_grad = datetime.now()
        grad_k_S = grad(w_k, X, Y, S)
        if verbose_level >= 1 and k%100 ==0:
            print('time calc grad:', (datetime.now() - start_t_calc_grad).total_seconds())

        if k == 0:
            grad_est = np.sqrt(n/len(S))*np.linalg.norm(grad_k_S)
        else:
            grad_est = alpha*np.linalg.norm(grad_k_S) + (1-alpha)*np.linalg.norm(grad_est)
        # grad_est = np.sqrt(n/len(S))*np.linalg.norm(grad_k_S)
        
        if schedule == 'adaptive':
            # hessian_F_est = np.sqrt(n**2/len(S)**2)*np.linalg.norm(hessian(w_k, X, Y, S),2)# calculate the 2-norm for the adaptive schedule
            
            if k == 0:
                hessian_F_est = np.linalg.norm(hessian(w_k, X, Y, S),2)# calculate the 2-norm for the adaptive schedule
            else:
                hessian_F_est = alpha*np.linalg.norm(hessian(w_k, X, Y, S),2) + (1-alpha)*np.linalg.norm(hessian_F_est)
          
        
        
        hess_v = lambda v: hess_vec(w_k, X, Y, S, v)
        # hess = lambda w: hessian(w, X, Y, S)
        if verbose_level >= 1 and k%100 ==0:
            start_t_calc_hess = datetime.now()
        hess_k_S = hessian(w_k, X, Y, S)
        if verbose_level >= 1 and k%100 ==0:
            print('time for calc Hess:', (datetime.now() - start_t_calc_hess).total_seconds())
        
        if verbose_level >= 1 and k%100 ==0:
            start_t_subsolv = datetime.now()
        (h,lambda_k) = solve_ARC_subproblem(solver,
                                            grad_k_S, hess_v, hess_k_S,
                                            H_k, w_k[S],
                                            successful_flag=False,lambda_k=lambda_k,
                                            exact_tol=1e-5,krylov_tol=1e-5,solve_each_i_th_krylov_space=1, 
                                            keep_Q_matrix_in_memory=True)
        if verbose_level >= 1 and k%100 ==0:
            print('time for subsolver:', (datetime.now() - start_t_subsolv).total_seconds())

        if trace and k%log_every==0:
            history['norm_s_k'].append(np.linalg.norm(h))
            if schedule == 'adaptive':
                history['hess_F_est'].append(hessian_F_est)
                history['adapt_sc_term1'].append(adapt_sched_term1)
                history['adapt_sc_term2'].append(adapt_sched_term2)
            history['tau'].append(tau_schedule)
        
        tmp_w_k = w_k.copy()
        tmp_w_k[S] += h
        
        if verbose_level >= 1 and k%100==0:
            start_t_f_eval = datetime.now()
        func_T = loss(tmp_w_k,X,Y)

        if verbose_level >= 1 and k%100==0:
            print('time for func eval.: ', (datetime.now() - start_t_f_eval).total_seconds())
        # func_T = loss(tmp_w_k[S],X_S,Y)

        #### III: Regularization Update #####

        function_decrease = func_k - func_T
        hn = np.linalg.norm(h)
        
        if verbose_level >= 1 and k%100==0:
            start_t_model_decr = datetime.now()
        model_decrease= -(np.dot(grad_k_S, h) + 0.5 * np.dot(h, hess_v(h))+ 1/6*H_k*hn**3)

        if verbose_level >= 1 and k%100==0:
            print('time for calc model decr. (Hess-vec prod.): ', (datetime.now() - start_t_model_decr).total_seconds())
        rho = function_decrease / model_decrease
        assert (model_decrease >=0), 'negative model decrease. This should not have happened'

        if verbose_level >= 1 and k%1==0:
            print('func_k: ', func_k)
            print('grad_norm_est: ', np.linalg.norm(grad_est))
        
        if verbose_level >= 2:
            print('func_T: ', func_T)
            print('iter: ', k)
            print('model decr.: ', model_decrease)
            print('func decr. : ', function_decrease)
            print('rho: ', rho)
            # print('norm(grad_k): ', np.linalg.norm(grad_k))
            print('norm(grad_k_S): ', np.linalg.norm(grad_k_S))
#               
            print('||h||: ', hn)
            print('norm(w_k)', np.linalg.norm(w_k))
            print('norm(tmp_w_k): ', np.linalg.norm(tmp_w_k))
            print('H_k: ', H_k)
        
        if verbose_level >=3:
            print('tmp_w_k', tmp_w_k)
            print('w_k:', w_k)
        
        
        # Update w if step s is successful
        if rho >= eta_1 and function_decrease >= 0:
            
            # Update the current point.
            w_k[S] += h
            func_k = func_T
            func_S_k = func_T
            successful_flag=True
        else:
            # history['accept'].append(0)
            func_k=func_k  

        #Update penalty parameter
        if rho >= eta_2:
            H_k=max(H_k/gamma_2,1e-10)
            #alternative (Cartis et al. 2011): sigma= max(min(grad_norm,sigma),np.nextafter(0,1)) 
        
        elif rho < eta_1 or np.isnan(rho):
            H_k = gamma_1*H_k #min(gamma_1*H_k, 1e20)
            successful_flag=False   
#             print ('unscuccesful iteration')  

        if np.linalg.norm(grad_est) <= tolerance:
            status = 'success'
            if tolerance_passed == False:
                tolerance_iter = k
            tolerance_passed = True
            
            if k - tolerance_iter >= 0:
                break

        if k == max_iter:
            status = 'iterations_exceeded'
            break
            
        if verbose_level >= 1 and k%100 ==0:
            print('time per iteration:', (datetime.now() - start_t_iter).total_seconds())


    return w_k, status, history

def do_experiment_logreg(A, b, x_0,
                            loss_func, gradient, Hv, hess, 
                            optimizer, solver, max_iter, tolerance,
                            rep, lams, taus, schedule='constant',                            
                            cs=[1.0], exps=[1.0], 
                            eps_1=1e-2, eps_2=1e-2,
                            betas=[0.2],
                            log_every=100):
    print('Experiment: \t n = %d.' % (A.shape[1]))    
        
    results = []

    if optimizer == 'SSCN':
        print('Optimizing with SSCN...')
        optimization_method = coordinate_cubic_newton_new
    elif optimizer == 'CD':
        print('Optimizing with CD...')
        optimization_method = coordinate_gradient_method
    else:
        ValueError('Unknown optimizer specified.')
    
    for i in range(rep):
        SSCN_results = []
        
        for lam in lams:
            for ind,tau in enumerate(taus):

                def loss(x, A, b):
                    return loss_func(x, A, b, lam) 

                def grad_x(x, A, b, S):
                    return gradient(x, A, b, S, lam)

                def hess_vec(x, A, b, S, h):
                    return Hv(x, A, b, S, h, lam)

                def hessian(x, A, b, S):
                    return hess(x, A, b, S, lam)

                

                if schedule == 'exponential':
                    for exp in exps:
                        for c in cs:
                            
                            start_timestamp = datetime.now()

                            w_k, status, history = \
                                optimization_method(solver, loss, grad_x, hess_vec, hessian, A, b, x_0, 
                                                    tolerance=tolerance, 
                                                    tau=tau,
                                                    max_iter=max_iter,
                                                    H_0 = 1.0, line_search=True, 
                                                    c=c, exp=exp,
                                                    schedule=schedule, log_every=log_every[ind])
                            t_secs = (datetime.now() - start_timestamp).total_seconds()
                            print(('%s with exponential schedule tau+c*e^(d*k) \t : lambda %.4f \t tau %d \t status \
                                   %s \t time %.4f \t c %.4f \t d %.4f'  % 
                                  (optimizer, lam, tau, status, t_secs, c, exp)), flush=True)
                            SSCN_results.append(history)
                elif schedule == 'adaptive':
                    for c in cs:
                        for beta in betas:
                            start_timestamp = datetime.now()
                            if optimizer == 'CD':
                                ValueError('Adaptive schedule is only possible for SSCN, but CD was chosen as optimizer.')
                            w_k, status, history = \
                                optimization_method(solver, loss, grad_x, hess_vec, hessian, A, b, x_0,
                                                        tolerance=tolerance, 
                                                        tau=tau,
                                                        max_iter=max_iter,
                                                        H_0 = 1.0, line_search=True,
                                                        c=c,
                                                        schedule=schedule,
                                                        eps_1=eps_1, eps_2=eps_2, log_every=log_every[ind],
                                                        beta=beta)
                            t_secs = (datetime.now() - start_timestamp).total_seconds()
                            print(('%s with adaptive schedule \t : lambda %.4f \t status \
                                    %s \t time %.4f'  % (optimizer, lam, status, t_secs)), flush=True)
                            SSCN_results.append(history)
                else:
                    start_timestamp = datetime.now()
                    w_k, status, history = \
                            optimization_method(solver, loss, grad_x, hess_vec, hessian, A, b, x_0,
                                                    tolerance=tolerance, 
                                                    tau=tau,
                                                    max_iter=max_iter,
                                                    H_0 = 1.0, line_search=True, schedule=schedule,
                                                    log_every=log_every[ind])
                    t_secs = (datetime.now() - start_timestamp).total_seconds()
                    print(('%s with const schedule tau \t : lambda %.4f \t tau %d \t status \
                           %s \t time %.4f'  % 
                          (optimizer, lam, tau, status, t_secs)), flush=True)
                    SSCN_results.append(history)
                    
        results.append(SSCN_results)

    return w_k,results

# logistic regression Experiment: Defining loss, gradient, Hessian-vector product and Hessian

# loss = logistic_loss_nonconvex

# def grad_x(w, X, Y, S, alpha=1e-3, beta=1):
#     n = X.shape[0]
#     d = X.shape[1]
#     z = X.dot(w)   # prediction <w, x>
#     h = phi(z)
#     w_s = w[S]
#     grad= X[:,S].T.dot(h-Y)/n
#     grad = grad + alpha*np.multiply(2*beta*w_s,(1+beta*w_s**2)**(-2))
#     return grad

# def hessian( w, X, Y, S, alpha=1e-3,beta=1):
#     n = X.shape[0]
#     d = X.shape[1]
#     s = len(S)
#     z= X.dot(w)
#     q=phi(z)
#     h= q*(1-phi(z))
#     X_S = X[:,S]
#     w_s = w[S]
#     H = np.dot(np.transpose(X_S),h[:, np.newaxis]* X_S) / n  
#     H = H + alpha * np.eye(s,s)*np.multiply(2*beta-6*beta**2*w_s**2,(beta*w_s**2+1)**(-3))
#     return H

# def hess_vec(w, X, Y, S, v,alpha=1e-3,beta=1): 
#     n = X.shape[0]
#     d = X.shape[1]
#     X_S = X[:,S]
#     w_s = w[S]
#     _z=X.dot(w)
#     _z = phi(-_z)
#     d_binary = _z * (1 - _z)
#     wa = d_binary * X_S.dot(v)
#     Hv = X_S.T.dot(wa)/n
#     out = Hv + alpha *np.multiply(np.multiply(2*beta-6*beta**2*w_s**2,(beta*w_s**2+1)**(-3)), v)
#     return out

# Support vector regression as done in Fuji et al. RANDOMIZED SUBSPACE REGULARIZED NEWTON METHOD FOR THE UNCONSTRAINED NON-CONVEX OPTIMIZATION
loss = support_vector_regression_cauchy_loss
pr = eta
grad_p = grad_eta
hess_p = hess_eta
# loss = support_vector_regression_geman_mcclure_loss
# pr = geman_mcclure
# grad_p = grad_geman_mcclure
# hess_p = hess_geman_mcclure

def grad_x(w, X, Y, S, alpha=1e-3, beta=1):
    n = X.shape[0]
    d = X.shape[1]
    P = X.dot(w)
    z = pr(P)
    zz = grad_p(Y-P) #phi(P)*(1-phi(P)) # phi'(P)

    w_s = w[S]
    X_S = X[:,S]
    grad = -np.dot(X_S.T, zz)/n + alpha * w_s

    #     grad = (-zz*(X.T.dot(Y)-np.dot(X.T,z)))/n
    # grad = -np.dot(X_S.T, zz*(Y-z))/n + alpha * w_s
    return grad

def hessian(w, X, Y, S, alpha=1e-3, beta=1):
    n = X.shape[0]
    d = X.shape[1]
    P = X.dot(w)
    z = pr(P)
    zz = grad_p(P) #phi(P)*(1-phi(P)) # phi'(P)
    zzz = hess_p(Y-P) #phi(P) - 3*phi(P)**2 + 2*phi(P)**3 # phi''(P)

    s = len(S)
    X_S = X[:,S]
    w_s = w[S]
    # H = np.average([(zz[i]**2 - (Y[i] - z[i])*zzz[i]) * np.outer(X_S[i,:],X_S[i,:])  for i in range(len(Y))], axis=0) \
    #     + (alpha * np.eye(s, s))
    H =  X_S.T @ (np.diag(zzz)@ X_S)\
        + (alpha * np.eye(s, s))

    return H

def hess_vec(w, X, Y, S, v, alpha=1e-3, beta=1):
    n = X.shape[0]
    d = X.shape[1]
    P = X.dot(w)
    z = pr(P)#phi(P)
    zz = grad_p(P)#phi(P)*(1-phi(P)) # phi'(P)
    zzz = hess_p(Y-P) #phi(P) - 3*phi(P)**2 + 2*phi(P)**3 # phi''(P)

    s = len(S)
    X_S = X[:,S]
    w_s = w[S]

    # Hv = np.average([(zz[i]**2 - (Y[i] - z[i])*zzz[i]) * np.outer(X_S[i,:],X_S[i,:])  for i in range(len(Y))],axis=0) @ v \
    #    + alpha * v
    
    Hv = (X_S.T @ (np.diag(zzz)@ X_S) + (alpha * np.eye(s, s))) @ v + alpha * v
            
    return Hv

# Non-linear least squares 
# loss = nnls_geman_mcclure_loss
# pr = geman_mcclure
# grad_p = grad_geman_mcclure
# hess_p = hess_geman_mcclure
# loss = nnls_cauchy_loss
# pr = eta
# grad_p = grad_eta
# hess_p = hess_eta

# def grad_x(w, X, Y, S, alpha=1e-3, beta=1):
#     n = X.shape[0]
#     d = X.shape[1]
#     P = X.dot(w)
#     z = pr(P)
#     zz = grad_p(P) 

#     w_s = w[S]
#     X_S = X[:,S]

#     #     grad = (-zz*(X.T.dot(Y)-np.dot(X.T,z)))/n
#     grad = -np.dot(X_S.T, zz*(Y-z))/n + alpha * w_s
#     return grad

# def hessian(w, X, Y, S, alpha=1e-3, beta=1):
#     n = X.shape[0]
#     d = X.shape[1]
#     P = X.dot(w)
#     z = pr(P)
#     zz = grad_p(P) 
#     zzz = hess_p(P) #phi(P) - 3*phi(P)**2 + 2*phi(P)**3 # phi''(P)

#     s = len(S)
#     X_S = X[:,S]
#     H = np.average([(zz[i]**2 - (Y[i] - z[i])*zzz[i]) * np.outer(X_S[i,:],X_S[i,:])  for i in range(len(Y))], axis=0) \
#         + (alpha * np.eye(s, s))

#     return H

# def hess_vec(w, X, Y, S, v, alpha=1e-3, beta=1):
#     n = X.shape[0]
#     d = X.shape[1]
#     P = X.dot(w)
#     z = pr(P)#phi(P)
#     zz = grad_p(P)#phi(P)*(1-phi(P)) # phi'(P)
#     zzz = hess_p(P) #phi(P) - 3*phi(P)**2 + 2*phi(P)**3 # phi''(P)

#     s = len(S)
#     X_S = X[:,S]
#     Hv = np.average([(zz[i]**2 - (Y[i] - z[i])*zzz[i]) * np.outer(X_S[i,:],X_S[i,:])  for i in range(len(Y))],axis=0) @ v \
#        + alpha * v
                
#     return Hv




# TODO: Doesn't seem to work like this. Need to change the above definitions for each loss function separately!!!
# def setup(loss_func):
#     '''
#     Wrapper to return function, gradient function and Hessian function from specified string loss_func
#     loss_func: string
#     '''
#     try:
#         loss = getattr(loss_functions, loss_func + "_loss")
#         # print(loss)
#         grad_x_full = getattr(loss_functions, loss_func + "_gradient")
#         # print(grad_x)
#         hessian_full = getattr(loss_functions, loss_func + "_hessian")
#         # print(hessian)
#         hess_vec_full = getattr(loss_functions, loss_func + "_Hv")
#         # print(hess_vec)
#         # print(f, grad_f, hess_f)
#     except:
#         raise ValueError('Unknown loss function specified!')

#     return loss, grad_x_full, hessian_full, hess_vec_full

def main(config):

    start_time = datetime.now()

    ## Loading dataset
    dataset = config['dataset']
    rep = config['repetitions']
    seed = config['seed']

    print('Loading data...')

    try:
        print(dataset)
        data_path = config['data_path']

        datapath = data_path + dataset + '.txt'

        data = load_svmlight_file(datapath)
        A, b = data[0].toarray(), data[1]

        A = A - np.mean(A,axis=0)
        A = A/np.std(A,axis=0)
        # b = b/np.std(b,axis=0)

        # print('statistics')
        # print(np.mean(A,axis=0),np.std(A,axis=0))
        # print(np.mean(b),np.std(b))

        # breakpoint()
        # b[b==-1] = 0
        
        n = A.shape[1]
        
        print(A.shape)
        np.random.seed(seed)
            
        x_0 = np.random.rand(n)
        # x_0 = np.zeros(n) 

        # breakpoint()
        # x_0 = np.abs(np.random.randn(n)) # 
        # x_0 = x_0/np.linalg.norm(x_0)   #
        
    except:
        ValueError('Unknown dataset specified.')

    np.random.seed(seed)

    tolerance = config['tolerance']
    max_iter = config['max_iterations']
    optimizer = config['optimizer'] # either CD or SSCN

    # if optimizer==SSCN
    solver = config['subsolver']

    lams = config['lambdas']
    taus = config['taus']

    loss_func = config['loss_func']

    # loss, grad_x, hessian, hess_vec = setup(loss_func)

 
    print('time ', (datetime.now()-start_time).total_seconds())

    print('Starting experiment...')

    schedule = config['coordinate_schedule']

    if schedule == 'constant':
        _, SSCN_logreg= do_experiment_logreg(A, b, x_0, loss, grad_x, hess_vec, hessian, 
                                                    optimizer, solver, max_iter, tolerance,
                                                    rep, lams, taus, schedule=config['coordinate_schedule'], 
                                                    log_every=config['log_every'])
    elif schedule == 'exponential':
        _, SSCN_logreg= do_experiment_logreg(A, b, x_0, loss, grad_x, hess_vec, hessian, 
                                                    optimizer, solver, max_iter, tolerance,
                                                    rep, lams, taus, schedule=config['coordinate_schedule'], 
                                                    cs=config['cs'], exps=config['exps'],
                                                    log_every=config['log_every'])
    elif schedule == 'adaptive':      
        _, SSCN_logreg= do_experiment_logreg(A, b, x_0, loss, grad_x, hess_vec, hessian, 
                                                    optimizer, solver, max_iter, tolerance,
                                                    rep, lams, taus, schedule=config['coordinate_schedule'], 
                                                    betas=config['beta'],
                                                    eps_1=config['eps_1'], eps_2=config['eps_2'],
                                                    log_every=config['log_every'])
    
    # save results in numpy file: 

    if optimizer == 'SSCN':
        outputfile = 'results_data/%s_%s_%s_SSCN_%s_subsolver_n=%d_lam=%.3f_%s_schedule.npy' % (loss_func, config['experiment_name'], dataset, solver, n, lams[0], config['coordinate_schedule'])
    elif optimizer == 'CD':
        outputfile = 'results_data/%s_%s_%s_CD_n=%d_lam=%.3f_%s_schedule.npy' % (loss_func, config['experiment_name'], dataset, n, lams[0], config['coordinate_schedule'])


    save_run(outputfile, A, b, lams[0], 1, SSCN_logreg)
    
    # to load data again:
    # A, b, lam, mu, beta, results = load_run(outputfile)
    
    context = 'poster'
    
    # Plotting results (w.r.t. #iterations, time and #coordinates^2)
    if optimizer == 'SSCN':
        method = 'SSCN_' + solver 
        plotting_wrapper(config, outputfile, SSCN_logreg, 'num_coord', '# (Coordinates$^2$ + Coordinates)', figurename='%s_ds=%s_n=%d_lam=%.3f_%s' % (loss_func, dataset, n, lams[0], method), save_fig=True, context=context)

    else:
        method = 'CD'
        # plotting_wrapper(config, outputfile, SSCN_logreg, 'num_coord', '# Coordinates', figurename='%s_ds=%s_n=%d_lam=%.3f_%s' % (loss_func, dataset, n, lams[0], method), save_fig=True, context=context)

    plotting_wrapper(config, outputfile, SSCN_logreg, None, 'Iterations $k$', figurename='%s_ds=%s_n=%d_lam=%.3f_%s' % (loss_func, dataset, n, lams[0], method), save_fig=True, context=context)

    plotting_wrapper(config, outputfile, SSCN_logreg, 'time', 'Time, s', figurename='%s_ds=%s_n=%d_lam=%.3f_%s' % (loss_func, dataset, n, lams[0], method), save_fig=True, context=context)

    # plotting_wrapper(config, outputfile, SSCN_logreg, 'norm_s_k', 'Iterations $k$', figurename='%s_ds=%s_n=%d_lam=%.3f_%s' % (loss_func, dataset, n, lams[0], method), subfigures=True, save_fig=False, context=context)

    # plotting_wrapper(config, outputfile, SSCN_logreg, 'tau', 'Iterations $k$', figurename='%s_ds=%s_n=%d_lam=%.3f_%s' % (loss_func, dataset, n, lams[0], method), subfigures=True, save_fig=True, context=context)

    if config['coordinate_schedule'] == 'adaptive':
        plotting_wrapper(config, outputfile, SSCN_logreg, 'hess_F_est', 'Iterations $k$', figurename='%s_ds=%s_n=%d_lam=%.3f_%s' % (loss_func, dataset, n, lams[0], method), subfigures=True, save_fig=True, save_grads=False, context=context)

    plotting_wrapper(config, outputfile, SSCN_logreg, 'func_full', 'Iterations $k$', figurename='%s_ds=%s_n=%d_lam=%.3f_%s' % (loss_func, dataset, n, lams[0], method), subfigures=True, save_fig=True, context=context)

    # plotting_wrapper(config, outputfile, SSCN_logreg, 'grad_est', 'Iterations $k$', figurename='%s_ds=%s_n=%d_lam=%.3f_%s' % (loss_func, dataset, n, lams[0], method), subfigures=True, save_fig=True, context=context)


if __name__ == '__main__':


    # plt.rcParams["figure.figsize"] = (5,5)

    config_files = ['config_E2006_SSCN'] 

    for conf_file in config_files:
        # load in YAML configuration
        config = {}
        base_config_path =  'config_files/' + conf_file + '.yaml'
        with open(base_config_path, 'r') as file:
            config.update(yaml.safe_load(file))

        # start training with name and config 
        main(config)