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
from plotting import *
from utils import *

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")


def coordinate_cubic_newton_new(solver, loss, grad, hess_vec, hessian, X, Y, w_0, 
                                tolerance, tau=1,
                                max_iter=10000, H_0=1.0, line_search=True, 
                                trace=True, schedule='constant',scale_lin=1.0, scale_quad=1.0, c=1.0, exp=0.05, eps_1=1e-2, eps_2=1e-2, jump_iter=1, jump_coord=1, verbose_level=0):

    history = defaultdict(list) if trace else None
    start_timestamp = datetime.now()
    w_k = np.copy(w_0)
    H_k = H_0
    n = w_k.shape[0]
    num_coord = 0 # total number of evaluated coordinates squared
    
    lambda_k = 0
    eta_1 = 0.1 # opt.get('success_treshold',0.1)
    eta_2 = 0.9 # opt.get('very_success_treshold',0.9)
    gamma_1 = 1.5 # opt.get('penalty_increase_multiplier',2.)
    gamma_2 = 2. # opt.get('penalty_derease_multiplier',2.)
    

    func_k = loss(w_k,X,Y)
    func_S_k = np.copy(func_k)
    
    

    # The whole gradient can be computed as:
    # grad_k = grad(w_k, X, Y, np.arange(n))

    if schedule == 'adaptive':
        hessian_F = np.linalg.norm(hessian(w_k, X, Y, np.arange(n)),'fro')

    tolerance_passed = False

    for k in range(max_iter + 1):

        # Choose randomly a subset of coordinates.
        if schedule == 'constant':
            tau_schedule = tau
        elif schedule == 'linear':
            tau_schedule = min(int(np.floor(tau+scale_lin*k)),len(w_0))
        elif schedule == 'quadratic':
            tau_schedule = min(int(np.floor(tau+scale_quad*k**2)),len(w_0))
        elif schedule == 'exponential':
            tau_schedule = min(int(np.floor(tau+c*np.exp(exp*k))),len(w_0))
        # elif schedule == 'adaptive':
        #     print('term_1: ', 1-(eps_1**2/(np.linalg.norm(grad_k,2)**2)))
        #     print('term_2: ',np.sqrt(1 - eps_2/(hessian_F**2)))
        #     tau_schedule = min(int(len(w_0)*max(1-eps_1**2/(np.linalg.norm(grad_k,2)**2), 
        #                                     np.sqrt(1 - eps_2/(hessian_F**2)))),
        #                        len(w_0))
        #     print('tau(S_%d) = %d' %(k, tau_schedule))
        elif schedule == 'jump':
            if k <= jump_iter:
                tau_schedule = tau
            else:
                tau_schedule = jump_coord
        else:
            print('Unknown schedule type. Using constant coordinate schedule.')
            tau_schedule = tau
                    
        num_coord += tau_schedule**2 + tau_schedule
        
        # grad_k = grad(w_k, X, Y, np.arange(n)) # calculate full gradient to check convergence 
        
        if schedule == 'adaptive':
            hessian_F = np.linalg.norm(hessian(w_k, X, Y, np.arange(n)),'fro')# calculate the Frobenius norm for the adaptive schedule
            
        S = np.random.choice(n, tau_schedule, replace=False)
        X_S = X[:, S]
        
        grad_k_S = grad(w_k, X, Y, S)

        alpha = 0.9
        if k == 0:
            grad_est = grad_k_S
        else:
            grad_est = alpha*grad_k_S + (1-alpha)*grad_est
        
        
        
        hess_v = lambda v: hess_vec(w_k, X, Y, S, v)
        # hess = lambda w: hessian(w, X, Y, S)
        hess_k_S = hessian(w_k, X, Y, S)
        
        (h,lambda_k) = solve_ARC_subproblem(solver,
                                            grad_k_S, hess_v, hess_k_S,
                                            H_k, w_k[S],
                                            successful_flag=False,lambda_k=lambda_k,
                                            exact_tol=1e-15,krylov_tol=1e-10,solve_each_i_th_krylov_space=1, 
                                            keep_Q_matrix_in_memory=True)
    
        
        
        if trace:
            history['w_k'].append(w_k.copy())
            history['grad_S'].append(np.linalg.norm(grad_k_S))
            history['grad_est'].append(np.linalg.norm(grad_est))
            # history['grad'].append(np.linalg.norm(grad_k))
            history['func_full'].append(func_k)
            history['func_S'].append(func_S_k)
            history['time'].append(
                (datetime.now() - start_timestamp).total_seconds())
            history['num_coord'].append(num_coord)
            history['norm_s_k'].append(np.linalg.norm(h))
            history['norm_s_k_squared'].append(np.linalg.norm(h)**2)
            history['H'].append(H_k)
        
        tmp_w_k = w_k.copy()
        tmp_w_k[S] += h
        
        func_T = loss(tmp_w_k,X,Y)
        func_S_T = loss(tmp_w_k[S],X_S,Y)

        #### III: Regularization Update #####

        function_decrease = func_k - func_T
        hn = np.linalg.norm(h)
        
        
        model_decrease= -(np.dot(grad_k_S, h) + 0.5 * np.dot(h, hess_v(h))+ 1/3*H_k*hn**3)
        

        
        rho = function_decrease / model_decrease
        assert (model_decrease >=0), 'negative model decrease. This should not have happened'

        
        # for debugging
        if verbose_level >= 1:
            print('iter: ', k)
            print('model decr.: ', model_decrease)
            print('func decr. : ', function_decrease)
            print('rho: ', rho)
            # print('norm(grad_k): ', np.linalg.norm(grad_k))
            print('norm(grad_k_S): ', np.linalg.norm(grad_k_S))
#               print('func_T: ', func_T)
            print('||h||: ', hn)
            print('norm(w_k)', np.linalg.norm(w_k))
            print('norm(tmp_w_k): ', np.linalg.norm(tmp_w_k))
            print('H_k: ', H_k)
        
        
        # Update w if step s is successful
        if rho >= eta_1:
            history['accept'].append(1)
            
            # Update the current point.
            w_k[S] += h
            func_k = func_T
            func_S_k = func_S_T
            successful_flag=True
        else:
            history['accept'].append(0)
            func_k=func_k  

        #Update penalty parameter
        if rho >= eta_2:
            H_k=max(H_k/gamma_2,1e-10)
            #alternative (Cartis et al. 2011): sigma= max(min(grad_norm,sigma),np.nextafter(0,1)) 
        
        elif rho < eta_1:
            H_k = min(gamma_1*H_k, 1e20)
            successful_flag=False   
#             print ('unscuccesful iteration')  

        if np.linalg.norm(grad_est) <= tolerance:
            status = 'success'
            if tolerance_passed == False:
                tolerance_iter = k
            tolerance_passed = True
            
            if k - tolerance_iter >= 300:
                break

        if k == max_iter:
            status = 'iterations_exceeded'
            break
                

    return w_k, status, history

def do_experiment_logsumexp(A, b, x_0,
                            loss_func, gradient, Hv, hess, 
                            optimizer, solver, max_iter, tolerance,
                            rep, mu, lams, taus, schedule='constant', 
                            scales_lin=[1.0], 
                            scales_quad=[1.0], 
                            cs=[1.0], exps=[1.0], 
                            eps_1=1e-2, eps_2=1e-2,
                            jump_iters=[1], jump_coord=1):
    print('Experiment: \t n = %d, \t mu = %f.' % (A.shape[1], mu))    
        
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
            for tau in taus:

                def loss(x, A, b):
                    return loss_func(x, A, b, lam, mu) 

                def grad_x(x, A, b, S):
                    return gradient(x, A, b, S, lam, mu)

                def hess_vec(x, A, b, S, h):
                    return Hv(x, A, b, S, h, lam, mu)

                def hessian(x, A, b, S):
                    return hess(x, A, b, S, lam, mu)

                start_timestamp = datetime.now()

                if schedule == 'exponential':
                    for exp in exps:
                        for c in cs:

                            w_k, status, history = \
                                optimization_method(solver, loss, grad_x, hess_vec, hessian, A, b, x_0, 
                                                    tolerance=tolerance, 
                                                    tau=tau,
                                                    max_iter=max_iter,
                                                    H_0 = 1.0, line_search=True, 
                                                    c=c, exp=exp,
                                                    schedule=schedule)
                            t_secs = (datetime.now() - start_timestamp).total_seconds()
                            print(('SSCN with exponential schedule tau+c*e^(d*k) \t : lambda %.4f \t tau %d \t status \
                                   %s \t time %.4f \t c %.4f \t d %.4f'  % 
                                  (lam, tau, status, t_secs, c, exp)), flush=True)
                            SSCN_results.append(history)
                elif schedule == 'linear':
                    for scale in scales_lin:
                        w_k, status, history = \
                            optimization_method(solver, loss, grad_x, hess_vec, hessian, A, b, x_0, 
                                                    tolerance=tolerance, 
                                                    tau=tau,
                                                    max_iter=max_iter,
                                                    H_0 = 1.0, line_search=True, 
                                                    scale_lin=scale,
                                                    schedule=schedule)
                        t_secs = (datetime.now() - start_timestamp).total_seconds()
                        print(('SSCN with linear schedule tau+s*k \t : lambda %.4f \t tau %d \t status \
                               %s \t time %.4f \t c %.4f'  % 
                              (lam, tau, status, t_secs, scale)), flush=True)
                        SSCN_results.append(history)
                elif schedule == 'quadratic':
                    for scale in scales_quad:
                        w_k, status, history = \
                            optimization_method(solver, loss, grad_x, hess_vec, hessian, A, b, x_0,
                                                    tolerance=tolerance, 
                                                    tau=tau,
                                                    max_iter=max_iter,
                                                    H_0 = 1.0, line_search=True, 
                                                    scale_quad=scale, 
                                                    schedule=schedule)
                        t_secs = (datetime.now() - start_timestamp).total_seconds()
                        print(('SSCN with quadratic schedule tau+s*k^2 \t : lambda %.4f \t tau %d \t status \
                               %s \t time %.4f \t c %.4f'  % 
                              (lam, tau, status, t_secs, scale)), flush=True)
                        SSCN_results.append(history)
                elif schedule == 'adaptive':
                    if optimizer == 'CD':
                        ValueError('Adaptive schedule is only possible for SSCN, but CD was chosen as optimizer.')
                    w_k, status, history = \
                        optimization_method(solver, loss, grad_x, hess_vec, hessian, A, b, x_0,
                                                tolerance=tolerance, 
                                                tau=tau,
                                                max_iter=max_iter,
                                                H_0 = 1.0, line_search=True,
                                                schedule=schedule,
                                                eps_1=eps_1, eps_2=eps_2)
                    t_secs = (datetime.now() - start_timestamp).total_seconds()
                    print(('SSCN with adaptive schedule'), flush=True)
                    SSCN_results.append(history)
                elif schedule == 'jump':
                    for _, jump_iter in enumerate(jump_iters):
                        w_k, status, history = \
                            optimization_method(solver, loss, grad_x, hess_vec, hessian, A, b, x_0,
                                                    tolerance=tolerance, 
                                                    tau=tau,
                                                    max_iter=max_iter,
                                                    H_0 = 1.0, line_search=True,
                                                    schedule=schedule,
                                                    jump_iter=jump_iter, jump_coord=jump_coord)
                        t_secs = (datetime.now() - start_timestamp).total_seconds()
                        print(('SSCN with jump schedule tau -> jump_coord \t: lambda %.4f \t tau %d \t jump_iter %d \t jump_coord %d \t status \
                               %s \t time %.4f' %(lam, tau, jump_iter, jump_coord, status, t_secs)), flush=True)
                        SSCN_results.append(history)
                else:
                    w_k, status, history = \
                            optimization_method(solver, loss, grad_x, hess_vec, hessian, A, b, x_0,
                                                    tolerance=tolerance, 
                                                    tau=tau,
                                                    max_iter=max_iter,
                                                    H_0 = 1.0, line_search=True, schedule=schedule)
                    t_secs = (datetime.now() - start_timestamp).total_seconds()
                    print(('SSCN with const schedule tau \t : lambda %.4f \t tau %d \t status \
                           %s \t time %.4f'  % 
                          (lam, tau, status, t_secs)), flush=True)
                    SSCN_results.append(history)
                    
        results.append(SSCN_results)

    return w_k,results

# logsumexp Experiment: Defining loss, gradient, Hessian-vector product and Hessian
def loss(x, A, b, lam, mu): 
    return mu * logsumexp(1.0 / mu * (A.dot(x) - b)) + lam * np.sum(x**2/(1+x**2))

def grad_x(x, A, b, S, lam, mu):
    mu_inv = 1.0 / mu
    Ax = A.dot(x)
    a = mu_inv * (Ax - b)
    pi = softmax(a)

    return A[:,S].T.dot(pi) + 2*lam*x[S]/((1+x[S]**2)**2)

def hess_vec(x, A, b, S, h, lam, mu):
    mu_inv = 1.0 / mu
    Ax = A.dot(x)
    a = mu_inv * (Ax - b)
    pi = softmax(a)

    grad_S = grad_x(x, A, b, S, lam, mu)

    return mu_inv * (A[:,S].T.dot(pi * A[:,S].dot(h)) - grad_S.dot(h) * grad_S) \
                     + 2*lam * np.diag(( 1 - 3 * x[S]**2 )/( 1 + x[S]**2 )**3).dot(h) 

def hessian(x, A, b, S, lam, mu):
    mu_inv = 1.0 / mu
    Ax = A.dot(x)
    a = mu_inv * (Ax - b)
    pi = softmax(a)

    grad_S = grad_x(x, A, b, S, lam, mu)

    hess = mu_inv * (A[:, S].T.dot(A[:, S] * pi.reshape(-1, 1)) \
              - np.outer(grad_S, grad_S.T)) + 2*lam * np.diag(( 1 - 3 * x[S]**2 )/( 1 + x[S]**2 )**3) 

    return hess

def grad_x_conv_reg(x, A, b, S, lam, mu, beta=1e-5):
    mu_inv = 1.0 / mu
    Ax = A.dot(x)
    a = mu_inv * (Ax - b)
    pi = softmax(a)

    return A[:,S].T.dot(pi) + 2*lam*x[S]/((1+x[S]**2)**2) + beta * x[S]

def hess_vec_conv_reg(x, A, b, S, h, lam, mu, beta=1e-5):
    mu_inv = 1.0 / mu
    Ax = A.dot(x)
    a = mu_inv * (Ax - b)
    pi = softmax(a)

    grad_S = grad_x_conv_reg(x, A, b, S, lam, mu, beta)

    return mu_inv * (A[:,S].T.dot(pi * A[:,S].dot(h)) - grad_S.dot(h) * grad_S) \
                     + 2*lam * np.diag(( 1 - 3 * x[S]**2 )/( 1 + x[S]**2 )**3).dot(h) \
                     + beta * h

def hessian_conv_reg(x, A, b, S, lam, mu, beta=1e-5):
    mu_inv = 1.0 / mu
    Ax = A.dot(x)
    a = mu_inv * (Ax - b)
    pi = softmax(a)

    grad_S = grad_x_conv_reg(x, A, b, S, lam, mu, beta)

    hess = mu_inv * (A[:, S].T.dot(A[:, S] * pi.reshape(-1, 1)) \
              - np.outer(grad_S, grad_S.T)) + 2*lam * np.diag(( 1 - 3 * x[S]**2 )/( 1 + x[S]**2 )**3) + (beta * np.eye(len(S), len(S)))

    return hess

def loss_conv_reg(x, A, b, lam, mu, beta=1e-5):
    return mu * logsumexp(1.0 / mu * (A.dot(x) - b)) + lam * np.sum(x**2/(1+x**2)) + 0.5 * beta * (np.linalg.norm(x) ** 2)

def main(config):


    ## Loading dataset
    dataset = config['dataset']
    mu=config['mu']
    rep = config['repetitions']
    seed = config['seed']

    if dataset == 'synthetic':
        n=config['n']
        
        rep_fac=config['replication_factor']

        if config['correlated_data'] == True:
        # generate correlated data
            print('Generating correlated data...')
            (A, b, x_star, f_star) = generate_logsumexp_w_covariance_matrix(n=n, mu=mu, replication_factor=rep_fac)
        else:
            print('Generating uncorrelated data...')
            (A, b, x_star, f_star) = generate_logsumexp(n=n, mu=mu, replication_factor=rep_fac)

        np.random.seed(seed)
                
        x_0 = np.random.randn(n*rep_fac)
    else:
        try:
            print(dataset)
            data_path = config['data_path']

            datapath = data_path + dataset + '.txt'

            data = load_svmlight_file(datapath)
            A, b = data[0].toarray(), data[1]

            n = A.shape[1]
            
            print(A.shape)
            np.random.seed(seed)
                
            x_0 = np.random.randn(n)
        except:
            ValueError('Unknown dataset specified.')

    # sanity check: check that max(taus, jump_coord) <= n
    assert (max(config['taus']) <= n), 'schedule chosen is larger than the number of dimensions'
    if config['coordinate_schedule'] == 'jump':
        assert (config['jump_coord'] <= n), 'schedule chosen is larger than the number of dimensions'
    

    np.random.seed(seed)

    tolerance = config['tolerance']
    max_iter = config['max_iterations']
    optimizer = config['optimizer'] # either CD or SSCN

    # if optimizer==SSCN
    solver = config['subsolver']

    lams = config['lambdas']
    taus = config['taus']

 


    _, SSCN_logsumexp= do_experiment_logsumexp(A, b, x_0, loss_conv_reg, grad_x_conv_reg, hess_vec_conv_reg, hessian_conv_reg, 
                                                optimizer, solver, max_iter, tolerance,
                                                rep, mu, lams, taus, schedule=config['coordinate_schedule'], 
                                                scales_lin=config['scales_lin'], 
                                                scales_quad=config['scales_quad'], 
                                                cs=config['cs'], exps=config['exps'],
                                                eps_1=config['eps_1'], eps_2=config['eps_2'],
                                                jump_iters=config['jump_iters'], jump_coord=config['jump_coord'])
    
    # save results in numpy file: 
    if optimizer == 'SSCN':
        outputfile = 'results_data/%s_%s_%s_SSCN_%s_subsolver_n=%d_sigma=%.3f_lam=%.3f_%s_schedule.npy' % (config['project_name'], config['experiment_name'], config['dataset'], solver, n, mu, lams[0], config['coordinate_schedule'])
    elif optimizer == 'CD':
        outputfile = 'results_data/%s_%s_%s_CD_n=%d_sigma=%.3f_lam=%.3f_%s_schedule.npy' % (config['project_name'], config['experiment_name'], config['dataset'], n, mu, lams[0], config['coordinate_schedule'])


    save_run(outputfile, A, b, lams[0], mu, 1e-5, SSCN_logsumexp)
    
    # to load data again:
    # A, b, lam, mu, beta, results = load_run(outputfile)
    
    
    # Plotting results (w.r.t. #iterations, time and #coordinates^2)
    if optimizer == 'SSCN':
        method = 'SSCN_' + solver 
        plotting_wrapper(config, outputfile, SSCN_logsumexp, 'num_coord', '# (Coordinates$^2$ + Coordinates)', figurename='logsumexp_ds=%s_n=%d_sigma=%.3f_lam=%.3f_%s' % (dataset, n, mu, lams[0], method), save_fig=True)

    else:
        method = 'CD'
        plotting_wrapper(config, outputfile, SSCN_logsumexp, 'num_coord', '# Coordinates', figurename='logsumexp_ds=%s_n=%d_sigma=%.3f_lam=%.3f_%s' % (dataset, n, mu, lams[0], method), save_fig=True)

    plotting_wrapper(config, outputfile, SSCN_logsumexp, None, 'Iterations $k$', figurename='logsumexp_ds=%s_n=%d_sigma=%.3f_lam=%.3f_%s' % (dataset, n, mu, lams[0], method), save_fig=True)

    plotting_wrapper(config, outputfile, SSCN_logsumexp, 'time', 'Time, s', figurename='logsumexp_ds=%s_n=%d_sigma=%.3f_lam=%.3f_%s' % (dataset, n, mu, lams[0], method), save_fig=True)

    plotting_wrapper(config, outputfile, SSCN_logsumexp, 'norm_s_k', 'Iterations $k$', figurename='logsumexp_ds=%s_n=%d_sigma=%.3f_lam=%.3f_%s' % (dataset, n, mu, lams[0], method), subfigures=True, save_fig=True)

    plotting_wrapper(config, outputfile, SSCN_logsumexp, 'func_full', 'Iterations $k$', figurename='logsumexp_ds=%s_n=%d_sigma=%.3f_lam=%.3f_%s' % (dataset, n, mu, lams[0], method), subfigures=True, save_fig=True)

    # plotting_wrapper(config, outputfile, SSCN_logsumexp, 'grad_est', r'$\nabla f(x_k)_{est}$', figurename='logsumexp_ds=%s_n=%d_sigma=%.3f_lam=%.3f_%s' % (dataset, n, mu, lams[0], method), subfigures=True, save_fig=True)


if __name__ == '__main__':


    plt.rcParams["figure.figsize"] = (5,5)

    config_files = ['config_rep_fac_1']

    for conf_file in config_files:
        # load in YAML configuration
        config = {}
        base_config_path =  conf_file + '.yaml'
        with open(base_config_path, 'r') as file:
            config.update(yaml.safe_load(file))

        # TODO: If I want to add a parser, adapt Code below

        # # TODO: add more if more parameters should be "sweepable"
        # # overwrite with sweep parameters - have to be given in with ArgumentParser for wandb
        # parser = argparse.ArgumentParser(description='Process some integers.')
        # parser.add_argument('--L2_clip', type=float, default=config['L2_clip'], help='L2 clip for DP')
        # args = parser.parse_args()

        # # TODO: check for easy way to convert args to dict to simply update config
        # config['L2_clip'] = args.L2_clip
        # config['max_epochs'] = args.max_epochs

        # start training with name and config 
        main(config)