from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from utils import *
from run_logsumexp_experiments import *

def plotting_wrapper(config, outputfile, results, xparam, xlabel, figurename, subfigures=False, save_fig=False, ind_ends=None):

    project_name = config['project_name']
    experiment_name = config['experiment_name']

    rep = config['repetitions']

    lams = config['lambdas']
    solver = config['subsolver']

    coord_schedule = config['coordinate_schedule']

    A, _, _, mu, _, _ = load_run(outputfile)
    n = A.shape[1]

    if coord_schedule == 'constant':
        taus = config['taus']

        labels = [(r'$\tau = %d$' % (tau)) for tau in taus ]
        title = r'SSCN, n=%d,  $\sigma = %.3f, \lambda= %.3f$, %s' %(n, mu, lams[0], solver)
    elif coord_schedule == 'linear':
        scales_lin = config['scales_lin']
        taus = config['taus']
        tau = taus[0]

        labels = [(r'$c_l= %.2f$' % (scale)) for scale in scales_lin]
        title = r'SSCN, $|S|=\tau + c_l k$, n=%d,  $\sigma = %.3f, \tau = %d, \lambda= %.3f$, %s' %(n, mu, tau, lams[0], solver)
    elif coord_schedule == 'exponential':
        cs = config['cs']
        exps = config['exps']
        taus = config['taus']
        tau = taus[0]

        labels = [(r'$c_e = %.2f, d = %.3f$' % (c, exp)) for c in cs for exp in exps ]
        title = r'SSCN, $|S|=\tau + c_e \exp(d \cdot k)$, n=%d,  $\sigma = %.3f, \tau = %d, \lambda= %.3f$, %s' %(n, mu, tau, lams[0], solver)
    elif coord_schedule == 'quadratic':
        scales_quad = config['scales_quad']
        taus = config['taus']
        tau = taus[0]

        labels = [(r'$c_q= %.5f$' % (scale)) for scale in scales_quad]
        title = r'SSCN, $|S|=\tau + c k^2$, n=%d,  $\sigma = %.3f, \tau = %d, \lambda= %.3f$, %s' %(n, mu, tau, lams[0], solver)        
    elif coord_schedule == 'adaptive':

        labels = [(r'adaptive')]
        title = r'SSCN, n=%d,  $\sigma = %.3f, \tau = %d, \lambda= %.3f$, %s' %(n, mu, tau, lams[0], solver) 
    
    elif coord_schedule == 'jump':
        taus = config['taus']
        tau = taus[0]
        jump_iters = config['jump_iters']
        jump_coord = config['jump_coord']
        labels = [(r'$j_i = %d$') % jump_iter for jump_iter in jump_iters]
        title = r'SSCN, jump schedule, n=%d,  $\sigma = %.3f, \tau = %d, \lambda= %.3f, jump_{coord} = %d$, %s' %(n, mu, tau, lams[0], jump_coord, solver)  
 
    
    plot_results(project_name, experiment_name, outputfile, rep, results, xparam, labels, 
                title, xlabel, figurename, subfigures=subfigures, save_fig=save_fig, ind_ends=ind_ends)
    

def plot_results(project_name, experiment_name, outputfile, rep, results, xparam, labels, 
                title, xlabel, figurename, subfigures=False, save_fig=False, ind_ends=None):
    
    sns.set_theme()

    plt.figure(figsize=(13, 13))
    
    linewidth = 2.5
    alpha = 1
    markeredgewidth=1.5
    markeredgecolor=[0,0,0,0.6]
    markevery= int(len(results[0][0]['time'])/6)+1

    markers = ["<","s","p","D","X","v","P","^","o",">"]

    colors = sns.color_palette("colorblind")
    
    plt.rc('xtick', labelsize=15) 
    plt.rc('ytick', labelsize=15)

    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(r'$||\nabla f||$', fontsize=18)

    for i,result in enumerate((results[0])):

        largest_common_ind = min([len(results[j][i]['time']) for j in range(rep)])
        
        func_mean = np.mean([results[j][i]['func_full'][:largest_common_ind] for j in range(rep)],axis=0)

        A, b, lam, mu, beta, _ = load_run(outputfile)
        n = A.shape[1]

        # calculate the mean gradient norm from the weights at each iteration
        grad_mean = np.mean(np.array([np.array([np.linalg.norm(grad_x_conv_reg(x, A, b, np.arange(n), lam, mu, beta)) for _,x in enumerate(results[j][i]['w_k'][:largest_common_ind])]) for j in range(len(results))]),axis=0)
        grad_std = np.std(np.array([np.array([np.linalg.norm(grad_x_conv_reg(x, A, b, np.arange(n), lam, mu, beta)) for _,x in enumerate(results[j][i]['w_k'][:largest_common_ind])]) for j in range(len(results))]),axis=0)

        
        # grad_mean = np.mean([results[j][i]['grad_est'][:largest_common_ind] for j in range(rep)],axis=0)
        # grad_std = np.std([results[j][i]['grad_est'][:largest_common_ind] for j in range(rep)],axis=0)

        if ind_ends is None:
            ind_end = largest_common_ind
        else:
            ind_end = min(ind_ends[i],largest_common_ind)

        if xparam == 'func_full':
            plt.semilogy(func_mean[:ind_end], 
                         label=labels[i],
                         color=colors[i],
                         marker = markers[i],
                         markevery=markevery,
                         linewidth=linewidth,
                         alpha=alpha,
                         markeredgewidth=markeredgewidth,
                         markeredgecolor=markeredgecolor
                        )
            plt.ylabel(r'$f(x_k)$')
            # plt.ylim([10e-9,1e2])  
        elif xparam == 'time':
            time_mean = np.mean([results[j][i]['time'][:largest_common_ind] for j in range(rep)],axis=0)
            
            plt.semilogy(time_mean[:ind_end], 
                         grad_mean[:ind_end], 
                         label=labels[i],
                         color=colors[i],
                         marker=markers[i],
                         markevery=markevery,
                         linewidth=linewidth,
                         alpha=alpha,
                         markeredgewidth=markeredgewidth,
                         markeredgecolor=markeredgecolor
                        )
            plt.ylim([10e-9,1e2])
        elif xparam == 'norm_s_k' or xparam == 'norm_s_k_squared':

            if subfigures == True: 
                eps = 10e-9
                plt.subplot(211)
                result[xparam] = np.array(result[xparam][:ind_end]) * np.array(result['accept'][:ind_end]) + eps 

                plt.semilogy(result[xparam][:ind_end],
                            label=labels[i],
                            color=colors[i],
                            marker=markers[i],
                            markevery=markevery,
                            linewidth=linewidth,
                            alpha=alpha,
                            markeredgewidth=markeredgewidth,
                            markeredgecolor=markeredgecolor
                            )
                if xparam == 'norm_s_k':
                    plt.ylabel(r'$||s_k||$', fontsize=18)
                elif xparam == 'norm_s_k_squared':
                    plt.ylabel(r'$||s_k||^2$', fontsize=18)
                plt.title(title, fontsize=18)
                plt.ylim([10e-9,1e2])

                plt.subplot(212)
                
                plt.ylabel(r'$||\nabla f||$', fontsize=18)
                plt.semilogy(grad_mean[:ind_end], 
                         label=labels[i],
                         color=colors[i],
                         marker = markers[i],
                         markevery=markevery,
                         linewidth=linewidth,
                         alpha=alpha,
                         markeredgewidth=markeredgewidth,
                         markeredgecolor=markeredgecolor
                        )
                plt.fill_between(np.arange(0,len(grad_mean))[:ind_end], (grad_mean-grad_std)[:ind_end], (grad_mean+grad_std)[:ind_end], 
                                color=colors[i], alpha=0.3)
                plt.xlabel(xlabel, fontsize=18)
                plt.ylim([10e-9,1e2])
                
            else:
                plt.semilogy(result[xparam][:ind_end],
                            label=labels[i],
                            color=colors[i],
                            marker=markers[i],
                            markevery=markevery,
                            linewidth=linewidth,
                            alpha=alpha,
                            markeredgewidth=markeredgewidth,
                            markeredgecolor=markeredgecolor
                            )
                plt.xlabel(xlabel, fontsize=18)
                if xparam == 'norm_s_k':
                    plt.ylabel(r'$||s_k||$', fontsize=18)
                elif xparam == 'norm_s_k_squared':
                    plt.ylabel(r'$||s_k||^2$', fontsize=18)
                plt.ylim([10e-9,1e2])

        elif xparam is not None:
            plt.semilogy(result[xparam][:ind_end], 
                         grad_mean[:ind_end], 
                         label=labels[i],
                         color=colors[i],
                         marker=markers[i],
                         markevery=markevery,
                         linewidth=linewidth,
                         alpha=alpha,
                         markeredgewidth=markeredgewidth,
                         markeredgecolor=markeredgecolor
                        )
            plt.fill_between(result[xparam][:ind_end], (grad_mean-grad_std)[:ind_end], (grad_mean+grad_std)[:ind_end], 
                             color=colors[i], alpha=0.5)  
            plt.ylim([10e-9,1e2])
        else: 
            plt.semilogy(grad_mean[:ind_end], 
                         label=labels[i],
                         color=colors[i],
                         marker = markers[i],
                         markevery=markevery,
                         linewidth=linewidth,
                         alpha=alpha,
                         markeredgewidth=markeredgewidth,
                         markeredgecolor=markeredgecolor
                        )
            plt.fill_between(np.arange(0,len(grad_mean)), (grad_mean-grad_std)[:ind_end], (grad_mean+grad_std)[:ind_end], 
                             color=colors[i], alpha=0.3)
            plt.ylim([10e-9,1e2])
    if xparam is None:

        plt.semilogy(np.arange(1,len(grad_mean))[:ind_end],(8*np.array(grad_mean)[0]*np.arange(1,len(grad_mean))**(-2/3))[:ind_end], 'b--', \
             label = r'$\mathcal{O}(k^{-2/3})$')
        plt.ylim([10e-9,1e2])
    
    
    
    plt.title(title, fontsize=18)
    plt.legend(fontsize=16,loc=(1,0))
    plt.tick_params(labelsize=14)
    
    if xparam is None:
        figure_name = 'plots/' + project_name + '_' + experiment_name +  '_convergence_in_iterations_' + figurename + '.pdf' 
    else:
        figure_name = 'plots/' + project_name + '_' + experiment_name + '_convergence_in_' + xparam + '_' + figurename +'.pdf'

    if save_fig == True:   
        plt.savefig(figure_name, bbox_inches='tight')

