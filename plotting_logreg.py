from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from utils import *
from run_logreg_experiments import *

def plotting_wrapper(config, outputfile, results, xparam, xlabel, figurename, subfigures=False, save_fig=False, save_grads=True, ind_ends=None, context='notebook'):
    # ind_ends is a list of final indices for each schedule. It has to match len(results[0])
    # if the index specified by ind_ends is larger than largest_common_ind, then largest_common_ind will be used instead.

    project_name = config['project_name']
    experiment_name = config['experiment_name']

    rep = config['repetitions']
    # lams = config['lambdas']
    log_every = config['log_every']
    
    optimizer = config['optimizer']
    if optimizer == 'SSCN':
        solver = config['subsolver']
        optimizer = 'SSCN ' + solver
        thresh_sk = True
    else:
        thresh_sk = False

    loss_func = config['loss_func']

    dataset = config['dataset']
    if dataset == 'synthetic':
        dataset = 'synthetic ' + config['correlated_data'] + ' correl.'

    coord_schedule = config['coordinate_schedule']

    A, _, lam, _, _ = load_run(outputfile)
    n = A.shape[1]

    if coord_schedule == 'constant':
        taus = config['taus']

        labels = [(r'$\tau = %d$' % (tau)) for tau in taus ]
        title = r'%s, %s, $n=%d, \lambda= %.1f$' %(optimizer, dataset, n, lam)
    elif coord_schedule == 'exponential':
        cs = config['cs']
        exps = config['exps']
        taus = config['taus']
        tau = taus[0]

        labels = [(r'$c_e = %.2f, d = %.3f$' % (c, exp)) for c in cs for exp in exps ]
        title = r'%s, %s, $|S|=\tau + c_e \exp(d \cdot k), n=%d, \tau = %d, \lambda= %.1f$' %(optimizer, dataset, n, tau, lam)
      
    elif coord_schedule == 'adaptive':

        cs = config['cs']
        betas = config['beta']
        labels = [(r'adaptive, $c = %.4f, \beta = %.1f$' % (c, beta) ) for c in cs for beta in betas ]
        title = r'%s, %s, n=%d, $\lambda= %.1f$' %(optimizer, dataset, n, lam) 
 

    plot_every = config['plot_every']

    plot_schedules = config['plot_schedules']
    
    
    plot_results(project_name, experiment_name, outputfile, rep, results, xparam, labels, log_every, plot_every, plot_schedules,
                title, xlabel, figurename, subfigures=subfigures, save_fig=save_fig, save_grads=save_grads, ind_ends=ind_ends, thresh_sk=thresh_sk, context=context)
    

def plot_results(project_name, experiment_name, outputfile, rep, results, xparam, labels, log_every, plot_every, plot_schedules,
                title, xlabel, figurename, subfigures=False, save_fig=False, save_grads=True, ind_ends=None, thresh_sk=False, context='notebook'):

    sns.set_style("white")

    sns.set_context(context)
    sns.set_style("ticks")
    # sns.despine()

    # print(sns.axes_style())

    # plt.figure(figsize=(5.5*7.7, 3*5.7))
    plt.figure(figsize=(7.7, 5.7))
    linewidth = 3.5
    alpha = 0.8
    markeredgewidth=1.5
    markeredgecolor=[0,0,0,0.6]
    
    markers = ["<","s","p","D","X","v","P","^","o",">"]

    colors = sns.color_palette("colorblind")
    
    plt.rc('xtick', labelsize=18) 
    plt.rc('ytick', labelsize=18)

    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(r'$||\nabla f(x_k)||$', fontsize=18)

    
    # plt.tick_params(labelsize=14)
    markers_ind = 0

    for i,result in enumerate((results[0])):
        if plot_schedules[i] == 0:
            continue
        
        
        # print('i: %d' %i)
        largest_common_ind = min([len(results[j][i]['time']) for j in range(rep)])
        
        func_mean = np.mean([results[j][i]['func_full'][:largest_common_ind] for j in range(rep)],axis=0)

        A, b, lam, _, _ = load_run(outputfile)
        n = A.shape[1]

        filename = outputfile[:-4] + '_grad_means_%d.npy' %i 
        try:
            with open(filename, 'rb') as f:
                grad_mean = np.load(f)
                grad_std = np.load(f)
                iter_ticks = np.load(f)
        except OSError as e:
        
            print('No existing gradient information exist...')
            print('Calculating gradient mean and std from iterates...')
            # calculate the mean gradient norm from the weights at each iteration
            
            grad_mean = 0
            grad_std = 0
            for j in range(len(results)):
                grad_norm = []
                for _,x in enumerate(results[j][i]['w_k'][:largest_common_ind]):
                    grad_norm.append(np.linalg.norm(grad_x(x, A, b, np.arange(n), lam)))
                grad_mean += np.array(grad_norm)
                # grad_mean += np.array([np.linalg.norm(grad_x_conv_reg(x, A, b, np.arange(n), lam, beta)) for _,x in enumerate(results[j][i]['w_k'][:largest_common_ind])])
            grad_mean /= len(results)
            iter_ticks = np.arange(0,len(grad_mean)*log_every[i]*plot_every[i],log_every[i]*plot_every[i])

            for j in range(len(results)):
                grad_norm = []
                for _,x in enumerate(results[j][i]['w_k'][:largest_common_ind]):
                    grad_norm.append(np.linalg.norm(grad_x(x, A, b, np.arange(n), lam)))
                grad_std += (np.array(grad_norm) - grad_mean)**2
            grad_std = 1/np.sqrt(len(results)) * np.sqrt(grad_std)

            if save_grads == True:
                with open(filename, 'wb') as f:
                    np.save(f, grad_mean)
                    np.save(f, grad_std)
                    np.save(f, iter_ticks)

        # for j in range(len(results)):
        #     grad_std += (np.array([np.linalg.norm(grad_x_conv_reg(x, A, b, np.arange(n), lam, beta)) for _,x in enumerate(results[j][i]['w_k'][:largest_common_ind])]) - grad_mean)**2
        # grad_std = 1/np.sqrt(len(results)) * np.sqrt(grad_std)
    

        # grad_mean = np.mean(np.array([np.array([np.linalg.norm(grad_x_conv_reg(x, A, b, np.arange(n), lam, beta)) for _,x in enumerate(results[j][i]['w_k'][:largest_common_ind])]) for j in range(len(results))]),axis=0)
        # grad_std = np.std(np.array([np.array([np.linalg.norm(grad_x_conv_reg(x, A, b, np.arange(n), lam, beta)) for _,x in enumerate(results[j][i]['w_k'][:largest_common_ind])]) for j in range(len(results))]),axis=0)

        

        
        print('Plotting figures...')
        
        if ind_ends is None:
            ind_end = largest_common_ind
        else:
            ind_end = min(ind_ends[i],largest_common_ind)
            print('i: %d, ind_end: %d ' % (i, ind_end))

        markevery= int(len(results[0][i]['time'][:ind_end:plot_every[i]])/6)+1

        if xparam == 'grad_est':
            grad_est_mean = np.mean([results[j][i]['grad_est'][:largest_common_ind] for j in range(rep)],axis=0)
            # grad_est_std = np.std([results[j][i]['grad_est'][:largest_common_ind] for j in range(rep)],axis=0)

            plt.semilogy(iter_ticks[:ind_end:plot_every[i]], grad_est_mean[:ind_end:plot_every[i]], 
                         label=labels[i],
                         color=colors[markers_ind],
                         marker = markers[markers_ind],
                         markevery=markevery,
                         linewidth=linewidth,
                         alpha=alpha,
                         markeredgewidth=markeredgewidth,
                         markeredgecolor=markeredgecolor
                        )
            plt.ylabel(r'$\nabla f(x_k)_{est}$')
            # plt.ylim([10e-9,1e2])  
        elif xparam == 'func_full':
            plt.semilogy(iter_ticks[:ind_end:plot_every[i]], func_mean[:ind_end:plot_every[i]], 
                         label=labels[i],
                         color=colors[markers_ind],
                         marker = markers[markers_ind],
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
            
            plt.semilogy(time_mean[:ind_end:plot_every[i]], 
                         grad_mean[:ind_end:plot_every[i]], 
                         label=labels[i],
                         color=colors[markers_ind],
                         marker=markers[markers_ind],
                         markevery=markevery,
                         linewidth=linewidth,
                         alpha=alpha,
                         markeredgewidth=markeredgewidth,
                         markeredgecolor=markeredgecolor
                        )
            
            plt.fill_between(time_mean[:ind_end:plot_every[i]], (grad_mean-grad_std)[:ind_end:plot_every[i]], (grad_mean+grad_std)[:ind_end:plot_every[i]], 
                             color=colors[markers_ind], alpha=0.5)  
            # plt.ylim([1e-6,1e2])
        elif xparam == 'norm_s_k':

            if subfigures == True: 
                eps = 10e-9
                plt.subplot(211)

                # if thresh_sk == True:
                #     result[xparam] = np.array(result[xparam][:ind_end:plot_every[i]]) * np.array(result['accept'][:ind_end:plot_every[i]]) + eps 

                plt.semilogy(iter_ticks[:ind_end:plot_every[i]], np.array(result[xparam][:ind_end:plot_every[i]])**2,
                            label=labels[i],
                            color=colors[markers_ind],
                            marker=markers[markers_ind],
                            markevery=markevery,
                            linewidth=linewidth,
                            alpha=alpha,
                            markeredgewidth=markeredgewidth,
                            markeredgecolor=markeredgecolor
                            )
                plt.ylabel(r'$||h_k||^2$', fontsize=18)
                
                plt.title(title, fontsize=18)
                # plt.ylim([10e-8,1e2])

                plt.subplot(212)
                
                plt.ylabel(r'$||\nabla f(x_k)||$', fontsize=18)
                plt.semilogy(iter_ticks[:ind_end:plot_every[i]], grad_mean[:ind_end:plot_every[i]], 
                         label=labels[i],
                         color=colors[markers_ind],
                         marker = markers[markers_ind],
                         markevery=markevery,
                         linewidth=linewidth,
                         alpha=alpha,
                         markeredgewidth=markeredgewidth,
                         markeredgecolor=markeredgecolor
                        )
                plt.fill_between(iter_ticks[:ind_end:plot_every[i]], (grad_mean-grad_std)[:ind_end:plot_every[i]], (grad_mean+grad_std)[:ind_end:plot_every[i]], 
                                color=colors[markers_ind], alpha=0.3)
                plt.xlabel(xlabel, fontsize=18)
                # plt.ylim([10e-8,1e2])

            else:
                plt.semilogy(iter_ticks[:ind_end:plot_every[i]], np.array(result[xparam][:ind_end:plot_every[i]])**2,
                            label=labels[i],
                            color=colors[markers_ind],
                            marker=markers[markers_ind],
                            markevery=markevery,
                            linewidth=linewidth,
                            alpha=alpha,
                            markeredgewidth=markeredgewidth,
                            markeredgecolor=markeredgecolor
                            )
                plt.ylabel(r'$||h_k||^2$', fontsize=18)
                
                plt.title(title, fontsize=18)
                plt.ylim([10e-14,1e0])
                # plt.xlim([-1300, 34000])

            
        elif xparam == 'tau':
            if subfigures == True: 
                plt.subplot(211)

                # if thresh_sk == True:
                #     result[xparam] = np.array(result[xparam][:ind_end:plot_every[i]]) * np.array(result['accept'][:ind_end:plot_every[i]]) + eps 

                plt.semilogy(iter_ticks[:ind_end:plot_every[i]], result[xparam][:ind_end:plot_every[i]],
                            label=labels[i],
                            color=colors[markers_ind],
                            marker=markers[markers_ind],
                            markevery=markevery,
                            linewidth=linewidth,
                            alpha=alpha,
                            markeredgewidth=markeredgewidth,
                            markeredgecolor=markeredgecolor
                            )
                plt.ylabel(r'$\tau$', fontsize=18)
                
                plt.title(title, fontsize=18)
                # plt.ylim([10e-8,1e2])

                plt.subplot(212)
                
                plt.ylabel(r'$||h_k||$', fontsize=18)
                plt.semilogy(iter_ticks[:ind_end:plot_every[i]], result['norm_s_k'][:ind_end:plot_every[i]], 
                        label=labels[i],
                        color=colors[markers_ind],
                        marker = markers[markers_ind],
                        markevery=markevery,
                        linewidth=linewidth,
                        alpha=alpha,
                        markeredgewidth=markeredgewidth,
                        markeredgecolor=markeredgecolor
                        )
                plt.xlabel(xlabel, fontsize=18)
                # plt.ylim([10e-8,1e2])

                
            else:
                plt.semilogy(iter_ticks[:ind_end:plot_every[i]], result[xparam][:ind_end:plot_every[i]],
                            label=labels[i],
                            color=colors[markers_ind],
                            marker=markers[markers_ind],
                            markevery=markevery,
                            linewidth=linewidth,
                            alpha=alpha,
                            markeredgewidth=markeredgewidth,
                            markeredgecolor=markeredgecolor
                            )
                plt.ylabel(r'$\tau$', fontsize=18)
                
                plt.title(title, fontsize=18)
                # plt.ylim([10e-8,1e2])
                
        elif xparam == 'hess_F_est':
            if subfigures == True: 

                plt.subplot(331)
                
                # plot the norm of step size
                plt.semilogy(iter_ticks[:ind_end:plot_every[i]], result['norm_s_k'][:ind_end:plot_every[i]], 
                        label=labels[i],
                        color=colors[markers_ind],
                        marker = markers[markers_ind],
                        markevery=markevery,
                        linewidth=linewidth,
                        alpha=alpha,
                        markeredgewidth=markeredgewidth,
                        markeredgecolor=markeredgecolor
                        )
                plt.xlabel(xlabel, fontsize=18)
                plt.ylabel(r'$||h_k||$', fontsize=18)

                # plt.ylim([10e-8,1e2])

                plt.subplot(332)

                # plot the (full) gradient norm 
                plt.semilogy(iter_ticks[:ind_end:plot_every[i]], grad_mean[:ind_end:plot_every[i]],
                            label=labels[i],
                            color=colors[markers_ind],
                            marker=markers[markers_ind],
                            markevery=markevery,
                            linewidth=linewidth,
                            alpha=alpha,
                            markeredgewidth=markeredgewidth,
                            markeredgecolor=markeredgecolor
                            )
                plt.ylabel(r'$\|| \nabla f(x_k)\||$', fontsize=18)
                plt.xlabel(xlabel, fontsize=18)

                plt.subplot(333)

                # plot the etimated gradient norm 
                plt.semilogy(iter_ticks[:ind_end:plot_every[i]], result['grad_est'][:ind_end:plot_every[i]],
                            label=labels[i],
                            color=colors[markers_ind],
                            marker=markers[markers_ind],
                            markevery=markevery,
                            linewidth=linewidth,
                            alpha=alpha,
                            markeredgewidth=markeredgewidth,
                            markeredgecolor=markeredgecolor
                            )
                plt.ylabel(r'$\|| \nabla f(x_k)_{est}\||$', fontsize=18)
                plt.xlabel(xlabel, fontsize=18)

                plt.subplot(334)

                # if thresh_sk == True:
                #     result[xparam] = np.array(result[xparam][:ind_end:plot_every[i]]) * np.array(result['accept'][:ind_end:plot_every[i]]) + eps 

                # plot the 2-norm of the estimated Hessian
                plt.semilogy(iter_ticks[:ind_end:plot_every[i]], result[xparam][:ind_end:plot_every[i]],
                            label=labels[i],
                            color=colors[markers_ind],
                            marker=markers[markers_ind],
                            markevery=markevery,
                            linewidth=linewidth,
                            alpha=alpha,
                            markeredgewidth=markeredgewidth,
                            markeredgecolor=markeredgecolor
                            )
                plt.ylabel(r'$||\nabla^2 f(x_k)_{est}||_2$', fontsize=18)
                # plt.ylabel(r'$||\nabla^2 f(x_k)||_2$', fontsize=18)
                plt.xlabel(xlabel, fontsize=18)
                
                plt.title(title, fontsize=18)
                # plt.ylim([10e-8,1e2])

                plt.subplot(335)

                plt.plot(iter_ticks[:ind_end:plot_every[i]], result['adapt_sc_term1'][:ind_end:plot_every[i]],
                            label=labels[i],
                            color=colors[markers_ind],
                            marker=markers[markers_ind],
                            markevery=markevery,
                            linewidth=linewidth,
                            alpha=alpha,
                            markeredgewidth=markeredgewidth,
                            markeredgecolor=markeredgecolor
                            )
                plt.ylabel(r'$( 1 - \frac{\epsilon_1^2}{||\nabla f(x_k)_{[S_k]}||^2} )$', fontsize=18)

                plt.xlabel(xlabel, fontsize=18)

                plt.subplot(336)

                plt.plot(iter_ticks[:ind_end:plot_every[i]], result['adapt_sc_term2'][:ind_end:plot_every[i]],
                            label=labels[i],
                            color=colors[markers_ind],
                            marker=markers[markers_ind],
                            markevery=markevery,
                            linewidth=linewidth,
                            alpha=alpha,
                            markeredgewidth=markeredgewidth,
                            markeredgecolor=markeredgecolor
                            )
                plt.ylabel(r'$\sqrt{1 - \frac{\epsilon_2}{||\nabla^2 f(x_k)_{[S_k]}||^2}}$', fontsize=18)

                plt.xlabel(xlabel, fontsize=18)


                plt.subplot(337)

                plt.plot(iter_ticks[:ind_end:plot_every[i]], result['tau'][:ind_end:plot_every[i]],
                            label=labels[i],
                            color=colors[markers_ind],
                            marker=markers[markers_ind],
                            markevery=markevery,
                            linewidth=linewidth,
                            alpha=alpha,
                            markeredgewidth=markeredgewidth,
                            markeredgecolor=markeredgecolor
                            )
                plt.ylabel(r'$\tau$', fontsize=18)

                plt.xlabel(xlabel, fontsize=18)

                plt.subplot(338)

                plt.plot(iter_ticks[:ind_end:plot_every[i]], (np.array(result['norm_s_k'][:ind_end:plot_every[i]])
                                                              /np.array(result['hess_F_est'][:ind_end:plot_every[i]]))**2,
                            label=labels[i],
                            color=colors[markers_ind],
                            marker=markers[markers_ind],
                            markevery=markevery,
                            linewidth=linewidth,
                            alpha=alpha,
                            markeredgewidth=markeredgewidth,
                            markeredgecolor=markeredgecolor
                            )
                # plt.ylabel(r'$||h_k||^2/|| \nabla^2 f(x_k)_{[S_k]} ||^2$', fontsize=18)
                plt.ylabel(r'$||h_k||^2/|| \nabla^2 f(x_k)_{[S_k]} ||^2$', fontsize=18)

                plt.xlabel(xlabel, fontsize=18)

                plt.subplot(339)

                plt.plot(iter_ticks[:ind_end:plot_every[i]], (np.array(result['norm_s_k'][:ind_end:plot_every[i]])**2
                                                              /grad_mean[:ind_end:plot_every[i]])**2,
                            label=labels[i],
                            color=colors[markers_ind],
                            marker=markers[markers_ind],
                            markevery=markevery,
                            linewidth=linewidth,
                            alpha=alpha,
                            markeredgewidth=markeredgewidth,
                            markeredgecolor=markeredgecolor
                            )
                plt.ylabel(r'$||h_k||^4/|| \nabla f(x_k)_{[S_k]} ||^2$', fontsize=18)

                plt.xlabel(xlabel, fontsize=18)



                
            else:
                plt.semilogy(iter_ticks[:ind_end:plot_every[i]], result[xparam][:ind_end:plot_every[i]],
                            label=labels[i],
                            color=colors[markers_ind],
                            marker=markers[markers_ind],
                            markevery=markevery,
                            linewidth=linewidth,
                            alpha=alpha,
                            markeredgewidth=markeredgewidth,
                            markeredgecolor=markeredgecolor
                            )
                plt.ylabel(r'$\tau$', fontsize=18)
                
                plt.title(title, fontsize=18)
                # plt.ylim([10e-8,1e2])

        elif xparam is not None:
            plt.semilogy(result[xparam][:ind_end:plot_every[i]], 
                         grad_mean[:ind_end:plot_every[i]], 
                         label=labels[i],
                         color=colors[markers_ind],
                         marker=markers[markers_ind],
                         markevery=markevery,
                         linewidth=linewidth,
                         alpha=alpha,
                         markeredgewidth=markeredgewidth,
                         markeredgecolor=markeredgecolor
                        )
            plt.fill_between(result[xparam][:ind_end:plot_every[i]], (grad_mean-grad_std)[:ind_end:plot_every[i]], (grad_mean+grad_std)[:ind_end:plot_every[i]], 
                             color=colors[markers_ind], alpha=0.5)  
            # plt.ylim([10e-7,1e2])
            
        else: 
            
            plt.semilogy(iter_ticks[:ind_end:plot_every[i]], grad_mean[:ind_end:plot_every[i]], 
                         label=labels[i],
                         color=colors[markers_ind],
                         marker = markers[markers_ind],
                         markevery=markevery,
                         linewidth=linewidth,
                         alpha=alpha,
                         markeredgewidth=markeredgewidth,
                         markeredgecolor=markeredgecolor
                        )
            plt.fill_between(iter_ticks[:ind_end:plot_every[i]], (grad_mean-grad_std)[:ind_end:plot_every[i]], (grad_mean+grad_std)[:ind_end:plot_every[i]], 
                             color=colors[markers_ind], alpha=0.3)
            # plt.ylim([10e-7,1e2])
            # plt.xlim([-120,2400])

        markers_ind += 1


    if xparam is None:
        # if markers_ind == 0:
        print(iter_ticks)
        asymptotic_bound_end_tick = iter_ticks[-2] 
        # print(iter_ticks[:ind_end:plot_every[i]])
        plt.semilogy(np.arange(0,asymptotic_bound_end_tick),(np.array(grad_mean)[0] *np.arange(0,asymptotic_bound_end_tick)**(-2/3)), 'b--', \
            label = r'$\mathcal{O}(k^{-2/3})$')
        # plt.ylim([10e-9,1e2])

    plt.title(title, fontsize=18)
    plt.legend(fontsize=15, loc='upper right')
    # plt.legend()

    if xparam is None:
        figure_name = 'plots/' + experiment_name +  '_convergence_in_iterations_' + figurename + '.pdf' 
    else:
        figure_name = 'plots/' + experiment_name + '_convergence_in_' + xparam + '_' + figurename +'.pdf'

    if save_fig == True:   
        plt.savefig(figure_name, bbox_inches='tight')

