from plotting_logreg import *
from utils import *
import yaml
from datetime import datetime

def generate_logreg_plots(conf_file, filename, ind_ends, context='poster', num_coord=True, iters=True, time=True, norm_sk=True, func_full=True, grad_est=True, hess_est=True):

    start_time = datetime.now()
    # conf_file = 'config_duke_SSCN'

    # load config file
    config = {}
    base_config_path =  'config_files/' + conf_file + '.yaml'
    with open(base_config_path, 'r') as file:
        config.update(yaml.safe_load(file))

    loss_func = config['loss_func']
    dataset = config['dataset']
    solver = config['subsolver']

    file_path = 'results_data/'
    # filename = 'logreg_nonconv_constant_schedule_duke_SSCN_exact_subsolver_n=7129_lam=0.100_constant_schedule.npy'
    
    # load numpy array
    outputfile = file_path + filename
    print(outputfile)
    A, b, lam, beta, results = load_run(outputfile)
    n = A.shape[1]


    assert(len(ind_ends) >= len(config['taus'])), 'ind_ends should have at least the same lengths as the number of schedules!'

    
    if config['optimizer'] == 'SSCN':
        method = 'SSCN_' + solver 
        if num_coord==True:
            plotting_wrapper(config, outputfile, results, 'num_coord', '# (Coordinates$^2$ + Coordinates)', figurename='%s_ds=%s_n=%d_lam=%.3f_%s_%s' % (loss_func, dataset, n, lam, method, context), save_fig=True, context=context, ind_ends=ind_ends)

    else:
        method = 'CD'
        if num_coord==True:
            plotting_wrapper(config, outputfile, results, 'num_coord', '# Coordinates', figurename='%s_ds=%s_n=%d_lam=%.3f_%s_%s' % (loss_func, dataset, n, lam, method, context), save_fig=True, context=context)

    print(hess_est)
    if iters==True:
        plotting_wrapper(config, outputfile, results, None, 'Iterations $k$', figurename='%s_ds=%s_n=%d_lam=%.3f_%s_%s' % (loss_func, dataset, n, lam, method, context), save_fig=True, context=context)

    if time==True:
        plotting_wrapper(config, outputfile, results, 'time', 'Time, s', figurename='%s_ds=%s_n=%d_lam=%.3f_%s_%s' % (loss_func, dataset, n, lam, method, context), save_fig=True, context=context, ind_ends=ind_ends)

    if norm_sk==True:
        plotting_wrapper(config, outputfile, results, 'norm_s_k', 'Iterations $k$', figurename='%s_ds=%s_n=%d_lam=%.3f_%s_%s' % (loss_func, dataset, n, lam, method, context), subfigures=False, save_fig=True, context=context)

    if func_full==True:
        plotting_wrapper(config, outputfile, results, 'func_full', 'Iterations $k$', figurename='%s_ds=%s_n=%d_lam=%.3f_%s_%s' % (loss_func, dataset, n, lam, method, context), subfigures=True, save_fig=True, context=context)

    if grad_est==True:
        plotting_wrapper(config, outputfile, results, 'grad_est', 'Iterations $k$', figurename='%s_ds=%s_n=%d_lam=%.3f_%s_%s' % (loss_func, dataset, n, lam, method, context), subfigures=True, save_fig=True, context=context)

    if hess_est==True:
        plotting_wrapper(config, outputfile, results, 'hess_F_est', 'Iterations $k$', figurename='%s_ds=%s_n=%d_lam=%.3f_%s' % (loss_func, dataset, n, lam, method), subfigures=True, save_fig=True, save_grads=False, context=context)

    print('Total time: %.3f' % (datetime.now()-start_time).total_seconds())



if __name__ == '__main__':

    conf_file = 'config_madelon_SSCN'
    filename = 'non_linear_square_loss_nonconvex_constant_schedule_madelon_SSCN_exact_subsolver_n=500_lam=0.100_constant_schedule.npy'

    context = 'poster' 
    # ind_ends = [5000,5000,5000,350,400]
    # ind_ends = [5000,5000,5000,500,500,500,350,100,100,100]
    ind_ends = [1505,1200,5000,9000,9000,400,500,9000,400,500]
    # ind_ends = [50,50]

    #         num_coord, iters, time,  norm_sk, func_full, grad_est,  hess_est
    plots_fig = [True,   True, True, True,  True ,    True,     True]



    generate_logreg_plots(conf_file, filename, ind_ends, context=context, 
                            num_coord=plots_fig[0], 
                            iters=plots_fig[1], 
                            time=plots_fig[2], 
                            norm_sk=plots_fig[3], 
                            func_full=plots_fig[4], 
                            grad_est=plots_fig[5],
                            hess_est=plots_fig[6])

