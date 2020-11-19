import numpy as np

CONFIG_SUPERNET = {
    'gpu_settings' : {
        'gpu_ids' : [3]
    },
    'lookup_table' : {
        'create_from_scratch' : False,
        'path_to_lookup_table' : './supernet_functions/lookup_table2.txt',
        'number_of_runs' : 50 # each operation run number_of_runs times and then we will take average
    },
    'logging' : {
        'path_to_log_file' : './supernet_functions/logs/logger/',
        'path_to_tensorboard_logs' : './supernet_functions/logs/tb'
    },
    'dataloading' : {
        'batch_size' : 32, #1000
        'w_share_in_train' : 0.80,
        'path_to_save_data' : './cifar10_data'
    },
    'optimizer' : {
        # SGD parameters for w
        'w_lr' : 0.01, #0.1, 0.007
        'w_momentum' : 0.9,
        'w_weight_decay' : 1e-4,
        # Adam parameters for thetas
        'thetas_lr' : 0.001, #0.01
        'thetas_weight_decay' : 5 * 1e-4
    },
    'loss' : {
        'alpha' : 3.0,
        'beta' : 1.0
    },
    'train_settings' : {
        'cnt_epochs' : 180, # 90
        'train_thetas_from_the_epoch' : 10,
        'print_freq' : 50,
        'path_to_save_model' : './supernet_functions/logs/1116-test/1116test13/best_model.pth',
        # for Gumbel Softmax
        'init_temperature' : 5.0,
        'exp_anneal_rate' : np.exp(-0.045)
    }
}
