from comet_ml import Experiment
import numpy as np
import torch
from torch import nn
from tensorboardX import SummaryWriter
from scipy.special import softmax
import argparse

from general_functions.dataloaders import get_loaders, get_test_loader
from general_functions.utils import get_logger, weights_init, load, create_directories_from_list, \
                                    check_tensor_in_list, writh_new_ARCH_to_fbnet_modeldef
from supernet_functions.lookup_table_builder import LookUpTable
from supernet_functions.model_supernet import FBNet_Stochastic_SuperNet, SupernetLoss
from supernet_functions.training_functions_supernet import TrainerSupernet
from supernet_functions.config_for_supernet import CONFIG_SUPERNET
from fbnet_building_blocks.fbnet_modeldef import MODEL_ARCH

#----以下全て, 再現性関連
import random                                                                                                                                 
'''
# cuDNNを使用しない
seed = 32
torch.backends.cudnn.deterministic = True  
random.seed(seed)  
np.random.seed(seed)  
torch.manual_seed(seed)  
# cuda でのRNGを初期化  
torch.cuda.manual_seed(seed) 
'''

experiment = Experiment(api_key="8ZVbUGyXPGq2wGdJg0kmFAXPu",
                    project_name="1120test-bacchus", workspace="oza15015")


parser = argparse.ArgumentParser("action")
parser.add_argument('--train_or_sample', type=str, default='', \
                    help='train means training of the SuperNet, sample means sample from SuperNet\'s results')
parser.add_argument('--architecture_name', type=str, default='', \
                    help='Name of an architecture to be sampled')
parser.add_argument('--hardsampling_bool_value', type=str, default='True', \
                    help='If not False or 0 -> do hardsampling, else - softmax sampling')
args = parser.parse_args()
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def train_supernet():
    manual_seed = 1
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.backends.cudnn.benchmark = True

    create_directories_from_list([CONFIG_SUPERNET['logging']['path_to_tensorboard_logs']])
    
    logger = get_logger(CONFIG_SUPERNET['logging']['path_to_log_file'])
    writer = SummaryWriter(log_dir=CONFIG_SUPERNET['logging']['path_to_tensorboard_logs'])
    
    #### LookUp table consists all information about layers
    lookup_table = LookUpTable(calulate_latency=CONFIG_SUPERNET['lookup_table']['create_from_scratch'])
    
    #### DataLoading
    train_w_loader, train_thetas_loader = get_loaders(CONFIG_SUPERNET['dataloading']['w_share_in_train'],
                                                      CONFIG_SUPERNET['dataloading']['batch_size'],
                                                      CONFIG_SUPERNET['dataloading']['path_to_save_data'],
                                                      logger)
    test_loader = get_test_loader(CONFIG_SUPERNET['dataloading']['batch_size'],
                                  CONFIG_SUPERNET['dataloading']['path_to_save_data'])
    
    #### Model
    model = FBNet_Stochastic_SuperNet(lookup_table, cnt_classes=10).to(device) #.cuda()
    model = model.apply(weights_init)
    model = nn.DataParallel(model, device_ids=[2])
   
    
    #### Loss, Optimizer and Scheduler
    criterion = SupernetLoss().to(device) #.cuda()

    thetas_params = [param for name, param in model.named_parameters() if 'thetas' in name]
    params_except_thetas = [param for param in model.parameters() if not check_tensor_in_list(param, thetas_params)]

    w_optimizer = torch.optim.SGD(params=params_except_thetas,
                                  lr=CONFIG_SUPERNET['optimizer']['w_lr'], 
                                  momentum=CONFIG_SUPERNET['optimizer']['w_momentum'],
                                  weight_decay=CONFIG_SUPERNET['optimizer']['w_weight_decay'])
    
    theta_optimizer = torch.optim.Adam(params=thetas_params,
                                       lr=CONFIG_SUPERNET['optimizer']['thetas_lr'],
                                       weight_decay=CONFIG_SUPERNET['optimizer']['thetas_weight_decay'])

    last_epoch = -1
    w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(w_optimizer,
                                                             T_max=CONFIG_SUPERNET['train_settings']['cnt_epochs'],
                                                             last_epoch=last_epoch)
    
    hyper_params = {
    'batch_size': CONFIG_SUPERNET['dataloading']['batch_size'],
    'w_share_in_train': CONFIG_SUPERNET['dataloading']['w_share_in_train'],
    'path_to_save_data': CONFIG_SUPERNET['dataloading']['path_to_save_data'],
    'w_lr': CONFIG_SUPERNET['optimizer']['w_lr'],
    'w_momentum': CONFIG_SUPERNET['optimizer']['w_momentum'],
    'w_weight_decay': CONFIG_SUPERNET['optimizer']['w_weight_decay'],
    'thetas_lr': CONFIG_SUPERNET['optimizer']['thetas_lr'],
    'thetas_weight_decay': CONFIG_SUPERNET['optimizer']['thetas_weight_decay'],
    'epoch': CONFIG_SUPERNET['train_settings']['cnt_epochs']}
    
    experiment.log_parameters(hyper_params)

    #### Training Loop
    trainer = TrainerSupernet(criterion, w_optimizer, theta_optimizer, w_scheduler, logger, writer, experiment)
    trainer.train_loop(train_w_loader, train_thetas_loader, test_loader, model)

# Arguments:
# hardsampling=True means get operations with the largest weights
#             =False means apply softmax to weights and sample from the distribution
# unique_name_of_arch - name of architecture. will be written into fbnet_building_blocks/fbnet_modeldef.py
#                       and can be used in the training by train_architecture_main_file.py
def sample_architecture_from_the_supernet(unique_name_of_arch, hardsampling=True):
    logger = get_logger(CONFIG_SUPERNET['logging']['path_to_log_file']) 
    lookup_table = LookUpTable()
    model = FBNet_Stochastic_SuperNet(lookup_table, cnt_classes=10).to(device) #.cuda()
    model = nn.DataParallel(model)

    load(model, CONFIG_SUPERNET['train_settings']['path_to_save_model'])

    ops_names = [op_name for op_name in lookup_table.lookup_table_operations]
    cnt_ops = len(ops_names)

    arch_operations=[]
    if hardsampling:
        for layer in model.module.stages_to_search:
            #print(layer)
            #quit()
            arch_operations.append(ops_names[np.argmax(layer.thetas.detach().cpu().numpy())])
    else:
        rng = np.linspace(0, cnt_ops - 1, cnt_ops, dtype=int)
        for layer in model.module.stages_to_search:
            distribution = softmax(layer.thetas.detach().cpu().numpy())
            arch_operations.append(ops_names[np.random.choice(rng, p=distribution)])
    
    logger.info("Sampled Architecture: " + " - ".join(arch_operations))
    writh_new_ARCH_to_fbnet_modeldef(arch_operations, my_unique_name_for_ARCH=unique_name_of_arch)
    logger.info("CONGRATULATIONS! New architecture " + unique_name_of_arch \
                + " was written into fbnet_building_blocks/fbnet_modeldef.py")
    
if __name__ == "__main__":
    assert args.train_or_sample in ['train', 'sample']
    if args.train_or_sample == 'train':
        train_supernet()
    elif args.train_or_sample == 'sample':
        print(args.architecture_name)
        assert args.architecture_name != '' and args.architecture_name not in MODEL_ARCH
        hardsampling = False if args.hardsampling_bool_value in ['False', '0'] else True
        sample_architecture_from_the_supernet(unique_name_of_arch=args.architecture_name, hardsampling=hardsampling)
