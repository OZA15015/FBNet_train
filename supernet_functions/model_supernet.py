from comet_ml import Experiment
import torch
from torch import nn
from collections import OrderedDict
from fbnet_building_blocks.fbnet_builder import ConvBNRelu, Flatten
from supernet_functions.config_for_supernet import CONFIG_SUPERNET
import numpy as np
from general_functions.utils import accuracy
#----以下全て, 再現性関連
import random
import time
# cuDNNを使用しない
seed = 32
torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# cuda でのRNGを初期化
torch.cuda.manual_seed(seed)

#commen.ml部
#experiment = Experiment(api_key="8ZVbUGyXPGq2wGdJg0kmFAXPu",
#                    project_name="1026test-fbnet-search1", workspace="oza15015")

class MixedOperation(nn.Module):
    
    # Arguments:
    # proposed_operations is a dictionary {operation_name : op_constructor}
    # latency is a dictionary {operation_name : latency}
    def __init__(self, layer_parameters, proposed_operations, latency, energy): #energy add
        super(MixedOperation, self).__init__()
        ops_names = [op_name for op_name in proposed_operations]
        
        self.ops = nn.ModuleList([proposed_operations[op_name](*layer_parameters)
                                  for op_name in ops_names])
        self.latency = [latency[op_name] for op_name in ops_names]
        self.energy = [energy[op_name] for op_name in ops_names] #energt add
        self.thetas = nn.Parameter(torch.Tensor([1.0 / len(ops_names) for i in range(len(ops_names))]))
    
    def forward(self, x, temperature, latency_to_accumulate, energy_to_accumulate): #energy add
        soft_mask_variables = nn.functional.gumbel_softmax(self.thetas, temperature)
        output  = sum(m * op(x) for m, op in zip(soft_mask_variables, self.ops))

        latency = sum(m * lat for m, lat in zip(soft_mask_variables, self.latency))
        energy = sum(m * lat for m, lat in zip(soft_mask_variables, self.energy)) #energy add
        #for m, lat in zip(soft_mask_variables, self.latency):
        #    print(lat)
        #print("=====================")
        #quit()
        latency_to_accumulate = latency_to_accumulate + latency
        energy_to_accumulate = energy_to_accumulate + energy # energy add
        return output, latency_to_accumulate, energy_to_accumulate #energy add

class FBNet_Stochastic_SuperNet(nn.Module):
    def __init__(self, lookup_table, cnt_classes=10):
        super(FBNet_Stochastic_SuperNet, self).__init__()
        # self.first identical to 'add_first' in the fbnet_building_blocks/fbnet_builder.py

        
        self.first = ConvBNRelu(input_depth=3, output_depth=16, kernel=3, stride=1,
                                pad = 3 // 2, no_bias=1, use_relu="relu", bn_type="bn")
        '''
        self.first = ConvBNRelu(input_depth=3, output_depth=16, kernel=3, stride=1,
                                pad = True, no_bias=1, use_relu="relu", bn_type="bn")
        '''
        # stride = 1へ変更
        self.stages_to_search = nn.ModuleList([MixedOperation(
                                                   lookup_table.layers_parameters[layer_id],
                                                   lookup_table.lookup_table_operations,
                                                   lookup_table.lookup_table_latency[layer_id],
                                                   lookup_table.lookup_table_energy[layer_id]) #energy add
                                               for layer_id in range(lookup_table.cnt_layers)])

        '''
        self.last_stages = nn.Sequential(OrderedDict([
            ("conv_k1", nn.Conv2d(lookup_table.layers_parameters[-1][1], 1504, kernel_size = 1)),
            ("avg_pool_k7", nn.AvgPool2d(kernel_size=7)),
            ("flatten", Flatten()),
            ("fc", nn.Linear(in_features=1504, out_features=cnt_classes)),
        ]))
        '''

        self.last_stages = nn.Sequential(OrderedDict([
            ("conv_k1", nn.Conv2d(lookup_table.layers_parameters[-1][1], 320, kernel_size = 1)),
            ("dropout", nn.Dropout(0.2)),
            ("flatten", Flatten()),
            ("fc", nn.Linear(in_features=1280, out_features=cnt_classes)),
        ]))

    
    def forward(self, x, temperature, latency_to_accumulate, energy_to_accumulate): #energy_to_accumulate add        
        #time_sum = 0
        #start_time = time.time()
        y = self.first(x) #ここで推論時間計算, ここに追加すべき, temperatureって何か確認
        for mixed_op in self.stages_to_search:
            y, latency_to_accumulate, energy_to_accumulate = mixed_op(y, temperature, latency_to_accumulate, energy_to_accumulate) #energy_to_accumulate add
        y = self.last_stages(y)
        return y, latency_to_accumulate, energy_to_accumulate #energy_to_accumulate add
    
class SupernetLoss(nn.Module):
    def __init__(self):
        super(SupernetLoss, self).__init__()
        self.alpha = CONFIG_SUPERNET['loss']['alpha']
        self.beta = CONFIG_SUPERNET['loss']['beta']
        self.gamma = CONFIG_SUPERNET['loss']['gamma']
        self.delta = CONFIG_SUPERNET['loss']['delta']
        self.weight_criterion = nn.CrossEntropyLoss()
    
    def forward(self, outs, targets, latency, energy, losses_ce, losses_lat, losses_energy, N): #energy add
        ce = self.weight_criterion(outs, targets)
        cal_loss = ce.to('cpu').detach().numpy().copy()
        prec1, _ = accuracy(outs, targets, topk=(1, 3)) #追加
        prec1 = prec1.to('cpu').detach().numpy().copy()
        
        '''   
        rate = prec1 / 0.35
        if prec1 >= 0.35:
            ce = torch.sub(ce, ce)
        else:
            ce = torch.sub(ce, cal_loss * rate)
        '''
        #ce = torch.add(ce, 1.0)
        

        if torch.isnan(ce) or ce < 0.0: #改良
            torch.tensor(0.0, requires_grad=True)
        
        cal_lat = torch.log(latency ** self.beta) #出力用, 改良
        #energy_cal = torch.log(energy ** self.beta) #出力用, 改良
        energy_cal = torch.log(energy ** self.delta)

        energy_cal = energy_cal.to('cpu').detach().numpy().copy() #計算用
        cal_lat = cal_lat.to('cpu').detach().numpy().copy() #計算用
        
        if energy_cal < 0 or np.isnan(energy_cal): #改良
            energy_cal = np.zeros(1)
        if cal_lat < 0 or np.isnan(cal_lat): #改良
            cal_lat = np.zeros(1)
        

        lat = torch.log(latency ** self.beta) #original
        #energy = torch.log(energy ** self.beta)
        energy = torch.log(energy ** self.delta)

        losses_ce.update(ce.item(), N)
        losses_lat.update(cal_lat.item(), N)
        losses_energy.update(energy_cal.item(), N) 
        
        if energy < 0 or torch.isnan(energy):
            energy = torch.tensor(0.0)
        if lat < 0 or torch.isnan(lat):
            lat = torch.tensor(0.0)
        
        #loss = self.alpha * ce * (lat + energy)
        loss = ce + (self.alpha * lat) + (self.gamma * energy)
        #loss = ce * (lat + energy)
        #loss = self.alpha * (ce + lat + energy)
        #loss = ce + lat + energy
        if loss < 0 or torch.isnan(loss):
            loss = torch.tensor(0.0, requires_grad=True)
        
        return loss
