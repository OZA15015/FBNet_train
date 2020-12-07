from comet_ml import Experiment
import torch
from torch.autograd import Variable
import time
from general_functions.utils import AverageMeter, save, accuracy
from supernet_functions.config_for_supernet import CONFIG_SUPERNET
import numpy as np

#----以下全て, 再現性関連
import random
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

i = 0
j = 0
k = 0
n = 0
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class TrainerSupernet:
    def __init__(self, criterion, w_optimizer, theta_optimizer, w_scheduler, logger, writer, experiment):
        self.top1       = AverageMeter()
        self.top3       = AverageMeter()
        self.losses     = AverageMeter()
        self.losses_lat = AverageMeter()
        self.losses_ce  = AverageMeter()
        self.losses_energy = AverageMeter() #energy add 
        self.logger = logger
        self.writer = writer
        
        self.criterion = criterion
        self.w_optimizer = w_optimizer
        self.theta_optimizer = theta_optimizer
        self.w_scheduler = w_scheduler
        self.experiment = experiment
        
        self.temperature                 = CONFIG_SUPERNET['train_settings']['init_temperature']
        self.exp_anneal_rate             = CONFIG_SUPERNET['train_settings']['exp_anneal_rate'] # apply it every epoch
        self.cnt_epochs                  = CONFIG_SUPERNET['train_settings']['cnt_epochs']
        self.train_thetas_from_the_epoch = CONFIG_SUPERNET['train_settings']['train_thetas_from_the_epoch']
        self.print_freq                  = CONFIG_SUPERNET['train_settings']['print_freq']
        self.path_to_save_model          = CONFIG_SUPERNET['train_settings']['path_to_save_model']
    
    def train_loop(self, train_w_loader, train_thetas_loader, test_loader, model):
        global n
        best_top1 = 0.0
        best_lat = 10000000
        best_energy = 10000000
        n = 1
        # firstly, train weights only
        for epoch in range(self.train_thetas_from_the_epoch):
            self.writer.add_scalar('learning_rate/weights', self.w_optimizer.param_groups[0]['lr'], epoch)
            
            self.logger.info("Firstly, start to train weights for epoch %d" % (epoch))
            self._training_step(model, train_w_loader, self.w_optimizer, epoch, info_for_logger="_w_step_")
            self.w_scheduler.step()
        
        for epoch in range(self.train_thetas_from_the_epoch, self.cnt_epochs):
            self.writer.add_scalar('learning_rate/weights', self.w_optimizer.param_groups[0]['lr'], epoch)
            self.writer.add_scalar('learning_rate/theta', self.theta_optimizer.param_groups[0]['lr'], epoch)
            
            self.logger.info("Start to train weights for epoch %d" % (epoch))
            self._training_step(model, train_w_loader, self.w_optimizer, epoch, info_for_logger="_w_step_")
            self.w_scheduler.step()
            
            self.logger.info("Start to train theta for epoch %d" % (epoch))
            self._training_step(model, train_thetas_loader, self.theta_optimizer, epoch, info_for_logger="_theta_step_")
            
            top1_avg, lat_avg, energy_avg = self._validate(model, test_loader, epoch)
            #if best_top1 < top1_avg and lat_avg < best_lat:
            #if best_top1 < top1_avg: #original
                
            '''
            if (best_top1 < top1_avg  and lat_avg < best_lat and energy_avg < best_energy) or best_top1 < top1_avg:
                if best_top1 < top1_avg:
                    best_top1 = top1_avg
                    print("Best Acc!!")
                if lat_avg < best_lat:
                    best_lat = lat_avg
                    print("Best Speed!!")
                if energy_avg < best_energy:
                    best_energy = energy_avg
                    print("Best Energy!!")
                self.logger.info("Best top1 acc by now. Save model")
                #print("Over Acc: 0.70")
                #print("Model Number = " + str(n))
                save(model, self.path_to_save_model + str(n) + '.pth')
                #n += 1
            '''
            if (top1_avg >= 0.35  and lat_avg < best_lat) or (top1_avg >= 0.35  and energy_avg < best_energy) :
                if lat_avg < best_lat:
                    best_lat = lat_avg
                    print("Best Latency!!")
                if energy_avg < best_energy:
                     best_energy = energy_avg
                     print("Best Energy!!")
                self.logger.info("Best top1 acc by now. Save model")
                print("Over Acc: 0.35")
                #print("Model Number = " + str(n))
                save(model, self.path_to_save_model + str(n) + '.pth')
                #n+=1
             
            self.temperature = self.temperature * self.exp_anneal_rate
        
    def _training_step(self, model, loader, optimizer, epoch, info_for_logger=""):
        model = model.train()
        start_time = time.time()
        global i
        global j
        for step, (X, y) in enumerate(loader):
            #X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
            # X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.shape[0]
            optimizer.zero_grad()
            latency_to_accumulate = Variable(torch.Tensor([[0.0]]), requires_grad=True).to(device) 
            energy_to_accumulate = Variable(torch.Tensor([[0.0]]), requires_grad=True).to(device) #energy add
            outs, latency_to_accumulate, energy_to_accumulate = model(X, self.temperature, latency_to_accumulate, energy_to_accumulate) #energy add
            loss = self.criterion(outs, y, latency_to_accumulate, energy_to_accumulate, self.losses_ce, self.losses_lat, self.losses_energy, N) #energy add
            loss.backward()
            optimizer.step()
            
            self._intermediate_stats_logging(outs, y, loss, step, epoch, N, len_loader=len(loader), val_or_train="Train")
            if info_for_logger == "_w_step_":
                i += 1
            elif info_for_logger == "_theta_step_":
                j += 1
            list1 = ["acc1", "acc3", "ce_loss", "lat_loss", "energy_loss"]
            cnt = 0
            for word in [self.top1, self.top3, self.losses_ce, self.losses_lat, self.losses_energy]:
                word = str(word)
                word = word.replace(' ', '')
                word = word.replace(':', '')
                if  info_for_logger == "_w_step_":
                    self.experiment.log_metric("w_train_" + list1[cnt], float(word), step=i)
                elif info_for_logger == "_theta_step_":
                    self.experiment.log_metric("theta_train_" + list1[cnt], float(word), step=j)
                cnt += 1
            total_loss = loss.to('cpu').detach().numpy().copy()
            total_loss = str(total_loss)
            total_loss =  total_loss.replace('[[', '')
            total_loss =  total_loss.replace(']]', '')
            if  info_for_logger == "_w_step_":
                self.experiment.log_metric("w_train_totalLoss", float(total_loss), step=i)
            elif info_for_logger == "_theta_step_":
                self.experiment.log_metric("theta_train_totalLoss", float(total_loss), step=j)            

        self._epoch_stats_logging(start_time=start_time, epoch=epoch, info_for_logger=info_for_logger, val_or_train='train')
        for avg in [self.top1, self.top3, self.losses]:
            avg.reset()
        
    def _validate(self, model, loader, epoch):
        model.eval()
        start_time = time.time()
        global k
        with torch.no_grad():
            for step, (X, y) in enumerate(loader):
                #X, y = X.cuda(), y.cuda()
                X, y = X.to(device), y.to(device)
                N = X.shape[0]
                latency_to_accumulate = torch.Tensor([[0.0]]).to(device)
                energy_to_accumulate = torch.Tensor([[0.0]]).to(device) #energy add

                outs, latency_to_accumulate, energy_to_accumulate = model(X, self.temperature, latency_to_accumulate, energy_to_accumulate) #energy add
                loss = self.criterion(outs, y, latency_to_accumulate, energy_to_accumulate, self.losses_ce, self.losses_lat, self.losses_energy, N) #energy add

                self._intermediate_stats_logging(outs, y, loss, step, epoch, N, len_loader=len(loader), val_or_train="Valid")
                k += 1
                list1 = ["acc1", "acc3", "ce_loss", "lat_loss", "energy_loss"]
                cnt = 0
                for word in [self.top1, self.top3, self.losses_ce, self.losses_lat, self.losses_energy]:
                    word = str(word)
                    word = word.replace(' ', '')
                    word = word.replace(':', '')
                    self.experiment.log_metric("val_" + list1[cnt], float(word), step=k)
                    cnt += 1
                total_loss = loss.to('cpu').detach().numpy().copy()
                total_loss = str(total_loss)
                total_loss =  total_loss.replace('[[', '')
                total_loss =  total_loss.replace(']]', '')

                self.experiment.log_metric("val_totalLoss", float(total_loss), step=k)
        top1_avg = self.top1.get_avg()
        self._epoch_stats_logging(start_time=start_time, epoch=epoch, val_or_train='val')
        for avg in [self.top1, self.top3, self.losses]:
            avg.reset()
        return top1_avg, self.losses_lat.get_avg(), self.losses_energy.get_avg() #lat, energy追加
    
    def _epoch_stats_logging(self, start_time, epoch, val_or_train, info_for_logger=''):
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_loss'+info_for_logger, self.losses.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_top1'+info_for_logger, self.top1.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_top3'+info_for_logger, self.top3.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_losses_lat'+info_for_logger, self.losses_lat.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_losses_ce'+info_for_logger, self.losses_ce.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_losses_energy'+info_for_logger, self.losses_energy.get_avg(), epoch) #energy add
        
        top1_avg = self.top1.get_avg()
        self.logger.info(info_for_logger+val_or_train + ": [{:3d}/{}] Final Prec@1 {:.4%} Time {:.2f}".format(
            epoch+1, self.cnt_epochs, top1_avg, time.time() - start_time))
        
    def _intermediate_stats_logging(self, outs, y, loss, step, epoch, N, len_loader, val_or_train):
        prec1, prec3 = accuracy(outs, y, topk=(1, 5))
        self.losses.update(loss.item(), N)
        self.top1.update(prec1.item(), N)
        self.top3.update(prec3.item(), N)
        
        if (step > 1 and step % self.print_freq == 0) or step == len_loader - 1:
            self.logger.info(val_or_train+
               ": [{:3d}/{}] Step {:03d}/{:03d} Loss {:.3f} "
               "Prec@(1,3) ({:.1%}, {:.1%}), ce_loss {:.3f}, lat_loss {:.3f}, energy_loss {:.3f}".format(
                   epoch + 1, self.cnt_epochs, step, len_loader - 1, self.losses.get_avg(),
                   self.top1.get_avg(), self.top3.get_avg(), self.losses_ce.get_avg(), self.losses_lat.get_avg(), self.losses_energy.get_avg())) #energy add
