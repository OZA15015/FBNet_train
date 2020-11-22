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
n = 0
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

class TrainerSupernet:
    def __init__(self, criterion, w_optimizer, theta_optimizer, w_scheduler, logger, writer, experiment):
        self.top1       = AverageMeter()
        self.top3       = AverageMeter()
        self.losses     = AverageMeter()
        self.losses_lat = AverageMeter()
        self.losses_ce  = AverageMeter()
        
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
        
        best_top1 = 0.0
        best_lat = 1000000
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
            
            top1_avg, lat_avg = self._validate(model, test_loader, epoch)
            #if best_top1 < top1_avg and lat_avg < best_lat:
            #if best_top1 < top1_avg: #original
            #if top1_avg >= 0.65  and lat_avg < best_lat:    
            if best_top1 < top1_avg:
                best_top1 = top1_avg
                best_lat = lat_avg
                self.logger.info("Best top1 acc by now. Save model")
                save(model, self.path_to_save_model)
            
            self.temperature = self.temperature * self.exp_anneal_rate
        
    def _training_step(self, model, loader, optimizer, epoch, info_for_logger=""):
        model = model.train()
        start_time = time.time()
        global i
        #n = 0
        #time_sum = 0
        for step, (X, y) in enumerate(loader):
            i#X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
            # X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.shape[0]
            optimizer.zero_grad()
            latency_to_accumulate = Variable(torch.Tensor([[0.0]]), requires_grad=True).to(device) #cuda()
            start_time = time.time()
            outs, latency_to_accumulate = model(X, self.temperature, latency_to_accumulate)
            #time_sum += time.time() - start_time
            #print(time.time() - start_time)
            #n += 1
            loss = self.criterion(outs, y, latency_to_accumulate, self.losses_ce, self.losses_lat, N)
            loss.backward()
            optimizer.step()
            
            self._intermediate_stats_logging(outs, y, loss, step, epoch, N, len_loader=len(loader), val_or_train="Train")

            '''
            print(self.losses_ce)
            print(type(self.losses_ce))
            print(self.losses_lat)
            print(type(self.losses_lat))
            print(loss)
            print(type(loss))
            self.top1 = np.array(self.top1)
            print(type(self.top1))
            print(self.top1)
            quit()
            '''
        self._epoch_stats_logging(start_time=start_time, epoch=epoch, info_for_logger=info_for_logger, val_or_train='train')
        for avg in [self.top1, self.top3, self.losses]:
            avg.reset()
        
    def _validate(self, model, loader, epoch):
        model.eval()
        start_time = time.time()

        with torch.no_grad():
            for step, (X, y) in enumerate(loader):
                #X, y = X.cuda(), y.cuda()
                X, y = X.to(device), y.to(device)
                N = X.shape[0]
                
                latency_to_accumulate = torch.Tensor([[0.0]]).to(device) #.cuda()
                outs, latency_to_accumulate = model(X, self.temperature, latency_to_accumulate)
                loss = self.criterion(outs, y, latency_to_accumulate, self.losses_ce, self.losses_lat, N)

                self._intermediate_stats_logging(outs, y, loss, step, epoch, N, len_loader=len(loader), val_or_train="Valid")
                '''
                global n
                n += 1
                list1 = ["acc1", "acc3", "ce_loss", "lat_loss"]
                cnt = 0
                for word in [self.top1, self.top3, self.losses_ce, self.losses_lat]:
                    word = str(word)
                    word = word.replace(' ', '')
                    word = word.replace(':', '')
                    experiment.log_metric("val_" + list1[cnt], float(word), step=n)
                    print(list1[cnt] + ": " + word)
                    cnt += 1
                '''
        top1_avg = self.top1.get_avg()
        self._epoch_stats_logging(start_time=start_time, epoch=epoch, val_or_train='val')
        for avg in [self.top1, self.top3, self.losses]:
            avg.reset()
        return top1_avg, self.losses_lat.get_avg() #lat追加
    
    def _epoch_stats_logging(self, start_time, epoch, val_or_train, info_for_logger=''):
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_loss'+info_for_logger, self.losses.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_top1'+info_for_logger, self.top1.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_top3'+info_for_logger, self.top3.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_losses_lat'+info_for_logger, self.losses_lat.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_losses_ce'+info_for_logger, self.losses_ce.get_avg(), epoch)
        
        top1_avg = self.top1.get_avg()
        self.logger.info(info_for_logger+val_or_train + ": [{:3d}/{}] Final Prec@1 {:.4%} Time {:.2f}".format(
            epoch+1, self.cnt_epochs, top1_avg, time.time() - start_time))
        
    def _intermediate_stats_logging(self, outs, y, loss, step, epoch, N, len_loader, val_or_train):
        prec1, prec3 = accuracy(outs, y, topk=(1, 5))
        self.losses.update(loss.item(), N)
        self.top1.update(prec1.item(), N)
        self.top3.update(prec3.item(), N)
        global i
        i += 1
        list1 = ["acc1", "acc3", "ce_loss", "lat_loss"]
        cnt = 0
        for word in [self.top1, self.top3, self.losses_ce, self.losses_lat]:
            word = str(word)
            word = word.replace(' ', '')
            word = word.replace(':', '')
            self.experiment.log_metric("train_" + list1[cnt], float(word), step=i)
            #print(list1[cnt] + ": " + word)
            cnt += 1
        total_loss = loss.to('cpu').detach().numpy().copy()
        total_loss = str(total_loss)
        total_loss =  total_loss.replace('[[', '')
        total_loss =  total_loss.replace(']]', '')
        
        self.experiment.log_metric("train_totalLoss", float(total_loss), step=i)
        #print("loss: " + str(loss))
        #self.experiment.log_metric("batch_accuracy", float(acc), step=i)        
       
        if (step > 1 and step % self.print_freq == 0) or step == len_loader - 1:
            self.logger.info(val_or_train+
               ": [{:3d}/{}] Step {:03d}/{:03d} Loss {:.3f} "
               "Prec@(1,3) ({:.1%}, {:.1%}), ce_loss {:.3f}, lat_loss {:.3f}".format(
                   epoch + 1, self.cnt_epochs, step, len_loader - 1, self.losses.get_avg(),
                   self.top1.get_avg(), self.top3.get_avg(), self.losses_ce.get_avg(), self.losses_lat.get_avg()))
