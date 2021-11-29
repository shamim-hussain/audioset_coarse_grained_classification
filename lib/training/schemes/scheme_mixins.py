
import torch
import numpy as np
from lib.training.training import TrainingBase

class AccuracyMetric:
    def initialize_metrics(self, logs, training):
        super().initialize_metrics(logs, training)
        self._accuracy_correct = np.array(0., dtype='float32')
        self._accuracy_total = np.array(0., dtype='float32')
    
    def update_metrics(self, outputs, inputs, logs, training):
        super().update_metrics(outputs, inputs, logs, training)
        y_true = inputs['label'].long()
        y_pred = outputs.detach().argmax(-1)
        
        new_correct = torch.sum((y_pred==y_true).float())
        new_total = torch.tensor(y_true.shape[0], 
                                 dtype=new_correct.dtype, 
                                 device=new_correct.device)
        
        if self.is_distributed:
            torch.distributed.all_reduce(new_correct)
            torch.distributed.all_reduce(new_total)
        

        self._accuracy_correct += new_correct.item()
        self._accuracy_total += new_total.item()
        
        accuracy_val = (self._accuracy_correct/self._accuracy_total).tolist()
        self.update_logs(logs=logs, training=training, acc=accuracy_val)
        
    def get_verbose_logs(self):
        vl = super().get_verbose_logs()
        vl.update(acc='.3%')
        return vl



class CosineAnnealWarmRestart(TrainingBase):
    def get_default_config(self):
        config = super().get_default_config()
        config.update(
            min_lr = 1e-5,
            anneal_epochs_0 = 1,
            anneal_epochs_mult = 2,
        )
        return config
    def on_batch_begin(self, i, logs, training):
        super().on_batch_begin(i, logs, training)
        if training:
            start, period = 0, self.config.anneal_epochs_0
            while self.state.current_epoch > start + period:
                start += period + 1
                period *= self.config.anneal_epochs_mult
                
            new_lr = self.config.min_lr + 0.5 * (self.config.max_lr - self.config.min_lr)\
                                    * (1 + np.cos(np.pi * (self.state.current_epoch - start) / period))
            for group in self.optimizer.param_groups:
                group['lr'] = new_lr
            logs['lr'] = float(new_lr)
            logs['current_start'] = int(start)
            logs['next_restart'] = int(start + period)
            
    def log_description(self, i, logs, training):
        descriptions = super().log_description(i, logs, training)
        if training:
            descriptions.append(f'[WR:{logs["current_start"]+1}-{logs["next_restart"]+1}]')
        return descriptions

