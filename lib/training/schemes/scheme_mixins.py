
import torch
import numpy as np

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