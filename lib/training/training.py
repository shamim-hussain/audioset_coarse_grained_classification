import torch
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.data import Sampler

import yaml
from yaml import SafeLoader as yaml_Loader, SafeDumper as yaml_Dumper
import os,sys

from tqdm import tqdm

from lib.base.dotdict import HDict
HDict.L.update_globals({'path':os.path})

def str_presenter(dumper, data):
  if len(data.splitlines()) > 1:  # check for multiline string
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
  return dumper.represent_scalar('tag:yaml.org,2002:str', data)
yaml.representer.SafeRepresenter.add_representer(str, str_presenter)


def read_config_from_file(config_file):
    with open(config_file, 'r') as fp:
        return yaml.load(fp, Loader=yaml_Loader)

def save_config_to_file(config, config_file):
    with open(config_file, 'w') as fp:
        return yaml.dump(config, fp, Dumper=yaml_Dumper)


class StopTrainingException(Exception):
    pass


class DistributedTestDataSampler(Sampler):
    def __init__(self, data_source, batch_size, rank, world_size):
        data_len = len(data_source)
        all_indices = np.arange(data_len, dtype=int)
        split_indices = np.array_split(all_indices, world_size)
        
        num_batches = (len(split_indices[0]) + batch_size -1) // batch_size
        self.batch_indices = [i.tolist() for i in np.array_split(split_indices[rank],
                                                                 num_batches)]
    
    def __iter__(self):
        return iter(self.batch_indices)
    
    def __len__(self):
        return len(self.batch_indices)


class TrainingBase:
    def __init__(self, config=None, ddp_rank=0, ddp_world_size=1):
        self.config_input = config
        self.config = self.get_default_config()
        if config is not None:
            for k in config.keys():
                if not k in self.config:
                    raise KeyError(f'Unknown config "{k}"')
            self.config.update(config)
        
        self.state = self.get_default_state()
        
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.is_distributed = (self.ddp_world_size > 1)
        self.is_main_rank = (self.ddp_rank == 0)


    @property
    def train_dataset(self):
        raise NotImplementedError
    
    @property
    def val_dataset(self):
        raise NotImplementedError
    
    @property
    def collate_fn(self):
        return None

    @property
    def train_sampler(self):
        try:
            return self._train_sampler
        except AttributeError:
            self._train_sampler = torch.utils.data.DistributedSampler(self.train_dataset,
                                                                      shuffle=True)
            return self._train_sampler
    
    @property
    def train_dataloader(self):
        try:
            return self._train_dataloader
        except AttributeError:
            if not self.is_distributed:
                self._train_dataloader = DataLoader(dataset=self.train_dataset,
                                                    batch_size=self.config.batch_size,
                                                    shuffle=True,
                                                    drop_last=False,
                                                    collate_fn=self.collate_fn,
                                                    pin_memory=True)
            else:
                self._train_dataloader = DataLoader(dataset=self.train_dataset,
                                                    batch_size=self.config.batch_size,
                                                    collate_fn=self.collate_fn,
                                                    sampler=self.train_sampler,
                                                    pin_memory=True)
            return self._train_dataloader
    
    @property
    def val_dataloader(self):
        try:
            return self._val_dataloader
        except AttributeError:
            if not self.is_distributed:
                self._val_dataloader = DataLoader(dataset=self.val_dataset,
                                                  batch_size=self.config.batch_size,
                                                  shuffle=False,
                                                  drop_last=False,
                                                  collate_fn=self.collate_fn,
                                                  pin_memory=True)
            else:
                sampler = DistributedTestDataSampler(data_source=self.val_dataset,
                                                 batch_size=self.config.batch_size,
                                                 rank=self.ddp_rank,
                                                 world_size=self.ddp_world_size)
                self._val_dataloader = DataLoader(dataset=self.val_dataset,
                                                  collate_fn=self.collate_fn,
                                                  batch_sampler=sampler,
                                                  pin_memory=True)
            return self._val_dataloader

    @property
    def base_model(self):
        raise NotImplementedError
    
    @property
    def model(self):
        try:
            return self._model
        except AttributeError:
            self._model = self.base_model
            if self.is_distributed:
                self._model = torch.nn.parallel.DistributedDataParallel(self._model,
                                                                        device_ids=[self.ddp_rank],
                                                                        output_device=self.ddp_rank)
            return self._model
    
    @property
    def optimizer(self):
        try:
            return self._optimizer
        except AttributeError:
            config = self.config
            optimizer_class = getattr(torch.optim, config.optimizer)
            self._optimizer = optimizer_class(self.model.parameters(),
                                              lr=config.max_lr, 
                                              **config.optimizer_params)
            return self._optimizer

    def get_default_config(self):
        return HDict(
            scheme               = None,
            model_name           = 'unnamed_model',
            distributed          = False,
            random_seed          = None,
            num_epochs           = 100,
            save_path            = HDict.L('c:path.join("models",c.model_name)'),
            checkpoint_path      = HDict.L('c:path.join(c.save_path,"checkpoint")'),
            config_path          = HDict.L('c:path.join(c.save_path,"config")'),
            summary_path         = HDict.L('c:path.join(c.save_path,"summary")'),
            log_path             = HDict.L('c:path.join(c.save_path,"logs")'),
            validation_frequency = 1,
            batch_size           = HDict.L('c:128 if c.distributed else 32'),
            optimizer            = 'Adam'    ,
            max_lr               = 5e-4      ,
            optimizer_params     = {}        ,
        )
    
    def get_default_state(self):
        state =  HDict(
            current_epoch = 0,
            global_step = 0,
        )
        return state
    
    def config_summary(self):
        if not self.is_main_rank: return
        for k,v in self.config.get_dict().items():
            print(f'{k} : {v}', flush=True)
    
    def save_config_file(self):
        if not self.is_main_rank: return
        os.makedirs(os.path.dirname(self.config.config_path), exist_ok=True)
        save_config_to_file(self.config.get_dict(), self.config.config_path+'.yaml')
        save_config_to_file(self.config_input, self.config.config_path+'_input.yaml')
    
    def model_summary(self):
        if not self.is_main_rank: return
        os.makedirs(os.path.dirname(self.config.summary_path), exist_ok=True)
        trainable_params = 0
        non_trainable_params = 0
        for p in self.model.parameters():
            if p.requires_grad:
                trainable_params += p.numel()
            else:
                non_trainable_params += p.numel()
        summary = dict(
            trainable_params = trainable_params,
            non_trainable_params = non_trainable_params,
            model_representation = repr(self.model),
        )
        with open(self.config.summary_path+'.txt', 'w') as fp:
            yaml.dump(summary, fp, Dumper=yaml_Dumper)
    
    def save_checkpoint(self):
        if not self.is_main_rank: return
        ckpt_path = self.config.checkpoint_path
        os.makedirs(ckpt_path, exist_ok=True)
        
        torch.save(self.state, os.path.join(ckpt_path, 'training_state'))
        torch.save(self.base_model.state_dict(), os.path.join(ckpt_path, 'model_state'))
        torch.save(self.optimizer.state_dict(), os.path.join(ckpt_path, 'optimizer_state'))
        print(f'Checkpoint saved to: {ckpt_path}',flush=True)
    
    def load_checkpoint(self):
        ckpt_path = self.config.checkpoint_path
        try:
            self.state.update(torch.load(os.path.join(ckpt_path, 'training_state')))
            self.base_model.load_state_dict(torch.load(os.path.join(ckpt_path, 'model_state')))
            self.optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, 'optimizer_state')))
            if self.is_main_rank:
                print(f'Checkpoint loaded from: {ckpt_path}',flush=True)
        except FileNotFoundError:
            pass
    
    # Callbacks
    def on_train_begin(self):
        pass
    def on_train_end(self):
        pass
    def on_epoch_begin(self, logs, training):
        pass
    def on_epoch_end(self, logs, training):
        pass
    def on_batch_begin(self, i, logs, training):
        pass
    def on_batch_end(self, i, logs, training):
        pass
    
    
    # Logging
    def get_verbose_logs(self):
        return OrderedDict(loss='0.4f')
    
    @property
    def verbose_logs(self):
        try:
            return self._verbose_logs
        except AttributeError:
            self._verbose_logs = self.get_verbose_logs()
            return self._verbose_logs
    
    def update_logs(self, logs, training, **updates):
        if training:
            logs.update(updates)
        else:
            logs.update(('val_'+k,v) for k,v in updates.items())
    
    def log_description(self, i, logs, training):
        if training:
            return list(f'{k} = {logs[k]:{f}}' 
                        for k,f in self.verbose_logs.items())
        else:
            return list(f'val_{k} = {logs["val_"+k]:{f}}' 
                        for k,f in self.verbose_logs.items())
    
    
    # Training loop
    def preprocess_batch(self, batch):
        return dict((k,v.cuda()) for k,v in batch.items())
    
    def calculate_loss(self, outputs, inputs):
        raise NotImplementedError
    
    def training_step(self, batch, logs):
        for param in self.model.parameters():
            param.grad = None
        outputs = self.model(batch)
        loss = self.calculate_loss(outputs=outputs, inputs=batch)
        loss.backward()
        self.optimizer.step()
        return outputs, loss
    
    def validation_step(self, batch, logs):
        outputs = self.model(batch)
        loss = self.calculate_loss(outputs=outputs, inputs=batch)
        return outputs, loss
    
    def initialize_metrics(self, logs, training):
        pass
    
    def update_metrics(self, outputs, inputs, logs, training):
        pass
    
    def initialize_losses(self, logs, training):
        self._total_loss = 0.
    
    def update_losses(self, i, loss, inputs, logs, training):
        if not self.is_distributed:
            step_loss = loss.item()
        else:
            if training:
                loss = loss.detach()
            torch.distributed.all_reduce(loss)
            step_loss = loss.item()/self.ddp_world_size
        self._total_loss += step_loss
        self.update_logs(logs=logs, training=training,
                         loss=self._total_loss/(i+1))
        
    
    def train_epoch(self, epoch, logs):
        self.model.train()
        self.initialize_losses(logs, True)
        self.initialize_metrics(logs, True)
        
        if self.is_distributed:
            self.train_sampler.set_epoch(epoch)
            
        if self.is_main_rank:
            gen = tqdm(self.train_dataloader, dynamic_ncols=True)
        else:
            gen = self.train_dataloader
        try:
            for i, batch in enumerate(gen):
                self.on_batch_begin(i, logs, True)
                batch = self.preprocess_batch(batch)
                outputs, loss = self.training_step(batch, logs)
                
                self.state.global_step = self.state.global_step + 1
                logs.update(global_step=self.state.global_step)
                
                self.update_losses(i, loss, batch, logs, True)
                self.update_metrics(outputs, batch, logs, True)
                
                self.on_batch_end(i, logs, True)
                
                if self.is_main_rank:
                    desc = 'Training: '+'; '.join(self.log_description(i, logs, True))
                    gen.set_description(desc)
        finally:
            if self.is_main_rank: gen.close()
            for param in self.model.parameters():
                param.grad = None
    
    def validation_epoch(self, epoch, logs):
        self.model.eval()
        self.initialize_losses(logs, False)
        self.initialize_metrics(logs, False)
        
        if self.is_main_rank:
            gen = tqdm(self.val_dataloader, dynamic_ncols=True)
        else:
            gen = self.val_dataloader
        try:
            with torch.no_grad():
                for i, batch in enumerate(gen):
                    self.on_batch_begin(i, logs, False)
                    batch = self.preprocess_batch(batch)
                    outputs, loss = self.validation_step(batch, logs)
                    
                    self.update_losses(i, loss, batch, logs, False)
                    self.update_metrics(outputs, batch, logs, False)
                    
                    self.on_batch_end(i, logs, False)
                    
                    if self.is_main_rank:
                        desc = 'Validation: '+'; '.join(self.log_description(i, logs, False))
                        gen.set_description(desc)
        finally:
            if self.is_main_rank: gen.close()
    
    def load_history(self):
        history_file = os.path.join(self.config.log_path, 'history.yaml')
        try:
            with open(history_file, 'r') as fp:
                return yaml.load(fp, Loader=yaml_Loader)
        except FileNotFoundError:
            return []
    
    def save_history(self, history):
        os.makedirs(self.config.log_path, exist_ok=True)
        history_file = os.path.join(self.config.log_path, 'history.yaml')
        with open(history_file, 'w') as fp:
            yaml.dump(history, fp, Dumper=yaml_Dumper)

    
    def train_model(self):
        if self.is_main_rank: 
            history = self.load_history()
        starting_epoch = self.state.current_epoch
        
        self.on_train_begin()
        should_stop_training = False
        try:
            for i in range(starting_epoch, self.config.num_epochs):
                self.state.current_epoch = i
                if self.is_main_rank: 
                    print(f'\nEpoch {i+1}/{self.config.num_epochs}:', flush=True)
                logs = dict(epoch = self.state.current_epoch, 
                            global_step = self.state.global_step)
                
                try:
                    self.on_epoch_begin(logs, True)
                    self.train_epoch(i, logs)
                    self.on_epoch_end(logs, True)
                except StopTrainingException:
                    should_stop_training = True
                
                try:
                    if (self.val_dataloader is not None)\
                            and (not ((i+1) % self.config.validation_frequency)):
                        self.on_epoch_begin(logs, False)
                        self.validation_epoch(i, logs)
                    self.on_epoch_end(logs, False)
                except StopTrainingException:
                    should_stop_training = True
                
                self.state.current_epoch = i + 1
                if self.is_main_rank:
                    self.save_checkpoint()
                    
                    history.append(logs)
                    self.save_history(history)
                
                if should_stop_training:
                    if self.is_main_rank:
                        print('Stopping training ...')
                    break
        finally:
            self.on_train_end()
    
    
    # Interface
    def prepare_for_training(self):
        self.config_summary()
        self.save_config_file()
        self.load_checkpoint()
        self.model_summary()
        
    def execute_training(self):
        self.prepare_for_training()
        self.train_model()
        self.finalize_training()
    
    def finalize_training(self):
        pass
    
        
