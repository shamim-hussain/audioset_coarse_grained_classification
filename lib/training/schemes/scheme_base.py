
import torch
import torch.nn.functional as F
from lib.training.training import TrainingBase
from lib.training.testing import TestingBase
from lib.training.training_mixins import SaveModel, VerboseLR
from lib.base.dotdict import HDict
from lib.data.dataset import StridedWindowedDataset
from .scheme_mixins import AccuracyMetric
import sys, os
import numpy as np


class AudiosetTraining(AccuracyMetric,VerboseLR,SaveModel,TrainingBase):
    def get_default_config(self):
        config = super().get_default_config()
        config.update(
            model_name          = 'unnamed_model',
            dataset_path        = HDict.L('c:"/gpfs/u/home/DLTM/DLTMhssn/scratch/datasets/audioset-derived/audioset-derived-features.zip"'+
                                          ' if c.distributed else'+
                                          ' "G:/datasets/audioset-derived.zip"'),
            dataset_name        = 'audioset',
            annotations_path    = 'data/annotations.csv.zip',
            ytids_path          = 'data/train_test_splits/train_ytids_folded.json',
            batch_size          = 128,
            train_folds         = [0,1,2,3,-1],
            val_folds           = [4],
            save_path           = HDict.L('c:path.join(f"models/{c.dataset_name.lower()}",c.model_name)'),
            window_size         = 200,
            window_stride       = HDict.L('c:c.window_size//2'),
            pad_value           = -16.,
        )
        return config
    
    
    def get_dataset_config(self):
        config = self.config
        dataset_config = dict(
            dataset_path      = config.dataset_path,
            annotations_path  = config.annotations_path,
            ytids_path        = config.ytids_path,
            window_size       = config.window_size,
            window_stride     = config.window_stride,
            windowed_features = ['log_mfb'],
            pad_value         = config.pad_value,
        )
        return dataset_config, StridedWindowedDataset
    
    def get_model_config(self):
        return {}, None
    
    @property
    def train_dataset(self):
        try:
            return self._train_dataset
        except AttributeError:
            config = self.config
            dataset_config, dataset_class = self.get_dataset_config()
            if dataset_class is None:
                raise NotImplementedError
            self._train_dataset = dataset_class(**dataset_config,
                                                ytid_keys=[str(k) for k in 
                                                           config.train_folds])
            self._train_dataset.load_data(verbose=self.is_main_rank)
            return self._train_dataset
    
    @property
    def val_dataset(self):
        try:
            return self._val_dataset
        except AttributeError:
            config = self.config
            dataset_config, dataset_class = self.get_dataset_config()
            if dataset_class is None:
                raise NotImplementedError
            self._val_dataset = dataset_class(**dataset_config,
                                              ytid_keys=[str(k) for k in 
                                                           config.val_folds])
            self._val_dataset.load_data(verbose=self.is_main_rank)
            return self._val_dataset
    
    @property
    def base_model(self):
        try:
            return self._base_model
        except AttributeError:
            model_config, model_class = self.get_model_config()
            if model_class is None:
                raise NotImplementedError
            self._base_model = model_class(**model_config).cuda()
            return self._base_model
    
    def calculate_loss(self, outputs, inputs):
        return F.nll_loss(outputs, inputs['label'])



class AudiosetTesting(TestingBase, AudiosetTraining):
    def __init__(self, config=None, ddp_rank=0, ddp_world_size=1):
        super().__init__(config, ddp_rank, ddp_world_size)
        if not self.is_main_rank:
            sys.exit(0)
    
    @property
    def test_dataset(self):
        try:
            return self._test_dataset
        except AttributeError:
            config = self.config
            dataset_config, dataset_class = self.get_dataset_config()
            if dataset_class is None:
                raise NotImplementedError
            dataset_config = dataset_config.copy()
            dataset_config.update(ytids_path = config.test_ytids_path)
            self._test_dataset = dataset_class(**dataset_config)
            self._test_dataset.load_data(verbose=self.is_main_rank)
            return self._test_dataset
    
    def get_default_config(self):
        config = super().get_default_config()
        config.update(
            config_path = HDict.L('c:path.join(c.save_path,"test_config")'),
            test_ytids_path = 'data/train_test_splits/test_ytids.json',
        )
        return config
    
    def prepare_for_testing(self):
        super().prepare_for_testing()
        self.save_config_file()
    
    def get_dataset_config(self):
        dataset_config, dataset_class = super().get_dataset_config()
        dataset_config.update(
            include_ytid = True,
            include_window = True
        )
        return dataset_config, dataset_class
    
    def preprocess_batch(self, batch):
        processed_batch = batch.copy()
        for k,v in batch.items():
            if k == 'ytid':
                k = np.array(v)
            elif k in ['window','label']:
                processed_batch[k] = v.cpu().numpy()
            else:
                processed_batch[k] = v.cuda()
        return processed_batch
    
    def test_step(self, batch):
        outputs = {k:batch[k] for k in ['ytid','window','label']}
        outputs.update(predictions = super().test_step(batch).cpu().numpy())
        return outputs
    
    def preprocess_predictions(self, outputs):
        collated = {}
        for k in outputs[0]:
            collated[k] = np.concatenate([v[k] for v in outputs])
        return collated
    
    def postprocess_predictions(self, outputs):
        return outputs
    
    def save_predictions(self, dataset_name, predictions):
        os.makedirs(self.config.predictions_path, exist_ok=True)
        predictions_file = os.path.join(self.config.predictions_path, f'{dataset_name}.pt')
        torch.save(predictions, predictions_file)
        print(f'Saved predictions to {predictions_file}')
    
    def evaluate_on(self, dataset_name, dataset, predictions):
        from sklearn.metrics import accuracy_score, confusion_matrix
        y_true = predictions['label']
        y_pred = predictions['predictions'].argmax(axis=-1)
        acc = accuracy_score(y_true, y_pred)
        cmat = confusion_matrix(y_true, y_pred, normalize='true')
        results = dict(
            accuracy = acc,
            confusion_matrix = cmat,
        )
        for k,v in results.items():
            if hasattr(v, 'tolist'):
                results[k] = v.tolist()
        return results
        
    
    