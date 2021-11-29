
import torch
import torch.nn.functional as F
from lib.training.training import TrainingBase
from lib.training.training_mixins import SaveModel, VerboseLR
from lib.base.dotdict import HDict
from lib.data.dataset import StridedWindowedDataset
from .scheme_mixins import AccuracyMetric


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
    