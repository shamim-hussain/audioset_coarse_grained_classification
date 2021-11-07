
from lib.training.training import TrainingBase
from lib.training.training_mixins import SaveModel, ReduceLR, VerboseLR
from lib.base.dotdict import HDict
import torch

class SwishNetTraining(ReduceLR,VerboseLR,SaveModel,TrainingBase):
    def get_default_config(self):
        config = super().get_default_config()
        config.update(
            model_name          = 'swishnet',
            dataset_path        = HDict.L('c:"/gpfs/u/home/DLTM/DLTMhssn/scratch/datasets/audioset-derived/audioset-derived.zip"'+
                                          ' if c.distributed else'+
                                          ' r"G:\datasets\audioset-derived.zip"'),
            dataset_name        = 'audioset',
            annotations_path    = 'data/train_test_splits/train_dataset.csv.zip',
            ytids_path          = 'data/train_test_splits/train_ytids_folded.json',
            train_folds         = [0,1,2,3],
            val_folds           = [4],
            save_path           = HDict.L('c:path.join(f"models/{c.dataset_name.lower()}",c.model_name)'),
            width_multiplier    = 1.0,
        )
        return config
    
    
    def get_dataset_config(self):
        config = self.config
        dataset_config = dict(
            dataset_path = config.dataset_path,
        )
        return dataset_config, None
    
    def get_model_config(self):
        config = self.config
        model_config = dict(
            width_multiplier = config.width_multiplier,
        )
        return model_config, None
    
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
    
