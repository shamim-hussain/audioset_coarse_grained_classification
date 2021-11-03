import pandas as pd
import numpy as np
import json
import zipfile
from tqdm import tqdm

class Dataset:
    _log_mfb_shape = (None, 64)
    def __init__(self, annotations_path, dataset_path, ytids_path,
                 class_associations=(('speech',0),('music',1),('noise',2)),
                 load_data=True):
        self.annotations = pd.read_csv(annotations_path).set_index('ytid')
        self.dataset_path = dataset_path
        with open(ytids_path, 'r') as f:
            self.ytids = json.load(f)
        self.class_associations = dict(class_associations)
        self.reverse_class_associations = dict((v,k) for k,v in class_associations)
        
        if load_data:
            self.load_data()
    
    def load_data(self):
        self.data = {}
        with zipfile.ZipFile(self.dataset_path, 'r') as zf:
            for ytid in tqdm(self.ytids, desc='Loading data'):
                data = {}
                with zf.open(self.annotations.loc[ytid,'log_mfb_path']) as f:
                    data['log_mfb'] = np.load(f)
                
                self.data[ytid] = (data, self.class_associations[
                                       self.annotations.loc[ytid,'plausible_superclass']])
    
    def __getitem__(self, index):
        ytid = self.ytids[index]
        data_without_ytid, label = self.data[ytid]
        data = {'ytid':ytid}
        data.update(data_without_ytid)
        return data, label
    
    def __len__(self):
        return len(self.data)
    
    @property
    def tf_dataloader(self):
        try:
            return self._tf_dataloader
        except AttributeError:
            import tensorflow as tf
            
            _data = self.data
            _log_mfb_shape = self._log_mfb_shape
            def _load_data(ytid):
                data, label = _data[ytid.numpy().decode('utf-8')]
                log_mfb = data['log_mfb']
                return log_mfb, label
            
            @tf.function
            def _load_data_tf(ytid):
                log_mfb, label = tf.py_function(_load_data, [ytid], [tf.float32, tf.int32])
                log_mfb.set_shape(_log_mfb_shape)
                label.set_shape(())
                return (
                    {
                        'ytid':ytid,
                        'log_mfb': log_mfb
                    },
                    label
                )
            self._tf_dataloader = _load_data_tf
            return self._tf_dataloader
    
    def get_shuffled_tf_dataset(self, ytids=None):
        if ytids is None:
            ytids = self.ytids
        return tf.data.Dataset.from_tensor_slices(ytids).shuffle(len(ytids)).map(self.tf_dataloader)
    
    def get_unshuffled_tf_dataset(self, ytids=None):
        if ytids is None:
            ytids = self.ytids
        return tf.data.Dataset.from_tensor_slices(ytids).map(self.tf_dataloader)


class RandomWindow:
    def __init__(self, window_size, features, pad_value):
        self.window_size = window_size
        self.features = features
        self.pad_value = pad_value
        
    def __call__(self, data, label):
        data_out = data.copy()
        
        data_len = tf.shape(data_out[self.features[0]])[0]
        
        pad_size = tf.maximum(0, self.window_size - data_len)
        for feature in self.features:
            data_out[feature] = tf.pad(data_out[feature], [[pad_size,0]]+[[0, 0]]*(data[feature].shape.rank-1),
                                       constant_values=self.pad_value)
        
        data_len = data_len + pad_size
        
        start_idx = tf.random.uniform((), 0, data_len-self.window_size+1, dtype=tf.int32)
        end_idx = start_idx + self.window_size
        
        window = tf.stack([start_idx, end_idx], axis=0)
        data_out['window'] = window
        
        for feature in self.features:
            data_out[feature] = data_out[feature][start_idx:end_idx]
        return data_out, label
    
    

