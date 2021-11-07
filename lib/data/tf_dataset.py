
import tensorflow as tf
from .datalset import Dataset


class TFDataset(Dataset):
    _log_mfb_shape = (None, 64)
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