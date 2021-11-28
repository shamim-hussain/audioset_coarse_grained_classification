
import pandas as pd
import numpy as np
import json
import zipfile
from tqdm import tqdm


class Dataset:
    def __init__(self, 
                 annotations_path, 
                 dataset_path, 
                 ytids_path,
                 ytid_keys          =  None,
                 class_associations = (('speech',0),('music',1),('noise',2)),
                 transform_fn       =  None,
                 include_ytid       = False):
        self.annotations = pd.read_csv(annotations_path).set_index('ytid')
        self.dataset_path = dataset_path
        with open(ytids_path, 'r') as f:
            ytids_content = json.load(f)
        if ytid_keys is not None:
            self.ytids = sum([ytids_content[key] for key in ytid_keys], [])
        else:
            self.ytids = ytids_content
        self.class_associations = dict(class_associations)
        self.reverse_class_associations = dict((v,k) for k,v in class_associations)
        self.transform_fn = transform_fn
        self.include_ytid = include_ytid
    
    def load_data(self):
        self.data = {}
        with zipfile.ZipFile(self.dataset_path, 'r') as zf:
            for ytid in tqdm(self.ytids.copy(), desc='Loading data'):
                data = {}
                try:
                    with zf.open(self.annotations.loc[ytid,'log_mfb_path']) as f:
                        data['log_mfb'] = np.load(f)
                    
                    data['label'] = self.class_associations[self.annotations.loc
                                                            [ytid,'plausible_superclass']]
                    self.data[ytid] = data
                except KeyError:
                    print(f'Warning: Key {ytid} not found!')
                    self.ytids.remove(ytid)
    
    def __getitem__(self, index):
        if isinstance(index, str):
            ytid = index
        else:
            ytid = self.ytids[index]
        
        data_without_ytid = self.data[ytid]
        if self.include_ytid:
            data = {'ytid': ytid}
            data.update(data_without_ytid)
        else:
            data = data_without_ytid
        
        if self.transform_fn is not None:
            data = self.transform_fn(data)
        return data
    
    def __len__(self):
        return len(self.data)
    
    

class RandomWindowedDataset(Dataset):
    def __init__(self, 
                 window_size, 
                 windowed_features, 
                 pad_value, 
                 include_window=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.windowed_features = windowed_features
        self.pad_value = pad_value
        self.include_window = include_window
        
    def __getitem__(self, index):
        data = super().__getitem__(index)
        data_out = data.copy()
        
        data_len = data_out[self.windowed_features[0]].shape[0]
        
        pad_size = max(0, self.window_size - data_len)
        if pad_size > 0:
            for feature in self.windowed_features:
                data_out[feature] = np.pad(data_out[feature], [[pad_size,0]]+[[0, 0]]*(len(data[feature].shape)-1),
                                        constant_values=self.pad_value)
            data_len = data_len + pad_size
            if self.include_window:
                data_out['window'] = np.array([0, data_len], dtype=np.int32)
        else:
            start_idx = np.random.randint(0, data_len-self.window_size+1, dtype=np.int32)
            end_idx = start_idx + self.window_size
            
            window = np.stack([start_idx, end_idx], axis=0)
            if self.include_window:
                data_out['window'] = window
            
            for feature in self.windowed_features:
                data_out[feature] = data_out[feature][start_idx:end_idx]
        return data_out



class StridedWindowedDataset(Dataset):
    def __init__(self, 
                 window_size, 
                 window_stride, 
                 windowed_features, 
                 pad_value, 
                 include_window=False,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.window_size = window_size
        self.window_stride = window_stride
        self.windowed_features = windowed_features
        self.pad_value = pad_value
        self.include_window = include_window
    
    
    def load_data(self):
        super().load_data()
        
        windows = []
        for ytid, data in self.data.items():
            data_len = data[self.windowed_features[0]].shape[0]
            
            for i in range(0, data_len - self.window_size, self.window_stride):
                windows.append((ytid, i))
            
            windows.append((ytid, max(data_len-self.window_size, 0)))
        
        self.windows = pd.DataFrame(windows, columns=['ytid', 'start'])
    
    def __getitem__(self, index):
        ytid, start_idx = self.windows.iloc[index]
        data = super().__getitem__(ytid)
        data_out = data.copy()
        
        data_len = data_out[self.windowed_features[0]].shape[0]
        
        pad_size = max(0, self.window_size - data_len)
        if pad_size > 0:
            for feature in self.windowed_features:
                data_out[feature] = np.pad(data_out[feature], [[pad_size,0]]+[[0, 0]]*(len(data[feature].shape)-1),
                                        constant_values=self.pad_value)
            if self.include_window:
                data_out['window'] = np.array([0, data_len + pad_size], dtype=np.int32)
        else:
            end_idx = start_idx + self.window_size
            if self.include_window:
                data_out['window'] = np.array([start_idx, end_idx], dtype=np.int32)
            
            for feature in self.windowed_features:
                data_out[feature] = data_out[feature][start_idx:end_idx]
        return data_out

    def __len__(self):
        return len(self.windows)
    

