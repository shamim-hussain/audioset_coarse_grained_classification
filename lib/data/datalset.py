import pandas as pd
import numpy as np
import json
import zipfile
from tqdm import tqdm

class Dataset:
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
    
    
    

