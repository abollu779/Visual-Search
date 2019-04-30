from torch.utils.data import Dataset, DataLoader
from refer import REFER
from pprint import pprint
import random
import torch
import json

class RefDataset(Dataset):
    def __init__(self):
        self.refer = REFER(dataset='refcoco+', splitBy='unc')
        self.ref_ids = self.refer.getRefIds()

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self,i):

        ref_id = self.ref_ids[i]
        ref = self.refer.loadRefs(ref_id)[0]

        for sent in ref['sentences']:
            s = sent['raw']
            sid = sent['sent_id']

        return s, sid
               

if __name__ == '__main__':

    dataset = RefDataset()

    d = dict()
    for s, sid in dataset:
        d[str(sid)] = s

    with open('sentences.json', 'w+') as outfile:  
        json.dump(d, outfile)
