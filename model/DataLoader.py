from torch.utils.data import Dataset, DataLoader
from refer import REFER
from pprint import pprint
import random
import torch


class RefDataset(Dataset):
    def __init__(self,split):
        self.refer = REFER(dataset='refcoco+', splitBy='unc')
        self.ref_ids = self.refer.getRefIds(split=split)
        print('Found {} referred objects in {} split.'.format(len(self.ref_ids),split))

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self,i):
        ref_id = self.ref_ids[i]
        ref = self.refer.loadRefs(ref_id)[0]

        bound_box = self.refer.getRefBox(ref_id)
        whole_file_name = ref['file_name']
        file_name = whole_file_name[:whole_file_name.rfind("_")]+".jpg"
        ref_expr = random.choice(ref['sentences'])['raw']

        return file_name, ref_expr, bound_box
        

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = RefDataset("train")
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=(8 if device == "cuda" else 0))

    for idx, (f,r,b) in enumerate(loader):
        print(idx,f,r,b)
