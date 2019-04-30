from torch.utils.data import Dataset, DataLoader
from refer import REFER
from pprint import pprint
import random
import torch
import os
import numpy as np

class RefDataset(Dataset):
    def __init__(self,split):
        self.refer = REFER(dataset='refcoco+', splitBy='unc')
        self.ref_ids = self.refer.getRefIds(split=split)
        self.image_embeds = np.load(os.path.join("data","embeddings","ImageEmbeddings.npy"))
        self.image_ids = list(np.load(os.path.join("data","embeddings","ImageIDs.npy")))
        self.text_embeds = np.load(os.path.join("data","embeddings","TextEmbeddings.npy"))
        self.text_ids = list(np.load(os.path.join("data","embeddings","TextIDs.npy")))
        print('Found {} referred objects in {} split.'.format(len(self.ref_ids),split))

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self,i):

        ref_id = self.ref_ids[i]
        ref = self.refer.loadRefs(ref_id)[0]

        image_id = ref['image_id']
        image = self.refer.Imgs[image_id]
        image_idx = self.image_ids.index(image_id)
        image_embed = self.image_embeds[image_idx,:,:,:]

        height = image['height']
        width = image['width']
        bound_box = self.refer.getRefBox(ref_id)
        bound_box[0] /= width
        bound_box[1] /= height
        bound_box[2] /= width
        bound_box[3] /= height

        #whole_file_name = ref['file_name']
        #file_name = whole_file_name[:whole_file_name.rfind("_")]+".jpg"

        sent = random.choice(ref['sentences'])
        ref_expr = sent['raw']
        text_id = sent['sent_id']

        text_idx = self.text_ids.index(text_id)
        text_embed = self.text_embeds[text_idx,:,:,:]

        return image_embed, text_embed, bound_box
        

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = RefDataset("train")
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=(8 if device == "cuda" else 0))

    for idx, (f,r,b) in enumerate(loader):
        print(f,r,b)
