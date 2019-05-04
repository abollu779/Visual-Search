from torch.utils.data import Dataset, DataLoader
from refer import REFER
from pprint import pprint
import random
import torch
import os
import numpy as np
import time

class RefDataset(Dataset):
    def __init__(self,split):
        self.refer = REFER(dataset='refcoco+', splitBy='unc')
        self.ref_ids = self.refer.getRefIds(split=split)

        self.image_embeds = np.load(os.path.join("data","embeddings","FINALImageEmbeddings.npy"))
        self.image_ids = list(np.load(os.path.join("data","embeddings","FINALImageIDs.npy")))
        before_text_embeds = time.time()
        self.text_embeds = np.concatenate((np.load(os.path.join("data","embeddings","FINALTextEmbeddings1of2.npy")), np.load(os.path.join("data","embeddings","FINALTextEmbeddings2of2.npy"))),axis=0)
        after_text_embeds = time.time()
        print("Text Embedding Time: ", after_text_embeds - before_text_embeds)
        assert(len(self.text_embeds)==141564)
        assert(self.text_embeds[0].shape[1]==3072)
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
        bound_box = torch.Tensor(self.refer.getRefBox(ref_id))
        bound_box[0] /= width
        bound_box[1] /= height
        bound_box[2] /= width
        bound_box[3] /= height
        #bound_box = bound_box.unsqueeze(dim=0)

        #whole_file_name = ref['file_name']
        #file_name = whole_file_name[:whole_file_name.rfind("_")]+".jpg"

        sent = random.choice(ref['sentences'])
        ref_expr = sent['raw']
        text_id = sent['sent_id']

        text_idx = text_id
        text_embed = torch.from_numpy(self.text_embeds[text_idx])

        return image_embed, text_embed, bound_box
        

def collate_fn(seq_list):
    im_em,text_em,bb = zip(*seq_list)
    lens = [len(seq) for seq in text_em]
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)

    im_em = [im_em[i] for i in seq_order]
    im_em = torch.tensor(np.stack(im_em, axis=0)) # batch_size x 1024 x 13 x 13

    text_em = [text_em[i] for i in seq_order] # batch_size x seq_len x 3072 (LIST)

    bb = [bb[i] for i in seq_order]
    bb = torch.stack(bb) # batch_size x 4
    return im_em,text_em,bb


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = RefDataset("train")
    #dataset = RefDataset("val")
    #dataset = RefDataset("test")
    loader = DataLoader(dataset, collate_fn=collate_fn, batch_size=128, shuffle=False, num_workers=(8 if device == "cuda" else 0))

    i = 0
    for idx, (f,r,b) in enumerate(loader):
        i+=1
    print(i)
