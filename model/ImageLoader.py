from torch.utils.data import Dataset
from refer import REFER
from pprint import pprint
import random
import torch


class ImageLoader(Dataset):
    def __init__(self):
        self.refer = REFER(dataset='refcoco+', splitBy='unc')
        self.image_ids = list(self.refer.getImgIds())
        print('Found {} images.'.format(len(self.image_ids)))

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self,i):
        image_id = self.image_ids[i]
        image = self.refer.Imgs[image_id]
        return image['file_name'], image['id']

if __name__ == '__main__':
    loader = ImageLoader()
    for image in loader:
        print(image)
