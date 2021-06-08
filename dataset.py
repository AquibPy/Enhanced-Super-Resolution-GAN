import os
import config
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torchvision.utils import save_image

class ImageData(Dataset):
    def __init__(self,root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.files_names = os.listdir(self.root_dir)
    
    def __len__(self):
        return len(self.files_names)
    
    def __getitem__(self, index):
        img = self.files_names[index]
        img_path = os.path.join(self.root_dir,img)
        image = np.array(Image.open(img_path))
        image = config.both_transforms(image=image)['image']
        high_res = config.highres_transform(image=image)['image']
        low_res = config.lowres_transform(image = image)['image']
        return low_res,high_res

if __name__=='__main__':
    dataset = ImageData(root_dir=config.ROOT_DIR)
    loader = DataLoader(dataset,batch_size=1,shuffle=True)
    for low_res,high_res in loader:
        print(low_res.shape)
        print(high_res.shape)
        save_image(low_res,'low.png')
        save_image(high_res,'high.png')
        break
