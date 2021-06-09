import torch
import config
import torch.optim as optim
from model import Generator
from torchvision.utils import save_image
from utils import load_checkpoint
from PIL import Image
import numpy as np
import os

DIR = 'test_images\\'
gen = Generator(in_channels=config.IMG_CHANNELS).to(config.DEVICE)
opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.999))
load_checkpoint(config.CHECKPOINT_GEN,gen,opt_gen,config.LEARNING_RATE)

for i in os.listdir(DIR):
    img_path = os.path.join(DIR,i)
    img = np.array(Image.open(img_path))
    img = config.test_transform(image=img)['image']
    img = torch.unsqueeze(img,0).to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        upscaled_img = gen(img)
        save_image(upscaled_img,f"output{i}.png")
    gen.train()