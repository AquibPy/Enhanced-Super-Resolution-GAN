  
import torch
import torch.optim as optim
from utils import load_checkpoint
import numpy as np
import secrets
from PIL import Image
import config
from model import Generator
from torchvision.utils import save_image

gen = Generator(in_channels=config.IMG_CHANNELS).to(config.DEVICE)
opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.999))
load_checkpoint(config.CHECKPOINT_GEN,gen,opt_gen,config.LEARNING_RATE)

def generate_image(image_path:str):
    input_image = np.array(Image.open(image_path))
    H,W,C = input_image.shape
    if (H or W) >=512:
        img = config.custom_transform(image=input_image)['image']
    else:
        img = config.test_transform(image=input_image)['image']
    # img = config.test_transform(image=input_image)['image']
    img = torch.unsqueeze(img,0).to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        upscaled_img = gen(img)
        hashed = secrets.token_hex(8)
        filename = f"{hashed}.png"
        save_image(upscaled_img,f"images/output/{filename}")
    return filename
