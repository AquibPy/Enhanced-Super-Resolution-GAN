import torch
import config
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from loss import VGGLoss
from dataset import ImageData
from model import Generator,Discriminator,initialize_weights
from utils import save_checkpoint,load_checkpoint,gradient_penalty

def train_fn(loader,disc,gen,opt_disc,opt_gen,l1,vgg_loss):
    loop = tqdm(loader,leave=True)
    for idx, (low_res,high_res) in enumerate(loop):
        low_res = low_res.to(config.DEVICE)
        high_res = high_res.to(config.DEVICE)

        fake = gen(low_res)
        disc_real = disc(high_res)
        disc_fake = disc(fake.detach())
        gp = gradient_penalty(disc, high_res, fake, device=config.DEVICE)
        loss_disc = (
                -(torch.mean(disc_real) - torch.mean(disc_fake))
                + config.LAMBDA_GP * gp
            )

        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()
        
        disc_fake = disc(fake)
        l1_loss = 1e-2 * l1(fake, high_res)
        adversarial_loss = 5e-3 * -torch.mean(disc(fake))
        loss_vgg = vgg_loss(fake,high_res)
        loss_gen = l1_loss+ adversarial_loss + loss_vgg

        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

def main():
    torch.cuda.empty_cache()
    dataset = ImageData(root_dir=config.ROOT_DIR)
    loader = DataLoader(dataset,batch_size=config.BATCH_SIZE,shuffle=True,pin_memory=True,num_workers=config.NUM_WORKERS)
    gen = Generator(in_channels=config.IMG_CHANNELS).to(config.DEVICE)
    initialize_weights(gen)
    disc = Discriminator(in_channels=config.IMG_CHANNELS).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.999))
    l1 = nn.L1Loss()
    vgg_loss = VGGLoss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN,gen,opt_gen,config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)
    
    for epoch in range(config.NUM_EPOCHS):
        train_fn(loader,disc,gen,opt_disc,opt_gen,l1,vgg_loss)

        if config.SAVE_MODEL:
            print('Model Saved for Epoch {}'.format(epoch))
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

if __name__=='__main__':
    main()