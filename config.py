from albumentations.augmentations.transforms import Resize
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

ROOT_DIR = "E:\\Aquib\\MCA\\Python\\SRGAN\\DIV2K_train_HR"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN = "ESRGAN_generator.pth"
CHECKPOINT_DISC = "ESRGAN_discriminator.pth"
LAMBDA_GP = 10
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10000
BATCH_SIZE = 16
NUM_WORKERS = 4
HIGH_RES = 128
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 3

highres_transform = A.Compose(
    [
        A.Normalize(mean=[0,0,0],std=[1,1,1,]),
        ToTensorV2()
    ]
)

lowres_transform = A.Compose(
    [
        A.Resize(width=LOW_RES,height=LOW_RES,interpolation= Image.BICUBIC),
        A.Normalize(mean=[0,0,0],std=[1,1,1]),
        ToTensorV2()
    ]
)
both_transforms = A.Compose(
    [
        A.RandomCrop(width=HIGH_RES, height=HIGH_RES),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
)

test_transform = A.Compose(
    [
        # A.Resize(512,512),
        A.Normalize(mean=[0,0,0],std=[1,1,1]),
        ToTensorV2()
    ]
)
custom_transform = A.Compose(
    [
        A.Resize(512,512),
        A.Normalize(mean=[0,0,0],std=[1,1,1]),
        ToTensorV2()
    ]
)