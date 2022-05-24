import os

img_list = ""
bg_upsampler = None
model_name = ""
arch = 'clean'
channel_multiplier = 2
model_name = 'GFPGANv1.3'
upscale = 2

model_path = "GFPGAN/experiments/pretrained_models/GFPGANv1.pth"
os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth -P pretrained_models")

def predictSrgan():
        print("GFPGANv1.pth descargado")
