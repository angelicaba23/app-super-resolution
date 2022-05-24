import os
#from gfpgan import GFPGANer

img_list = ""
bg_upsampler = None
model_name = ""
arch = 'clean'
channel_multiplier = 2
model_name = 'GFPGANv1.3'
upscale = 2

model_path = "GFPGAN/experiments/pretrained_models/GFPGANv1.pth"
os.system("curl -LJO https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth")

print(os.system("ls"))

def predictSrgan():
        print("GFPGANv1.pth descargado")

"""
restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)
 """       