import os
from gfpgan import GFPGANer
#from basicsr import *

img_list = ""
bg_upsampler = None
model_name = ""
arch = 'clean'
channel_multiplier = 2
model_name = 'GFPGANv1.3'
upscale = 2

model_path = "GFPGANv1.pth"
os.system("curl -LJO https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth")

#print(os.system("ls pretrained_models "))
os.system("BASICSR_EXT=True pip install basicsr==1.3.5")
os.system("python setup.py develop")
print(os.system("pip list"))

def predictSrgan():
        print(os.system('-d "GFPGANv1.pth'))
        print("GFPGANv1.pth descargado")
        

restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)

