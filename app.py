from tkinter import Tk
from customtkinter import (
    CTkLabel,
    CTkButton,
    CTkEntry
)

import torch
from torch import autocast, cuda
from diffusers import StableDiffusionPipeline

from PIL.ImageTk import PhotoImage as Image
from uuid import uuid1

# https://developer.nvidia.com/cuda-downloads?target_os=Windows
device = torch.device('cuda' if cuda.is_available() else 'cpu')
model = 'CompVis/stable-diffusion-v1-4' 

pipeline = StableDiffusionPipeline.from_pretrained(model, revision="fp16", torch_dtype=torch.float16) 
pipeline.to(device) 

window = Tk()
window.title('Image Generation')
window.geometry('532x632')

label = CTkLabel(window, 512, 512, text='')
label.place(x=10, y=110)

prompts = CTkEntry(window, 40, 512, font=('Arial', 20), text_color="black", fg_color="white")
prompts.place(x=10, y=10)

def generate(): 
    with autocast(device_type=device): 
        image = pipeline(prompts.get(), guidance_scale=8.5).images[0]
    

    image.save(f'assests/{uuid1()}.png')
    img = Image(image)
    label.configure(image=img)

generation = CTkButton(window, height=40, width=120, text="Generate", font=("Arial", 20), text_color="white", fg_color="blue", command=generate) 
generation.place(x=206, y=60) 

window.mainloop()
