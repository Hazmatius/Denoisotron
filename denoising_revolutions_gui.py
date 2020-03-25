import torch
from modules import Denoiser
from skimage import io
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons

# we need a way to scroll through data by point and channel


def get_data(directory, point_name, channel_name):
    img = io.imread(os.path.join(directory, point_name, 'TIFs', channel_name+'.tif')).astype(int)
    img = torch.tensor(img, dtype=torch.uint8).unsqueeze(0).unsqueeze(0)
    return img

data_path = '/home/hazmat/GitHub/Denoisotron/data/'
model_path = '/home/hazmat/GitHub/Denoisotron/models/'
point = 'Point15'
channel = 'CD45'

input_img = get_data(data_path, point, channel)
input_img = input_img.float().cuda()

denoiser = Denoiser.load_model(model_path, 'denoiser')
denoiser.cuda()

# print(denoiser)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
l = plt.imshow(input_img[0, 0, :, :].cpu())

axcolor = 'lightgoldenrodyellow'
axlam = plt.axes([0.15, 0.1, 0.75, 0.03], facecolor=axcolor)
axcap = plt.axes([0.15, 0.15, 0.75, 0.03], facecolor=axcolor)

slam = Slider(axlam, 'Lambda', 0, 5.0)
scap = Slider(axcap, 'Display Cap', 0, 10)

def update(val):
    lam = slam.val
    x_lam = torch.zeros(input_img.shape) + lam
    denoised_img = denoiser.denoise(input_img, x_lam)
    # img.autoscale()
    l.set_data(denoised_img[0, 0, :, :].detach().cpu())
    l.set_clim(vmax=scap.val)
    # l.set_ydata(amp*np.sin(2*np.pi*20*t))
    fig.canvas.draw_idle()


slam.on_changed(update)
scap.on_changed(update)

plt.show()