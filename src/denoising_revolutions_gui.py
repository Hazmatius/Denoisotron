import os

import matplotlib.pyplot as plt
import scipy.signal as signal
import skimage
import torch
from matplotlib.widgets import Slider, TextBox
from skimage import io

from src.modules import Denoiser


def get_data(directory, point_name, channel_name):
    img = io.imread(os.path.join(directory, point_name, 'TIFs', channel_name+'.tif')).astype(int)
    img = torch.tensor(img, dtype=torch.uint8).unsqueeze(0).unsqueeze(0)
    return img


def lambda_textbox_update(val):
    lam = float(val)
    global current_lambda
    if lam != current_lambda:
        current_lambda = lam
        print('Denoising with a value of ' + val + '...')

        # x_lam = torch.zeros(target_img.shape) + lam
        x_lam = blur_tensor(contaminating_img, 4) * lam

        denoised_img = denoise_tensor(target_img, x_lam, 'vanilla')

        l2.set_data(denoised_img[0, 0, :, :].detach().cpu())
        # l2.set_clim(vmin=torch.min(denoised_img), vmax=torch.max(denoised_img))
        # l2.set_clim(vmax=dispcap_slider.val)
        fig1.canvas.draw_idle()
    else:
        pass


def dispcap_slider_update(val):
    l2.set_clim(vmax=dispcap_slider.val)
    l1.set_clim(vmax=dispcap_slider.val)
    fig1.canvas.draw_idle()


def blur_tensor(tens, blur):
    return torch.tensor(skimage.filters.gaussian(tens[0, 0, :, :].numpy(), blur)).unsqueeze(0).unsqueeze(0)


def get_lam_est_distribution(data):
    blurred_data = blur_tensor(data, 50)
    medfilt_data = signal.medfilt2d(blurred_data.numpy(), kernel_size=49)
    # medfilt_data = blurred_data
    return medfilt_data.flatten()


def denoise_tensor(tens, lam, kind, **kwargs):
    if kind == 'vanilla':
        denoised_img = denoiser.denoise(tens, lam)
    elif kind == 'intensity_filter':
        c_img = denoiser.denoise(tens, lam)
        denoised_img = torch.tensor(tens)
        denoised_img[c_img < kwargs['threshold']] = 0
    elif kind == 'ratiometric_filter':
        c_img = denoiser.denoise(tens, lam)
        ratio = tens/(tens-c_img)
        denoised_img = torch.tensor(tens)
        denoised_img[ratio < 1.5] = 0
    else:
        print('Invalid denoising kind')
    return denoised_img


data_path = '/Users/raymondbaranski/Downloads/'
model_path = '/Users/raymondbaranski/GitHub/Denoisotron/models/'
point = '16_31773_4_8'
contaminating_channel = 'Ecad'
target_channel = 'FoxP3'
current_lambda = 0

contaminating_img = get_data(data_path, point, contaminating_channel).float()
target_img = get_data(data_path, point, target_channel).float()

denoiser = Denoiser.load_model(model_path, 'denoiser')

# print(denoiser)

fig1 = plt.figure()
plt.subplots_adjust(bottom=0.25)

# x = torch.tensor(input_img)
# for i in range(10):
#     print(i)
#     y = denoise_tensor(x, 0.1)
#     x[y < 0.5] = 0
#
# input_img = x

ax1 = plt.subplot(1, 2, 1)
plt.title('Original Target Channel [' + target_channel + ']')
l1 = plt.imshow(target_img[0, 0, :, :].cpu())
plt.axis('off')

ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
plt.title('Cleaned Target Channel [' + target_channel + ']')
l2 = plt.imshow(target_img[0, 0, :, :].cpu())
plt.axis('off')

axcolor = 'lightgoldenrodyellow'
lambda_axes = plt.axes([0.75, 0.15, 0.15, 0.03], facecolor=axcolor)
dispcap_axes = plt.axes([0.65, 0.1, 0.25, 0.03], facecolor=axcolor)

lambda_textbox = TextBox(lambda_axes, 'Evaluate', '0')
dispcap_slider = Slider(dispcap_axes, 'Display Cap', 0, 10, valinit=10)


lambda_textbox.on_submit(lambda_textbox_update)
dispcap_slider.on_changed(dispcap_slider_update)

# fig2 = plt.figure()
# local_lambda_estimates = get_lam_est_distribution(input_img)
# print(np.mean(input_img.numpy().flatten()))
# n, bins, patches = plt.hist(local_lambda_estimates, 100, facecolor='blue')
plt.show()