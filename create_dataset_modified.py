import argparse
import os
import torch.utils.data
import yaml
import model
import utils
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
from receptive_cal import *


parser = argparse.ArgumentParser(description='Apply the trained model to create a dataset')
parser.add_argument('--checkpoint', default=None, type=str, help='checkpoint model to use')
parser.add_argument('--Generator', default=None, type=str, help='Generator model to use')
parser.add_argument('--Discriminator', default=None, type=str, help='Discriminator model to use')
parser.add_argument('--artifacts', default='', type=str, help='selecting different artifacts type')
parser.add_argument('--name', default='', type=str, help='additional string added to folder path')
parser.add_argument('--dataset', default='df2k', type=str, help='selecting different datasets')
parser.add_argument('--track', default='train', type=str, help='selecting train or valid track')
parser.add_argument('--num_res_blocks', default=8, type=int, help='number of ResNet blocks')
parser.add_argument('--cleanup_factor', default=2, type=int, help='downscaling factor for image cleanup')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[4], help='super resolution upscale factor')
opt = parser.parse_args()

# define input and target directories
with open('paths.yml', 'r') as stream:
    PATHS = yaml.load(stream)

if opt.dataset == 'aim2019':
    path_sdsr = PATHS['datasets']['aim2019'] + '/generated/sdsr/'
    path_tdsr = PATHS['datasets']['aim2019'] + '/generated/tdsr/'
    input_source_dir = PATHS['aim2019']['tdsr']['source']
    input_target_dir = PATHS['aim2019']['tdsr']['target']
    input_target_dir = '/media/4T/Dizzy/AIM/AIM_datasets/DIV2K_train_HR/'
    source_files = [os.path.join(input_source_dir, x) for x in os.listdir(input_source_dir) if utils.is_image_file(x)]
    target_files = [os.path.join(input_target_dir, x) for x in os.listdir(input_target_dir) if utils.is_image_file(x)]
else:
    path_sdsr = PATHS['datasets'][opt.dataset] + '/generated/' + opt.artifacts + '/' + opt.track + opt.name + '_sdsr/'
    path_tdsr = PATHS['datasets'][opt.dataset] + '/generated/' + opt.artifacts + '/' + opt.track + opt.name + '_tdsr/'
    input_source_dir = PATHS[opt.dataset][opt.artifacts]['hr'][opt.track]
    input_target_dir = None
    source_files = [os.path.join(input_source_dir, x) for x in os.listdir(input_source_dir) if utils.is_image_file(x)]
    target_files = []

sdsr_hr_dir = path_sdsr + 'HR'
sdsr_lr_dir = path_sdsr + 'LR'
tdsr_hr_dir = path_tdsr + 'HR'
tdsr_lr_dir = '/media/4T/Dizzy/real-world-sr/results/' + opt.name + 'LR'
tdsr_lr_img_dir = os.path.join(tdsr_lr_dir, 'imgs')
tdsr_lr_ddm_dir = os.path.join(tdsr_lr_dir, 'ddm')

if not os.path.exists(sdsr_hr_dir):
    os.makedirs(sdsr_hr_dir)
if not os.path.exists(sdsr_lr_dir):
    os.makedirs(sdsr_lr_dir)
if not os.path.exists(tdsr_hr_dir):
    os.makedirs(tdsr_hr_dir)
if not os.path.exists(tdsr_lr_dir):
    os.makedirs(tdsr_lr_dir)
if not os.path.exists(tdsr_lr_img_dir):
    os.makedirs(tdsr_lr_img_dir)
if not os.path.exists(tdsr_lr_ddm_dir):
    os.makedirs(tdsr_lr_ddm_dir)

# prepare neural networks
if opt.Generator == 'DSGAN':
    model_g = model.Generator(n_res_blocks=opt.num_res_blocks)
elif opt.Generator == 'DeResnet':
    model_g = model.De_resnet(n_res_blocks=opt.num_res_blocks)

if opt.Discriminator == 'FSD':
    # model_d = model.Discriminator_wavelet(patchgan=False)
    model_d = model.Discriminator_Gau(patchgan=False, gaussian=True)
    convnet = [[5, 1, 2], [5, 1, 2], [5, 1, 2], [5, 1, 2]]
elif opt.Discriminator == 'n_layer_D_s1':
    model_d = model.Discriminator_wavelet(cs='cat', patchgan='s1')
    convnet = [[4, 1, 1], [4, 1, 1], [4, 1, 1], [4, 1, 1]]
elif opt.Discriminator == 'n_layer_D_s2':
    model_d = model.Discriminator_wavelet(cs='cat', patchgan='s2')
    convnet = [[4, 2, 1], [4, 2, 1], [4, 1, 1], [4, 1, 1]]
model_g = model_g.eval()
model_d = model_d.eval()
print('# generator parameters:', sum(param.numel() for param in model_g.parameters()))
if torch.cuda.is_available():
    model_g = model_g.cuda()
    model_d = model_d.cuda()

# load/initialize parameters
if opt.checkpoint is not None:
    checkpoint = torch.load(opt.checkpoint)
    epoch = checkpoint['epoch']
    model_g.load_state_dict(checkpoint['model_g_state_dict'])
    model_d.load_state_dict(checkpoint['models_d_state_dict'])
    print('Using model at epoch %d' % epoch)
else:
    print('Use --checkpoint to define the model parameters used')
    exit()

# generate the noisy images
smallest_size = 1000000000
with torch.no_grad():
    for file in tqdm(target_files, desc='Generating images from target'):
        # load HR image
        input_img = Image.open(file)
        input_img = TF.to_tensor(input_img)

        # # Save input_img as HR image for TDSR
        # path = os.path.join(tdsr_hr_dir, os.path.basename(file))
        # TF.to_pil_image(input_img).save(path, 'PNG')
        # Apply model to input_img
        if torch.cuda.is_available():
            input_img = input_img.unsqueeze(0).cuda()
        input_noisy_img = model_g(input_img)

        D_out = model_d(input_noisy_img).cpu().detach().numpy()

        realorfake_shape = (input_noisy_img.shape[0], 1,
                            input_noisy_img.shape[2], input_noisy_img.shape[3])
        realorfake_shape = torch.zeros(realorfake_shape)
        currentLayer_h, currentLayer_w = receptive_cal(realorfake_shape.shape[2], convnet), \
                                         receptive_cal(realorfake_shape.shape[3], convnet)
        realorfake = getWeights(D_out, realorfake_shape, currentLayer_h, currentLayer_w)
        # print(realorfake.min(), realorfake.max())
        # Save input_noisy_img as HR image for TDSR
        input_noisy_img = input_noisy_img.squeeze(0).cpu()
        path = os.path.join(tdsr_lr_img_dir, os.path.basename(file))
        TF.to_pil_image(input_noisy_img).save(path, 'PNG')
        np.save(os.path.join(tdsr_lr_ddm_dir, os.path.basename(file).split('.')[0]), realorfake)

