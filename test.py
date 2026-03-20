"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import numpy as np
import time
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from sklearn.metrics import mean_squared_error

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    # Initialize variables to accumulate metrics
    ssim_scores = []
    psnr_scores = []
    mse_scores = []
    inference_times = []

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader

        start_time = time.time()
        model.test()           # run inference
        end_time = time.time()
        
        # Calculate inference time
        inference_time = end_time - start_time
        inference_times.append(inference_time)

        visuals = model.get_current_visuals()  # get image results   visuals['real_A', 'fake_B', 'real_B']
        img_path = model.get_image_paths()     # get image paths

        # Convert tensors to NumPy arrays for metric calculation
        real_B = visuals['real_B'].cpu().numpy()[0].transpose(1, 2, 0)  # [H, W, C]
        fake_B = visuals['fake_B'].cpu().numpy()[0].transpose(1, 2, 0)  # [H, W, C]

        # print(real_B.min(), real_B.max(), fake_B.min(), fake_B.max())
        # print(visuals['real_B'].shape)
        # Normalize values to [0, 1] (assuming the values are not already normalized)
        real_B = (real_B - real_B.min()) / (real_B.max() - real_B.min())
        fake_B = (fake_B - fake_B.min()) / (fake_B.max() - fake_B.min())

        # Calculate SSIM and PSNR with proper data_range
        ssim = compare_ssim(real_B, fake_B, data_range=1.0, win_size=3)
        psnr = compare_psnr(real_B, fake_B, data_range=1.0)
        mse = mean_squared_error(real_B.flatten(), fake_B.flatten())

        # Append metrics to lists
        ssim_scores.append(ssim)
        psnr_scores.append(psnr)
        mse_scores.append(mse)

        if i % 1 == 0:  # save images to an HTML file
            print(f'Processing ({i:04d})-th image...SSIM: {ssim:.4f}, PSNR: {psnr:.4f}, MSE: {mse:.4e}')
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)

    webpage.save()  # save the HTML

    # Calculate average metrics
    avg_ssim = sum(ssim_scores) / len(ssim_scores)
    avg_psnr = sum(psnr_scores) / len(psnr_scores)
    avg_mse = sum(mse_scores) / len(mse_scores)
    avg_inference_time = sum(inference_times) / len(inference_times)

    # Print average metrics
    print(f'Average SSIM: {avg_ssim:.4f}')
    print(f'Average PSNR: {avg_psnr:.4f}')
    print(f'Average MSE: {avg_mse:.4e}')
    print(f'Average Inference Time: {avg_inference_time:.4f} seconds')


    # Function to save a list to a text file
    def save_list_to_txt(filename, data):
        with open(filename, 'w') as f:
            for item in data:
                f.write(f"{item}\n")

    # Save each list into separate text files
    save_list_to_txt("ssim_scores.txt", ssim_scores)
    save_list_to_txt("psnr_scores.txt", psnr_scores)
    save_list_to_txt("mse_scores.txt", mse_scores)