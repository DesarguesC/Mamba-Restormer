# Restormer: Efficient Transformer for High-Resolution Image Restoration
# Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
# https://arxiv.org/abs/2111.09881

import numpy as np
import cv2, torch, os
from os.path import join as oj
from skimage.metrics import structural_similarity as ssim



datasets = ['Test100', 'Rain100H', 'Rain100L', 'Test2800', 'Test1200']
num_set = len(datasets)

psnr_alldatasets = 0
ssim_alldatasets = 0

# tic
# delete(gcp('nocreate'))
# parpool('local',20);

def compute_ssim(original_img, edited_img, multichannel: bool = True, channel_axis=2) -> float:
    """
        calculate SSIM score between original and edited images
        """
    if not isinstance(original_img, list):
        return ssim(original_img, edited_img, multichannel=multichannel, channel_axis=channel_axis)
    else:
        assert len(original_img) == len(
            edited_img), f'len(original_img) = {len(original_img)}, len(edited_img) = {len(edited_img)}'
        SSIM = [float(ssim(original_img[i], edited_img[i], multichannel=multichannel, channel_axis=channel_axis)) for i
                in range(len(original_img))]
        return np.mean(SSIM)
    return

def compute_psnr(original_img, edited_img, max_pixel: int = 255) -> float:
    """
    计算两幅图像之间的PSNR值。
    """
    if not isinstance(original_img, list):
        mse = np.mean((original_img - edited_img) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(max_pixel / np.sqrt(mse))
    else:
        assert len(original_img) == len(edited_img), f'len(original_img) = {len(original_img)}, len(edited_img) = {len(edited_img)}'
        mse = [np.mean((original_img[i] - edited_img[i]) ** 2) for i in range(len(original_img))]
        tot = np.mean(mse)
        if tot == 0:
            return float('inf')
        else:
            return 20 * np.log10(max_pixel / np.sqrt(tot))





for idx_set in range(1,num_set+1):
    file_path = f'./results/{datasets[idx_set]}/'
    gt_path = f'./Datasets/test/{datasets[idx_set]}/target/'
    path_list = [oj(file_path, file_name) for file_name in os.listdir(file_path) if file_name.endswith('jpg') or file_name.endswidth('png')]
    gt_list = [oj(gt_path, file_name) for file_name in os.listdir(gt_path) if file_name.endswith('jpg') or file_name.endswidth('png')]
    img_num = len(path_list)

    total_psnr = 0
    total_ssim = 0

    if img_num > 0 :
        for j in range(1, img_num+1):
           image_name = path_list[j].name;
           gt_name = gt_list[j].name;
           input = cv2.imread(oj(file_path,image_name))
           gt = cv2.imread(oj(gt_path, gt_name));
           ssim_val = compute_ssim(input, gt)
           psnr_val = compute_psnr(input, gt)
           total_ssim = total_ssim + ssim_val
           total_psnr = total_psnr + psnr_val

    qm_psnr = total_psnr / img_num;
    qm_ssim = total_ssim / img_num;

    print('For %s dataset PSNR: %f SSIM: %f\n', datasets[idx_set], qm_psnr, qm_ssim);

    psnr_alldatasets = psnr_alldatasets + qm_psnr
    ssim_alldatasets = ssim_alldatasets + qm_ssim
    


print('For all datasets PSNR: %f SSIM: %f\n', psnr_alldatasets/num_set, ssim_alldatasets/num_set);

# delete(gcp('nocreate'))
# toc



