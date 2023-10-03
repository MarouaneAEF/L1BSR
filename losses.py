
import random
import utils
import torch
import torch.nn.functional as F


def csr_loss_batch(im_0, im_1, csr_net):
    
    
    device = im_0.device
    i = random.choice([0, 1, 2, 3])
    j = random.choice([0, 1, 2, 3])
    
    # the normalizing input image according to its mean and standard deviation
    imT_0 = utils.contrast_normalization([im_0])[0]
    imT_1 = utils.contrast_normalization([im_1])[0]
    
    # Randomly select whether to use imT_0 or imT_1 for concatenation
    use_imT_0 = random.choice([True, False])
    concat_j = imT_0[:, j:j+1]
    if not use_imT_0:
        concat_j = imT_1[:, j:j+1]
    
    concat = torch.cat((imT_1[:, i:i+1].expand(-1, 4, -1, -1), concat_j.expand(-1,4,-1,-1)), 1)
    flow_1_3 = csr_net(concat.to(device)) 
    
    im_0_i = imT_0[:, i:i+1]
    im_1_i = imT_1[:, i:i+1]
    im1_i_warp = utils.warp(im_1_i, flow_1_3,  mode="bicubic", padding_mode="reflection")
    resized_warped_im_1_i = F.interpolate(im1_i_warp, size=(im_0_i.size(2), im_0_i.size(3)), mode='bicubic')
    concat = torch.cat((resized_warped_im_1_i.expand(-1,4,-1,-1), im_0_i.expand(-1,4,-1,-1)), dim=1)
    composition = csr_net(concat.to(device))
    warped_01 = utils.warp(im_0_i, composition,  mode="bicubic", padding_mode="reflection")
    
    resized_warped_01 = F.interpolate(warped_01, size=(im_1_i.size(2), im_1_i.size(3)), mode='bicubic', align_corners=False)
    l1_norm = torch.abs(resized_warped_01 - im_1_i).mean()
    
    return l1_norm
    



def rec_loss_batch(im_0, im_1, rec_net, csr_net):
    
    device = im_0.device
    # bands registration
    imT_0 = utils.contrast_normalization([im_0])[0]
    imT_1 = utils.contrast_normalization([im_1])[0]
    
    concat = torch.cat((imT_0[:, 1:2].expand(-1, 4, -1, -1), imT_1), 1)
    flow = csr_net(concat.to(device))
    im_hat_0 = rec_net(imT_0.to(device))
    im_r = utils.warp(
        im_hat_0, flow, mode="bicubic", padding_mode="reflection"
    )
    resized_sub_imr = im_r#torch.nn.functional.interpolate(im_r, size=(im_1.size(2), im_1.size(3)), mode='bicubic', align_corners=True)
    l1_norm = torch.abs(resized_sub_imr - im_1).mean()
    return l1_norm

