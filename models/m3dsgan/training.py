from models.eval import (
    calculate_activation_statistics, calculate_frechet_distance)
from models.training import (
    toggle_grad, compute_grad2, compute_bce, compute_L1, compute_L1_sem, update_average)
from torchvision.utils import save_image, make_grid
import os
import torch
from models.training import BaseTrainer
from tqdm import tqdm
import logging
logger_py = logging.getLogger(__name__)
from models.utils import tensor2label, tensor2im
import cv2
import numpy as np

class Trainer(BaseTrainer):
    ''' Trainer object for mm3dsgan.

    Args:
        model (nn.Module): mm3dsgan model
        optimizer (optimizer): generator optimizer object
        optimizer_d (optimizer): discriminator optimizer object
        device (device): pytorch device
        vis_dir (str): visualization directory
        multi_gpu (bool): whether to use multiple GPUs for training
        fid_dict (dict): dicionary with GT statistics for FID
        n_eval_iterations (int): number of eval iterations
        overwrite_visualization (bool): whether to overwrite
            the visualization files
    '''
    def __init__(self, model, optimizer, optimizer_d, optimizer_s, optimizer_ds, device=None,
                 vis_dir=None,
                 multi_gpu=False,
                 n_eval_iterations=10,
                 overwrite_visualization=True, cfg=None, **kwargs):

        self.model = model
        self.optimizer = optimizer
        self.optimizer_s = optimizer_s
        self.optimizer_d = optimizer_d
        self.optimizer_ds = optimizer_ds
        self.device = device
        self.vis_dir = vis_dir
        self.multi_gpu = multi_gpu
        self.cfg = cfg

        self.overwrite_visualization = overwrite_visualization
        self.n_eval_iterations = n_eval_iterations

        self.vis_dict = model.generator.get_vis_dict(self.cfg['training']['batch_size'])

        if multi_gpu:
            self.generator = torch.nn.DataParallel(self.model.generator)
            self.discriminator = torch.nn.DataParallel(
                self.model.discriminator)
            self.stylegenerator = torch.nn.DataParallel(self.model.stylegenerator)
            self.stylediscriminator = torch.nn.DataParallel(self.model.stylediscriminator)
            if self.model.generator_test is not None:
                self.generator_test = torch.nn.DataParallel(
                    self.model.generator_test)
            else:
                self.generator_test = None
        else:
            self.generator = self.model.generator
            self.discriminator = self.model.discriminator
            self.stylegenerator = self.model.stylegenerator
            self.stylediscriminator = self.model.stylediscriminator
            self.generator_test = self.model.generator_test

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data, it=None):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
            it (int): training iteration
        '''
        loss_g, loss_sp, loss_ss = self.train_step_generator(data, it)
        loss_d, reg_d, fake_d, real_d = self.train_step_discriminator(data, it)
        loss_style_g = self.train_step_stylegenerator(data,it)
        loss_style_d, reg_img, d_loss_fake_img, d_loss_real_img = self.train_step_stylediscriminator(data,it)

        return {
            'generator': loss_g,
            'discriminator': loss_d,
            'regularizer': reg_d,
            'loss_sp': loss_sp,
            'loss_ss': loss_ss,
            'loss_style_g':loss_style_g,
            'loss_style_d': loss_style_d,
            'reg_img': reg_img,
            'd_loss_fake_img': d_loss_fake_img,
            'd_loss_real_img': d_loss_real_img
        }

    def eval_step(self):
        ''' Performs a validation step.

        Args:
            data (dict): data dictionary
        '''
        gen = self.model.generator_test
        if gen is None:
            gen = self.model.generator
        gen.eval()

        x_fake = []
        n_iter = self.n_eval_iterations

        for i in tqdm(range(n_iter)):
            with torch.no_grad():
                x_fake.append(gen().cpu()[:, :3])
        x_fake = torch.cat(x_fake, dim=0)
        x_fake.clamp_(0., 1.)
        mu, sigma = calculate_activation_statistics(x_fake)
        fid_score = calculate_frechet_distance(
            mu, sigma, self.fid_dict['m'], self.fid_dict['s'], eps=1e-4)
        eval_dict = {
            'fid_score': fid_score
        }
        return eval_dict

    def train_step_generator(self, data, it=None, z=None):

        generator = self.generator
        discriminator = self.discriminator
        toggle_grad(generator, True)
        toggle_grad(discriminator, False)

        generator.train()
        discriminator.train()

        self.optimizer.zero_grad()

        if self.multi_gpu:
            latents = generator.module.get_vis_dict(self.cfg['training']['batch_size'])
            x_fake_seg = generator(**latents)
        else:
            latents = generator.get_vis_dict(self.cfg['training']['batch_size'])
            x_fake_seg = generator(**latents)

        # seg loss + image loss
        d_fake = discriminator(x_fake_seg)

        gloss = compute_bce(d_fake, 1)
        gloss = gloss

        #same bg loss:
        if self.multi_gpu:
            latent1, latent2, latent3 = generator.module.get_vis_dict_(self.cfg['training']['batch_size'])
            x_fake_seg1 = generator(**latent1)
            x_fake_seg2 = generator(**latent2)
            x_fake_seg3 = generator(**latent3)
        else:
            latent1, latent2, latent3 = generator.get_vis_dict_(self.cfg['training']['batch_size'])
            x_fake_seg1 = generator(**latent1)
            x_fake_seg2 = generator(**latent2)
            x_fake_seg3 = generator(**latent3)

        x_fake_seg1_sum = torch.sum(x_fake_seg1, dim=[2,3])
        x_fake_seg2_sum = torch.sum(x_fake_seg2, dim=[2,3])

        #sploss, for stable training, we use sp_loss after 100000 steps.
        if it  < 100000:
            sp_loss = compute_L1_sem(x_fake_seg1_sum, x_fake_seg2_sum, coeff=0)
        else:
            sp_loss = compute_L1_sem(x_fake_seg1_sum, x_fake_seg2_sum, coeff=0.01)
        gloss += self.cfg['training']['lambda_sp'] * sp_loss

        #ssloss
        ss_loss = compute_L1(x_fake_seg1[:,0,:,:], x_fake_seg3[:,0,:,:])
        gloss += self.cfg['training']['lambda_ss'] * ss_loss

        gloss.backward()
        self.optimizer.step()

        if self.generator_test is not None:
            update_average(self.generator_test, generator, beta=0.999)

        return gloss.item(), sp_loss.item(), ss_loss.item()

    def train_step_stylegenerator(self, data, it):

        stylegenerator = self.stylegenerator
        #generator = self.generator
        stylediscriminator = self.stylediscriminator

        toggle_grad(stylegenerator, True)
        toggle_grad(stylediscriminator, False)
        #toggle_grad(generator, False)

        stylegenerator.train()
        #generator.train()
        stylediscriminator.train()

        self.optimizer_s.zero_grad()

        #get data pair

        """# data initialize
        if self.multi_gpu:
            latents = generator.module.get_vis_dict(self.cfg['training']['batch_size'])
            x_real_img = data.get('image').to(self.device)
            #style_code =torch.randn(self.cfg['training']['batch_size'], 512, device=self.device)

            fake_seg = generator(**latents)
        else:
            latents = generator.get_vis_dict(self.cfg['training']['batch_size'])
            x_real_img = data.get('image').to(self.device)
            #style_code = torch.randn(self.cfg['training']['batch_size'])
            fake_seg = generator(**latents)"""

        x_real_seg = data.get('seg').to(self.device)
        x_real_img = data.get('image').to(self.device)

        fake_img = stylegenerator(x_real_seg, x_real_img).to(self.device)

        # loss function
        d_fake = stylediscriminator(fake_img)

        g1_loss = compute_bce(d_fake, 1)
        g1_loss = g1_loss

        g1_loss.backward()
        self.optimizer_s.step()


        return g1_loss.item()

    def train_step_discriminator(self, data, it=None, z=None):

        generator = self.generator
        discriminator = self.discriminator

        toggle_grad(generator, False)
        toggle_grad(discriminator, True)

        generator.train()
        discriminator.train()

        self.optimizer_d.zero_grad()
        x_real_seg = data.get('seg').to(self.device)

        loss_d_full = 0.

        x_real_seg.requires_grad_()

        d_real = discriminator(x_real_seg)

        #loss
        d_loss_real = compute_bce(d_real, 1)
        loss_d_full += d_loss_real

        reg = 10. * compute_grad2(d_real, x_real_seg).mean()
        loss_d_full += reg

        with torch.no_grad():
            if self.multi_gpu:
                latents = generator.module.get_vis_dict(self.cfg['training']['batch_size'])
                x_fake_seg = generator(**latents)
            else:
                latents = generator.get_vis_dict(self.cfg['training']['batch_size'])
                x_fake_seg = generator(**latents)

        x_fake_seg.requires_grad_()

        d_fake = discriminator(x_fake_seg)

        d_loss_fake = compute_bce(d_fake, 0)

        loss_d_full += d_loss_fake

        loss_d_full.backward()

        self.optimizer_d.step()

        d_loss = (d_loss_fake + d_loss_real)

        return (
            d_loss.item(), reg.item(), d_loss_fake.item(), d_loss_real.item()
        )

    def train_step_stylediscriminator(self,data, it=None):
        stylegenerator= self.stylegenerator
        stylediscriminator= self.stylediscriminator
        #generator = self.generator

        toggle_grad(stylegenerator, False)
        #toggle_grad(generator, False)
        toggle_grad(stylediscriminator, True)

        stylegenerator.train()
        #generator.train()
        stylediscriminator.train()


        self.optimizer_ds.zero_grad()
        x_real_img = data.get('image').to(self.device)
        x_real_seg = data.get('seg').to(self.device)
        loss_d_full =0.

        x_real_img.requires_grad_()
        d_real_img= stylediscriminator(x_real_img)

        d_loss_real_img = compute_bce(d_real_img, 1)
        loss_d_full +=d_loss_real_img

        reg_img = 10 * compute_grad2(d_real_img, x_real_img).mean()
        loss_d_full += reg_img

        x_fake_img = stylegenerator(x_real_seg, x_real_img)
        x_fake_img.requires_grad_()
        d_fake_img = stylediscriminator(x_fake_img)

        d_loss_fake_img = compute_bce(d_fake_img, 0)
        loss_d_full += d_loss_fake_img

        loss_d_full.backward()
        self.optimizer_ds.step()
        d_loss_img = (d_loss_fake_img+d_loss_real_img)

        return(
            d_loss_img.item(), reg_img.item(), d_loss_fake_img.item(), d_loss_real_img.item())

    def visualize(self, it=0, real=None):
        ''' Visualized the data.

        Args:
            it (int): training iteration
        '''
        gen = self.model.generator_test
        gen1 = self.model.stylegenerator
        if gen is None:
            gen = self.model.generator
        gen.eval()
        gen1.eval()

        real_seg = real['seg']
        real_img = real['image']


        with torch.no_grad():
            image_fake_seg = self.generator(**self.vis_dict, mode='val')
            image_fake_segflip = torch.fliplr(image_fake_seg)
            image_fake_seg = image_fake_seg.cpu()
            style_code = torch.randn(self.cfg['training']['batch_size'], 512).cpu()
            #image_fake = self.stylegenerator(image_fake_seg, style_code, None, is_sampling=True, is_latent=True)
            real_seg = real_seg.to(self.device)
            real_img = real_img.to(self.device)
            image_fake = self.stylegenerator(real_seg, real_img)
            image_fake = image_fake.cpu()


        if self.overwrite_visualization:
            out_file_name = 'seg.png'
            out_file_name_flip = 'segflip.png'
            out_file_real_name = 'seg_real.png'
            out_file_real_name_syn = 'syn_real.png'
            out_file_real_img = 'syn_image.png'

        else:
            out_file_name = '%010d_seg.png' % it
            out_file_name_flip = '%010d_segflip.png' % it
            out_file_real_name = '%010d_seg_real.png' % it
            out_file_real_name_syn = '%010d_syn_real.png' % it
            out_file_real_img = '%010d_syn_image.png' % it

        # image_grid = make_grid(image_real.tensor(), nrow=4)
        # save_image(image_grid, os.path.join(self.vis_dir, out_file_name_real))
        image_grids = make_grid(image_fake_seg.clamp_(0., 1.), nrow=4)
        image_grids = tensor2label(image_grids, 8)
        cv2.imwrite(os.path.join(self.vis_dir, out_file_name), image_grids)

        real_img = real['image']
        image_grids = make_grid(real_img.clamp_(0., 1.), nrow=4)
        save_image(image_grids, os.path.join(self.vis_dir, out_file_real_name_syn))

        real_seg = real_seg.cpu().numpy()[0,0:1,:,:]
        real_seg = np.transpose(real_seg, (1,2,0)) * 255.0
        cv2.imwrite(os.path.join(self.vis_dir, out_file_real_name), real_seg)

        image_gird = make_grid(image_fake.clamp_(0., 1.), nrow=4)
        image_gird = tensor2im(image_gird)
        cv2.imwrite(os.path.join(self.vis_dir, out_file_real_img), image_gird)

        return image_grids
