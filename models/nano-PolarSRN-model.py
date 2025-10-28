import numpy as np
import random
import torch
import torch.nn as nn
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import DiffJPEG, USMSharp, get_root_logger
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
from torch.nn import functional as F


def to_tensor(x):
    while isinstance(x, (tuple, list)):
        if len(x) == 0:
            return None
        x = x[0]
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    return x


class _AlignedLossAdapter(nn.Module):
    def __init__(self, base_loss: nn.Module, align_fn):
        super().__init__()
        self.base = base_loss
        self.align_fn = align_fn
    def forward(self, pred, target, *args, **kwargs):
        target = self.align_fn(target, pred)
        return self.base(pred, target, *args, **kwargs)


class _AlignedPerceptualAdapter(nn.Module):
    def __init__(self, base_loss: nn.Module, align_fn):
        super().__init__()
        self.base = base_loss
        self.align_fn = align_fn
    def forward(self, pred, target, *args, **kwargs):
        target = self.align_fn(target, pred)
        return self.base(pred, target, *args, **kwargs)


@MODEL_REGISTRY.register()
class RealESRGANModelPolar(SRGANModel):

    def __init__(self, opt):
        super(RealESRGANModelPolar, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()
        self.usm_sharpener = USMSharp().cuda()

        def _safe_w(key, default):
            val = opt.get(key, default)
            try:
                v = float(val)
            except Exception:
                v = default
            if not np.isfinite(v) or abs(v) > 1e6:
                print(f"[Warn] {key}={val} invalid; fallback to {default}")
                v = default
            return v

        self.dolp_loss_weight = _safe_w('dolp_loss_weight', 1.0)
        self.aop_loss_weight  = _safe_w('aop_loss_weight',  1.0)
        self.quads_l1_weight  = _safe_w('quads_l1_weight',  1.0)
        self.polar_boost      = _safe_w('polar_boost',       3.0)

        self.polar_eps        = opt.get('polar_eps', 1e-6)
        self.soft_tau_s0      = opt.get('soft_tau_s0', 1e-2)
        self.use_usm_for_polar= opt.get('use_usm_for_polar', False)
        self.clip_output_to01 = opt.get('clip_output_to01', True)
        self.aop_min_dolp     = float(opt.get('aop_min_dolp', 0.0))  # keep AoP active when DoLP~0

        # GT format: mosaic2x2 | quads4ch | stokes3ch
        self.gt_polar_format  = opt.get('gt_polar_format', 'mosaic2x2')
        self.gt_quad_order    = opt.get('gt_quad_order', [0,1,2,3])

        self.enable_ms_polar    = opt.get('enable_ms_polar', True)
        self.scale              = opt.get('scale', 4)
        self.gan_warmup_iters   = opt.get('gan_warmup_iters', 300)
        self.polar_ramp_iters   = opt.get('polar_ramp_iters', 100)
        self.grad_balance       = opt.get('grad_balance', True)
        self.grad_balance_clip  = opt.get('grad_balance_clip', 2.0)
        self.debug_print_every  = opt.get('debug_print_every', None)

        self.stokes_mid_weight = opt.get('stokes_mid_weight', 0.0)
        self.stokes_mid_w_s0   = opt.get('stokes_mid_w_s0', 0.5)
        self.stokes_mid_w_s1   = opt.get('stokes_mid_w_s1', 1.0)
        self.stokes_mid_w_s2   = opt.get('stokes_mid_w_s2', 1.0)
        self.stokes_mid_tau_s0 = opt.get('stokes_mid_tau_s0', 1e-2)
        self.stokes_head_act   = opt.get('stokes_head_act', 'lrelu')
        self.mid_feat_module   = opt.get('mid_feat_module', 'conv_body')

        if hasattr(self, 'cri_pix') and self.cri_pix is not None:
            self.cri_pix = _AlignedLossAdapter(self.cri_pix, self._align_to)
        if hasattr(self, 'cri_perceptual') and self.cri_perceptual is not None:
            self.cri_perceptual = _AlignedPerceptualAdapter(self.cri_perceptual, self._align_to)

        self.criterion_l1 = nn.L1Loss(reduction='none')

        self._mid_feat = None
        self._mid_hook = None
        self._stokes_head_mid = None
        self._polar_head_hr = None
        self._polar_head_hr_in_optim = False
        self._stokes_head_mid_in_optim = False

        self._register_g_mid_hook()
        print("[PolarModel] loaded from:", __file__)

    # ---------- utils ----------
    @staticmethod
    def _align_to(x, ref):
        Ht, Wt = ref.shape[-2:]
        Hx, Wx = x.shape[-2:]
        if (Hx == Ht) and (Wx == Wt):
            return x
        if (Hx > Ht) or (Wx > Wt):
            return F.interpolate(x, size=(Ht, Wt), mode='area')
        else:
            return F.interpolate(x, size=(Ht, Wt), mode='bilinear', align_corners=False)

    @staticmethod
    def _even_crop(x):
        H, W = x.shape[-2:]
        H2 = H - (H % 2); W2 = W - (W % 2)
        return x if (H2 == H and W2 == W) else x[..., :H2, :W2]

    def _register_g_mid_hook(self):
        target = None
        if hasattr(self.net_g, self.mid_feat_module):
            target = getattr(self.net_g, self.mid_feat_module)
        else:
            for name, m in self.net_g.named_modules():
                if self.mid_feat_module in name:
                    target = m; break
        if target is None:
            for name, m in self.net_g.named_modules():
                if 'conv_body' in name:
                    target = m; break
        if target is None:
            for name, m in self.net_g.named_modules():
                if name.endswith('body') or 'body' in name:
                    target = m; break
        if target is None:
            print(f'[Warn] Mid feature module "{self.mid_feat_module}" not found; deep Stokes supervision disabled.')
            return
        def _hook(_, __, output):
            self._mid_feat = output
        self._mid_hook = target.register_forward_hook(_hook)
        print(f'[Info] Mid feature hook on: {self.mid_feat_module}')

    def _build_polar_head_hr_if_needed(self, x):
        if self._polar_head_hr is None:
            self._polar_head_hr = nn.Sequential(
                nn.Conv2d(x.size(1), 32, 1, 1, 0),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(32, 4, 1, 1, 0)
            ).to(x.device)
            for m in self._polar_head_hr.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                    if m.bias is not None: nn.init.constant_(m.bias, 0.0)
            print('[Info] Built HR polar head (params added to optimizer_g on first use).')

        if (not self._polar_head_hr_in_optim) and hasattr(self, 'optimizer_g'):
            self.optimizer_g.add_param_group({'params': self._polar_head_hr.parameters(),
                                              'lr': self.optimizer_g.param_groups[0]['lr']})
            self._polar_head_hr_in_optim = True

    def _build_stokes_head_if_needed(self, feat_mid):
        if feat_mid is None:
            return
        if self._stokes_head_mid is None:
            in_ch = feat_mid.size(1)
            act = nn.LeakyReLU(0.1, inplace=True) if self.stokes_head_act == 'lrelu' else nn.ReLU(inplace=True)
            self._stokes_head_mid = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, 1, 1), act,
                nn.Conv2d(in_ch, max(64, in_ch // 2), 3, 1, 1), act,
                nn.Conv2d(max(64, in_ch // 2), 3, 3, 1, 1)  # -> (S0,S1,S2)
            ).to(feat_mid.device)
            for m in self._stokes_head_mid.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                    if m.bias is not None: nn.init.constant_(m.bias, 0.0)
            print('[Info] Built mid Stokes head (params added to optimizer_g on first use).')

        if (not self._stokes_head_mid_in_optim) and hasattr(self, 'optimizer_g'):
            self.optimizer_g.add_param_group({'params': self._stokes_head_mid.parameters(),
                                              'lr': self.optimizer_g.param_groups[0]['lr']})
            self._stokes_head_mid_in_optim = True

    def feed_data(self, data):
        if not (self.is_train and self.opt.get('high_order_degradation', True)):
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = (data['gt'][0] if isinstance(data['gt'], (tuple, list)) else data['gt']).to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)
        else:
            self.gt = (data['gt'][0] if isinstance(data['gt'], (tuple, list)) else data['gt']).to(self.device)
            self.gt_usm = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]
            out = filter2D(self.gt_usm, self.kernel1)

            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
            scale = (np.random.uniform(1, self.opt['resize_range'][1]) if updown_type == 'up'
                     else np.random.uniform(self.opt['resize_range'][0], 1) if updown_type == 'down' else 1)
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)

            gray_noise_prob = self.opt['gray_noise_prob']
            if np.random.uniform() < self.opt['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(out, sigma_range=self.opt['noise_range'], clip=True, rounds=False,
                                                   gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(out, scale_range=self.opt['poisson_scale_range'],
                                                  gray_prob=gray_noise_prob, clip=True, rounds=False)

            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)

            if np.random.uniform() < self.opt['second_blur_prob']:
                out = filter2D(out, self.kernel2)

            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
            scale = (np.random.uniform(1, self.opt['resize_range2'][1]) if updown_type == 'up'
                     else np.random.uniform(self.opt['resize_range2'][0], 1) if updown_type == 'down' else 1)
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out,
                                size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)),
                                mode=mode)

            gray_noise_prob = self.opt['gray_noise_prob2']
            if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False,
                                                   gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(out, scale_range=self.opt['poisson_scale_range2'],
                                                  gray_prob=gray_noise_prob, clip=True, rounds=False)

            if np.random.uniform() < 0.5:
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)

            self.lq = torch.clamp(out, 0, 1)

        self.gt = to_tensor(self.gt)
        self.gt_usm = to_tensor(self.gt_usm)
        self.lq = to_tensor(self.lq)

    @staticmethod
    def _to_gray(img):
        if img.size(1) == 3:
            r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
            return 0.2989 * r + 0.5870 * g + 0.1140 * b
        return img

    def split_polar(self, merged):
        merged = self._even_crop(merged)
        B, C, H, W = merged.size()
        h_half, w_half = H // 2, W // 2
        I0   = merged[:, :, :h_half, :w_half]
        I45  = merged[:, :, :h_half, w_half:]
        I90  = merged[:, :, h_half:, :w_half]
        I135 = merged[:, :, h_half:, w_half:]
        return [I0, I45, I90, I135]

    @staticmethod
    def _stokes_from_quads(I0, I45, I90, I135):
        S0 = I0 + I90
        S1 = I0 - I90
        S2 = I45 - I135
        return S0, S1, S2

    @staticmethod
    def _dolp_aop(S0, S1, S2, eps=1e-6):
        dolp = torch.sqrt(S1 * S1 + S2 * S2) / (S0.abs() + eps)
        aop  = 0.5 * torch.atan2(S2, S1 + 1e-12)
        return dolp, aop

    @staticmethod
    def _weighted_mean(x, w, eps=1e-6):
        return (x * w).sum() / (w.sum() + eps)

    # --------- FIXED: split first, then align each quad to ref size ---------
    def _extract_gt_quads(self, gt_tensor, ref_shape=None):
        fmt = self.gt_polar_format.lower()

        if fmt == 'mosaic2x2':
            I0, I45, I90, I135 = self.split_polar(gt_tensor)  # each is H/2 x W/2
            quads = [self._to_gray(x) for x in (I0, I45, I90, I135)]
        elif fmt == 'quads4ch':
            if gt_tensor.size(1) < 4:
                raise RuntimeError('[Polar GT] gt_polar_format=quads4ch but gt channel < 4.')
            idx = self.gt_quad_order
            I0   = gt_tensor[:, idx[0]:idx[0]+1]
            I45  = gt_tensor[:, idx[1]:idx[1]+1]
            I90  = gt_tensor[:, idx[2]:idx[2]+1]
            I135 = gt_tensor[:, idx[3]:idx[3]+1]
            quads = [self._to_gray(x) for x in (I0, I45, I90, I135)]
        elif fmt == 'stokes3ch':
            if gt_tensor.size(1) < 3:
                raise RuntimeError('[Polar GT] gt_polar_format=stokes3ch but gt channel < 3.')
            S0, S1, S2 = gt_tensor[:, 0:1], gt_tensor[:, 1:2], gt_tensor[:, 2:3]
            I0   = (S0 + S1) / 2
            I90  = (S0 - S1) / 2
            I45  = (S0 + S2) / 2
            I135 = (S0 - S2) / 2
            quads = [I0, I45, I90, I135]
        else:
            raise RuntimeError(f'[Polar GT] Unknown gt_polar_format: {self.gt_polar_format}')

        # align each quad to reference spatial size if provided
        if ref_shape is not None:
            quads = [self._align_to(q, ref_shape) for q in quads]

        return quads

    def _polar_losses_from_pred_quads(self, pred_quads, gt_tensor):

        fmt = self.gt_polar_format.lower()
        if fmt == 'mosaic2x2':
            pred_quads = self._even_crop(pred_quads)
            gt_tensor = self._align_to(self._even_crop(gt_tensor), pred_quads)

            B, C, H, W = pred_quads.shape
            h2, w2 = H // 2, W // 2

            I0g, I45g, I90g, I135g = self.split_polar(gt_tensor)
            I0g, I45g, I90g, I135g = [self._to_gray(x) for x in (I0g, I45g, I90g, I135g)]

            I0p = pred_quads[:, 0:1, :h2, :w2]
            I45p = pred_quads[:, 1:2, :h2, w2:]
            I90p = pred_quads[:, 2:3, h2:, :w2]
            I135p = pred_quads[:, 3:4, h2:, w2:]

            l_q = (
                          self.criterion_l1(I0p, I0g).mean() +
                          self.criterion_l1(I45p, I45g).mean() +
                          self.criterion_l1(I90p, I90g).mean() +
                          self.criterion_l1(I135p, I135g).mean()
                  ) * 0.25

            S0p, S1p, S2p = self._stokes_from_quads(I0p, I45p, I90p, I135p)
            S0g, S1g, S2g = self._stokes_from_quads(I0g, I45g, I90g, I135g)

            dolp_p, aop_p = self._dolp_aop(S0p, S1p, S2p, eps=self.polar_eps)
            dolp_g, aop_g = self._dolp_aop(S0g, S1g, S2g, eps=self.polar_eps)

            w_s0 = (S0g.abs()) / (S0g.abs() + self.soft_tau_s0)
            w_d = w_s0
            if self.aop_min_dolp > 0:
                w_a = w_s0 * torch.clamp(dolp_g, self.aop_min_dolp, 1.0)
            else:
                w_a = w_s0 * torch.clamp(dolp_g, 0.0, 1.0)

            l_d = self._weighted_mean(torch.abs(dolp_p - dolp_g), w_d)
            delta = aop_p - aop_g
            l_a = self._weighted_mean(1.0 - torch.cos(2.0 * delta), w_a)

            l_total = self.quads_l1_weight * l_q + self.dolp_loss_weight * l_d + self.aop_loss_weight * l_a
            return l_total, l_q, l_d, l_a

        else:

            I0g, I45g, I90g, I135g = self._extract_gt_quads(gt_tensor, ref_shape=pred_quads)
            I0p, I45p, I90p, I135p = pred_quads[:, 0:1], pred_quads[:, 1:2], pred_quads[:, 2:3], pred_quads[:, 3:4]

            l_q = (
                          self.criterion_l1(I0p, I0g).mean() +
                          self.criterion_l1(I45p, I45g).mean() +
                          self.criterion_l1(I90p, I90g).mean() +
                          self.criterion_l1(I135p, I135g).mean()
                  ) * 0.25

            S0p, S1p, S2p = self._stokes_from_quads(I0p, I45p, I90p, I135p)
            S0g, S1g, S2g = self._stokes_from_quads(I0g, I45g, I90g, I135g)

            dolp_p, aop_p = self._dolp_aop(S0p, S1p, S2p, eps=self.polar_eps)
            dolp_g, aop_g = self._dolp_aop(S0g, S1g, S2g, eps=self.polar_eps)

            w_s0 = (S0g.abs()) / (S0g.abs() + self.soft_tau_s0)
            w_d = w_s0
            if self.aop_min_dolp > 0:
                w_a = w_s0 * torch.clamp(dolp_g, self.aop_min_dolp, 1.0)
            else:
                w_a = w_s0 * torch.clamp(dolp_g, 0.0, 1.0)

            l_d = self._weighted_mean(torch.abs(dolp_p - dolp_g), w_d)
            delta = aop_p - aop_g
            l_a = self._weighted_mean(1.0 - torch.cos(2.0 * delta), w_a)

            l_total = self.quads_l1_weight * l_q + self.dolp_loss_weight * l_d + self.aop_loss_weight * l_a
            return l_total, l_q, l_d, l_a

    def _polar_losses_ms(self, pred_quads_hr, gt_hr):

        l_hr, lq_hr, ld_hr, la_hr = self._polar_losses_from_pred_quads(pred_quads_hr, gt_hr)


        if self.enable_ms_polar and self.scale and self.scale > 1:
            H, W = pred_quads_hr.shape[-2:]
            lr_size = (max(2, H // self.scale), max(2, W // self.scale))
            pred_quads_lr = F.interpolate(pred_quads_hr, size=lr_size, mode='area')

            gt_lr = F.interpolate(self._align_to(gt_hr, pred_quads_hr), size=lr_size, mode='area')

            l_lr, lq_lr, ld_lr, la_lr = self._polar_losses_from_pred_quads(pred_quads_lr, gt_lr)

            l_tot = 0.5 * (l_hr + l_lr)
            l_q = 0.5 * (lq_hr + lq_lr)
            l_d = 0.5 * (ld_hr + ld_lr)
            l_a = 0.5 * (la_hr + la_lr)
        else:
            l_tot, l_q, l_d, l_a = l_hr, lq_hr, ld_hr, la_hr

        return l_tot, l_q, l_d, l_a

    # --------- FIXED: for mid Stokes, align channels/quads AFTER split ---------
    def compute_stokes_mid_loss(self, feat_mid, gt_hr):
        if feat_mid is None or self.stokes_mid_weight <= 0:
            z = torch.zeros((), device=gt_hr.device)
            return z, z, z, z
        self._build_stokes_head_if_needed(feat_mid)

        Sp_pred = self._stokes_head_mid(feat_mid)  # (B,3,hm,wm)
        S0p_pred, S1p_pred, S2p_pred = Sp_pred[:, 0:1], Sp_pred[:, 1:2], Sp_pred[:, 2:3]

        fmt = self.gt_polar_format.lower()
        if fmt == 'stokes3ch' and gt_hr.size(1) >= 3:
            S0g = self._align_to(gt_hr[:,0:1], S0p_pred)
            S1g = self._align_to(gt_hr[:,1:2], S1p_pred)
            S2g = self._align_to(gt_hr[:,2:3], S2p_pred)
        else:
            I0g, I45g, I90g, I135g = self._extract_gt_quads(gt_hr, ref_shape=S0p_pred)
            S0g, S1g, S2g = self._stokes_from_quads(I0g, I45g, I90g, I135g)

        w_s0 = (S0g.abs()) / (S0g.abs() + self.stokes_mid_tau_s0)
        l_s0 = self._weighted_mean(torch.abs(S0p_pred - S0g), w_s0)
        l_s1 = self._weighted_mean(torch.abs(S1p_pred - S1g), w_s0)
        l_s2 = self._weighted_mean(torch.abs(S2p_pred - S2g), w_s0)

        loss_mid = self.stokes_mid_weight * (
            self.stokes_mid_w_s0 * l_s0 + self.stokes_mid_w_s1 * l_s1 + self.stokes_mid_w_s2 * l_s2
        )
        return loss_mid, l_s0, l_s1, l_s2

    @staticmethod
    def _grad_norm(loss_scalar, wrt_tensor):
        try:
            g = torch.autograd.grad(loss_scalar, wrt_tensor, retain_graph=True, allow_unused=True)[0]
            return torch.zeros((), device=wrt_tensor.device) if g is None else g.norm().detach()
        except RuntimeError:
            return torch.zeros((), device=wrt_tensor.device)

    def _polar_ramp_factor(self, it):
        if self.polar_ramp_iters <= 0:
            return 1.0
        return float(np.clip(it / max(1, self.polar_ramp_iters), 0.0, 1.0))

    def _gan_active(self, it):
        return it >= max(0, self.gan_warmup_iters)

    def optimize_parameters(self, current_iter):
        l1_gt = self.gt_usm if self.opt.get('l1_gt_usm', False) else self.gt
        percep_gt = self.gt_usm if self.opt.get('percep_gt_usm', False) else self.gt
        gan_gt = self.gt_usm if self.opt.get('gan_gt_usm', False) else self.gt

        gt_for_polar = self.gt_usm if (self.use_usm_for_polar and hasattr(self, 'gt_usm')) else self.gt

        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self._mid_feat = None

        self.output = self.net_g(self.lq)
        if self.clip_output_to01:
            self.output = torch.clamp(self.output, 0, 1)

        self._build_polar_head_hr_if_needed(self.output)
        pred_quads_hr = torch.sigmoid(self._polar_head_hr(self.output))

        l_polar_base, l_quads_l1, l_dolp, l_aop = self._polar_losses_ms(pred_quads_hr, gt_for_polar)
        ramp = self._polar_ramp_factor(current_iter)
        balance_scale = 1.0
        if self.grad_balance and hasattr(self, 'cri_pix') and self.cri_pix is not None:
            gn_pix   = self._grad_norm(self.cri_pix(self.output, l1_gt), self.output) + 1e-12
            gn_polar = self._grad_norm(l_polar_base, self.output) + 1e-12
            ratio = (gn_pix / gn_polar).clamp(1.0 / self.grad_balance_clip, self.grad_balance_clip)
            balance_scale = float(ratio.item())
        l_g_polar = self.polar_boost * ramp * balance_scale * l_polar_base

        l_g_pix = self.cri_pix(self.output, l1_gt) if hasattr(self, 'cri_pix') and self.cri_pix is not None else 0.0
        l_g_total = l_g_pix if hasattr(l_g_pix, 'detach') else torch.tensor(l_g_pix, device=self.output.device)

        if hasattr(self, 'cri_perceptual') and self.cri_perceptual is not None:
            l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
            if l_g_percep is not None: l_g_total = l_g_total + l_g_percep
            if l_g_style  is not None: l_g_total = l_g_total + l_g_style

        if self._gan_active(current_iter):
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total = l_g_total + l_g_gan
        else:
            l_g_gan = torch.zeros((), device=self.output.device)

        l_stokes_mid, l_s0, l_s1, l_s2 = self.compute_stokes_mid_loss(self._mid_feat, gt_for_polar)
        if (self._stokes_head_mid is not None) and (not self._stokes_head_mid_in_optim) and hasattr(self, 'optimizer_g'):
            self.optimizer_g.add_param_group({'params': self._stokes_head_mid.parameters(),
                                              'lr': self.optimizer_g.param_groups[0]['lr']})
            self._stokes_head_mid_in_optim = True
        l_g_total = l_g_total + l_stokes_mid

        l_g_total = l_g_total + l_g_polar

        l_g_total.backward()
        self.optimizer_g.step()

        loss_dict = OrderedDict()
        loss_dict['l_g_pix']           = l_g_pix.detach() if hasattr(l_g_pix, 'detach') else torch.tensor(float(l_g_pix), device=self.output.device)
        loss_dict['l_g_percep']        = l_g_percep.detach() if 'l_g_percep' in locals() and l_g_percep is not None else torch.zeros((), device=self.output.device)
        loss_dict['l_g_style']         = l_g_style.detach()  if 'l_g_style'  in locals() and l_g_style  is not None else torch.zeros((), device=self.output.device)
        loss_dict['l_g_gan']           = l_g_gan.detach()
        loss_dict['l_stokes_mid']      = l_stokes_mid.detach()
        loss_dict['l_stokes_mid_s0']   = l_s0.detach()
        loss_dict['l_stokes_mid_s1']   = l_s1.detach()
        loss_dict['l_stokes_mid_s2']   = l_s2.detach()

        loss_dict['l_g_polar']         = l_g_polar.detach()
        loss_dict['l_g_polar_quads']   = (self.polar_boost * ramp * balance_scale * l_quads_l1).detach()
        loss_dict['l_g_polar_dolp']    = (self.polar_boost * ramp * balance_scale * l_dolp).detach()
        loss_dict['l_g_polar_aop']     = (self.polar_boost * ramp * balance_scale * l_aop).detach()
        loss_dict['polar_ramp']        = torch.tensor(ramp, device=self.output.device)
        loss_dict['polar_balance']     = torch.tensor(balance_scale, device=self.output.device)
        loss_dict['polar_boost']       = torch.tensor(self.polar_boost, device=self.output.device)
        loss_dict['gn_polar_dbg']      = self._grad_norm(l_polar_base, self.output)
        loss_dict['gn_pix_dbg']        = self._grad_norm(self.cri_pix(self.output, l1_gt) if hasattr(self,'cri_pix') and self.cri_pix is not None else torch.zeros((), device=self.output.device, requires_grad=True), self.output)

        for p in self.net_d.parameters():
            p.requires_grad = True
        self.optimizer_d.zero_grad()
        if self._gan_active(current_iter):
            gan_gt_aligned = self._align_to(gan_gt, self.output)
            real_d_pred = self.net_d(gan_gt_aligned)
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real']   = l_d_real.detach()
            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            l_d_real.backward()

            fake_d_pred = self.net_d(self.output.detach().clone())
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake']   = l_d_fake.detach()
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
            l_d_fake.backward()
            self.optimizer_d.step()
        else:
            loss_dict['l_d_real']   = torch.zeros((), device=self.output.device)
            loss_dict['out_d_real'] = torch.zeros((), device=self.output.device)
            loss_dict['l_d_fake']   = torch.zeros((), device=self.output.device)
            loss_dict['out_d_fake'] = torch.zeros((), device=self.output.device)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)

        try:
            rank0 = (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)
        except Exception:
            rank0 = True
        if rank0:
            logger = get_root_logger()
            print_freq = int(self.opt.get('logger', {}).get('print_freq', 100))
            if self.debug_print_every is not None:
                print_freq = int(self.debug_print_every)
            if current_iter % max(1, print_freq) == 0:
                logger.info(
                    f"[polar] iter:{current_iter:>6} "
                    f"l_g_polar:{float(self.log_dict['l_g_polar']):.4e} "
                    f"(quads:{float(self.log_dict['l_g_polar_quads']):.4e}, "
                    f"dolp:{float(self.log_dict['l_g_polar_dolp']):.4e}, "
                    f"aop:{float(self.log_dict['l_g_polar_aop']):.4e}) "
                    f"gn_polar:{float(self.log_dict['gn_polar_dbg']):.4e} "
                    f"ramp:{float(self.log_dict['polar_ramp']):.3f} "
                    f"bal:{float(self.log_dict['polar_balance']):.3f} "
                    f"boost:{float(self.log_dict['polar_boost']):.1f}"
                )
