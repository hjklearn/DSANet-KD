import torch
from torch import nn
import torch.nn.functional as F


class KLDLoss(nn.Module):
    def __init__(self, alpha=1, tau=1, resize_config=None, shuffle_config=None, transform_config=None,
                 warmup_config=None, earlydecay_config=None):
        super().__init__()
        self.alpha_0 = alpha
        self.alpha = alpha
        self.tau = tau

        self.resize_config = resize_config
        self.shuffle_config = shuffle_config
        # print("self.shuffle", self.shuffle_config)
        self.transform_config = transform_config
        self.warmup_config = warmup_config
        self.earlydecay_config = earlydecay_config

        self.KLD = torch.nn.KLDivLoss(reduction='sum')

    def resize(self, x, gt):
        mode = self.resize_config['mode']
        align_corners = self.resize_config['align_corners']
        x = F.interpolate(
            input=x,
            size=gt.shape[2:],
            mode=mode,
            align_corners=align_corners)
        return x

    def shuffle(self, x_student, x_teacher, n_iter):
        interval = self.shuffle_config['interval']
        print(interval, "1")
        B, C, W, H = x_student.shape
        if n_iter % interval == 0:
            print("2")
            idx = torch.randperm(C)
            x_student = x_student[:, idx, :, :].contiguous()
            x_teacher = x_teacher[:, idx, :, :].contiguous()
        print("3")
        return x_student, x_teacher

    def transform(self, x):
        B, C, W, H = x.shape
        loss_type = self.transform_config['loss_type']
        if loss_type == 'pixel':
            x = x.permute(0, 2, 3, 1)
            x = x.reshape(B, W * H, C)
        elif loss_type == 'channel':
            group_size = self.transform_config['group_size']
            if C % group_size == 0:
                x = x.reshape(B, C // group_size, -1)
            else:
                n = group_size - C % group_size
                x_pad = -1e9 * torch.ones(B, n, W, H).cuda()
                x = torch.cat([x, x_pad], dim=1)
                x = x.reshape(B, (C + n) // group_size, -1)
        return x

    def warmup(self, n_iter):
        # print("war")
        mode = self.warmup_config['mode']
        warmup_iters = self.warmup_config['warmup_iters']
        if n_iter > warmup_iters:
            return
        elif n_iter == warmup_iters:
            self.alpha = self.alpha_0
            return
        else:
            if mode == 'linear':
                self.alpha = self.alpha_0 * (n_iter / warmup_iters)
            elif mode == 'exp':
                self.alpha = self.alpha_0 ** (n_iter / warmup_iters)
            elif mode == 'jump':
                self.alpha = 0

    def earlydecay(self, n_iter):
        mode = self.earlydecay_config['mode']
        earlydecay_start = self.earlydecay_config['earlydecay_start']
        earlydecay_end = self.earlydecay_config['earlydecay_end']

        if n_iter < earlydecay_start:
            return
        elif n_iter > earlydecay_start and n_iter < earlydecay_end:
            if mode == 'linear':
                self.alpha = self.alpha_0 * ((earlydecay_end - n_iter) / (earlydecay_end - earlydecay_start))
            elif mode == 'exp':
                self.alpha = 0.001 * self.alpha_0 ** ((earlydecay_end - n_iter) / (earlydecay_end - earlydecay_start))
            elif mode == 'jump':
                self.alpha = 0
        elif n_iter >= earlydecay_end:
            self.alpha = 0

    def forward(self, x_student, x_teacher, gt, n_iter):
        # print("start kld")
        if self.warmup_config:
            print("warm")
            self.warmup(n_iter)
        if self.earlydecay_config:
            print("decay")
            self.earlydecay(n_iter)

        if self.resize_config:
            print("resize(")
            x_student, x_teacher = self.resize(x_student, gt), self.resize(x_teacher, gt)
        if self.shuffle_config:
            print("shuffle")
            x_student, x_teacher = self.shuffle(x_student, x_teacher, n_iter)
        if self.transform_config:
            print("transform")
            x_student, x_teacher = self.transform(x_student), self.transform(x_teacher)
        # print("hhh")

        x_student = F.log_softmax(x_student / self.tau, dim=-1)
        x_teacher = F.softmax(x_teacher / self.tau, dim=-1)
        loss = self.KLD(x_student, x_teacher) / (x_student.numel() / x_student.shape[-1])
        # print("self.alpha", self.alpha)
        loss = self.alpha * loss
        return loss


class OFD(nn.Module):
    '''
	A Comprehensive Overhaul of Feature Distillation
	http://openaccess.thecvf.com/content_ICCV_2019/papers/
	Heo_A_Comprehensive_Overhaul_of_Feature_Distillation_ICCV_2019_paper.pdf
	'''

    def __init__(self, in_channels, out_channels):
        super(OFD, self).__init__()
        self.connector = nn.Sequential(*[
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, fm_s, fm_t):
        margin = self.get_margin(fm_t)
        fm_t = torch.max(fm_t, margin)
        fm_s = self.connector(fm_s)

        mask = 1.0 - ((fm_s <= fm_t) & (fm_t <= 0.0)).float()
        loss = torch.mean((fm_s - fm_t) ** 2 * mask)

        return loss

    def get_margin(self, fm, eps=1e-6):
        mask = (fm < 0.0).float()
        masked_fm = fm * mask

        margin = masked_fm.sum(dim=(0, 2, 3), keepdim=True) / (mask.sum(dim=(0, 2, 3), keepdim=True) + eps)

        return margin


class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduce='mean')
        self.KLD = nn.KLDivLoss(reduction='sum')

    def _resize(self,x,x_t):
        x = F.interpolate(
            input=x,
            size=x_t.shape[2:],
            mode='bilinear',align_corners=False)
        return x

    def forward(self,x_student, x_teacher):
        # x_student = self._resize(x_student,x_teacher)
        loss_AT = self.mse(x_student.mean(dim=1), x_teacher.mean(dim=1))

        x_student = F.log_softmax(x_student,dim=1)
        x_teacher = F.softmax(x_teacher,dim=1)

        loss_PD = self.KLD(x_student, x_teacher)/(x_student.numel()/x_student.shape[1])
        loss = loss_AT + loss_PD
        return loss

class Similarity(nn.Module):
    ##Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author##
    def __init__(self):
        super(Similarity, self).__init__()

    def forward(self, g_s, g_t):
        return self.similarity_loss(g_s, g_t)

    def similarity_loss(self, f_s, f_t):
        # print('f_s', f_s.shape)
        bsz = f_s.shape[0]
        # print('bsz', bsz)
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)
        # print('f_s', f_s.shape)

        G_s = torch.mm(f_s, torch.t(f_s))
        # G_s = G_s / G_s.norm(2)
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        # G_t = G_t / G_t.norm(2)
        G_t = torch.nn.functional.normalize(G_t)
        # print('G_t', G_t.shape)
        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        # print('(G_diff * G_diff).view(-1, 1)', (G_diff * G_diff).view(-1, 1).shape)
        return loss

class Attention(nn.Module):
    """Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks
    via Attention Transfer
    code: https://github.com/szagoruyko/attention-transfer"""
    def __init__(self, p=2):
        super(Attention, self).__init__()
        self.p = p

    def forward(self, g_s, g_t):
        return self.at_loss(g_s, g_t)

    def at_loss(self, f_s, f_t):
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))


class CriterionPairWiseforWholeFeatAfterPool(nn.Module):
    def __init__(self, scale):
        '''inter pair-wise loss from inter feature maps'''
        super(CriterionPairWiseforWholeFeatAfterPool, self).__init__()
        self.criterion = sim_dis_compute
        self.scale = scale

        # self.connector = nn.Sequential(*[
        #     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(out_channels)
        # ])

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, preds_S, preds_T):
        feat_S = preds_S
        feat_T = preds_T
        feat_T.detach()

        total_w, total_h = feat_T.shape[2], feat_T.shape[3]
        patch_w, patch_h = int(total_w*self.scale), int(total_h*self.scale)
        maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True) # change
        loss = self.criterion(maxpool(feat_S), maxpool(feat_T))
        return loss

def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3]) + 1e-8

def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
    return torch.einsum('icm,icn->imn', [feat, feat])

def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis

a = torch.randn(2, 3, 224, 224)
b = torch.randn(2, 3, 224, 224)
model = Similarity()
result = model(a, b)
