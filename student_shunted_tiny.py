import torch
import torch.nn as nn
from backbone.Shunted.SSA import *

class BasicConv2d(nn.Module):
    def __init__(self, in_planes):
        super(BasicConv2d, self).__init__()

        self.conv1x3 = nn.Conv2d(in_planes, in_planes,
                                 kernel_size=(1, 3), stride=1,
                                 padding=(0, 1), dilation=1, bias=False)
        self.conv3x1 = nn.Conv2d(in_planes, in_planes,
                                 kernel_size=(3, 1), stride=1,
                                 padding=(1, 0), dilation=1, bias=False)
        self.bn = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1x3(x)
        x = self.conv3x1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class single_fusion(nn.Module):
    def __init__(self, in_planes, stride=1, dilation=1):
        super(single_fusion, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.sigomid = nn.Sigmoid()
        self.fc = nn.Conv2d(2 * in_planes, 2 * in_planes, 1)

    def forward(self, r4, r3):
        r4_avg = self.avgpool(r4)
        r4_max = self.maxpool(r4)
        add_r4 = r4_avg + r4_max
        r3_avg = self.avgpool(r3)
        r3_max = self.maxpool(r3)
        add_r3 = r3_max + r3_avg

        mul_r43 = add_r3.mul(add_r4)
        add_r43 = add_r3 + add_r4
        cat_all1 = torch.cat((mul_r43, add_r43), dim=1)
        cat_all = self.fc(cat_all1)
        cat_all = self.sigomid(cat_all)
        _, c, _, _ = cat_all.size()
        cat_all_w1, cat_all_w2 = cat_all.split(c//2, dim=1)

        r4_new = r4 * cat_all_w1
        r3_new = r3 * cat_all_w2
        out = r3_new + r4_new
        return out

class Decoder(nn.Module):
    def __init__(self, inchannel, outchannel, n):
        super(Decoder, self).__init__()
        self.conv3x3 = BasicConv2d(inchannel)
        self.d1 = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.Upsample(scale_factor=n, mode='bilinear', align_corners=True))


    def forward(self, x):
        x = self.conv3x3(x)
        x = self.d1(x)
        return x


class fusion(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(fusion, self).__init__()
        self.conv3 = nn.Sequential(nn.Conv2d(3 * inchannel, outchannel, kernel_size=(1, 3), stride=1, padding=(0, 1), dilation=1, bias=False),
                                   nn.Conv2d(outchannel, outchannel, kernel_size=(3, 1), stride=1, padding=(1, 0), dilation=1, bias=False),
                                   nn.BatchNorm2d(outchannel),
                                   nn.ReLU(inplace=True))
        self.conv31 = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=(1, 3), stride=1, padding=(0, 1), dilation=1, bias=False),
                                   nn.Conv2d(outchannel, outchannel, kernel_size=(3, 1), stride=1, padding=(1, 0), dilation=1, bias=False),
                                   nn.BatchNorm2d(outchannel),
                                   nn.ReLU(inplace=True))
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(3 * outchannel, outchannel, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=outchannel, out_channels=3, kernel_size=1),
                                     nn.Softmax(dim=1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, r, d):
        mul = r * d
        cat_rd = torch.cat([r, d, mul], dim=1)
        cat_rdc = self.conv3(cat_rd)
        mean_c = torch.mean(cat_rdc, dim=1, keepdim=True)
        mean_h = torch.mean(cat_rdc, dim=2, keepdim=True)
        mean_w = torch.mean(cat_rdc, dim=3, keepdim=True)

        mean_c = self.sigmoid(mean_c)
        mean_h = self.sigmoid(mean_h)
        mean_w = self.sigmoid(mean_w)

        mul_c = mean_c * cat_rdc
        mul_h = mean_h * cat_rdc
        mul_w = mean_w * cat_rdc

        cat_rd_attention = self.avgpool(cat_rd)
        w1, w2, w3 = cat_rd_attention.split(1, dim=1)
        xxx = (mul_c * w1)
        zzz = (mul_h * w2)
        yyy = (mul_w * w3)
        out = xxx + zzz + yyy
        out = self.conv31(out)

        return out

class fusion1(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(fusion1, self).__init__()
        self.conv3 = nn.Sequential(nn.Conv2d(3 * inchannel, outchannel, kernel_size=(1, 3), stride=1, padding=(0, 1), dilation=1, bias=False),
                                   nn.Conv2d(outchannel, outchannel, kernel_size=(3, 1), stride=1, padding=(1, 0), dilation=1, bias=False),
                                   nn.BatchNorm2d(outchannel),
                                   nn.ReLU(inplace=True))
        self.conv31 = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=(1, 3), stride=1, padding=(0, 1), dilation=1, bias=False),
                                   nn.Conv2d(outchannel, outchannel, kernel_size=(3, 1), stride=1, padding=(1, 0), dilation=1, bias=False),
                                   nn.BatchNorm2d(outchannel),
                                   nn.ReLU(inplace=True))
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(3 * outchannel, outchannel, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=outchannel, out_channels=3, kernel_size=1),
                                     nn.Softmax(dim=1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, rd, r, d):
        mul = r * d
        cat_rd = torch.cat([r, d, mul], dim=1)
        cat_rdc = self.conv3(cat_rd)
        mean_c = torch.mean(cat_rdc, dim=1, keepdim=True)
        mean_h = torch.mean(cat_rdc, dim=2, keepdim=True)
        mean_w = torch.mean(cat_rdc, dim=3, keepdim=True)

        mean_c = self.sigmoid(mean_c)
        mean_h = self.sigmoid(mean_h)
        mean_w = self.sigmoid(mean_w)

        mul_c = mean_c * cat_rdc
        mul_h = mean_h * cat_rdc
        mul_w = mean_w * cat_rdc

        cat_rd_attention = self.avgpool(cat_rd)
        w1, w2, w3 = cat_rd_attention.split(1, dim=1)
        xxx = (mul_c * w1)
        zzz = (mul_h * w2)
        yyy = (mul_w * w3)
        out = xxx + zzz + yyy + rd
        out = self.conv31(out)

        return out



class fourmodel_student(nn.Module):
    def __init__(self, ):
        super(fourmodel_student, self).__init__()

        self.backbonet = shunted_t()
        self.backboner = shunted_t()

        self.s4_decoder = Decoder(64, 1, 4)
        self.s2_decoder = Decoder(128, 1, 8)
        self.s1_decoder = Decoder(256, 1, 16)
        self.s3_decoder = Decoder(64, 1, 4)
        self.d_decoder = Decoder(64, 1, 4)
        self.r_decoder = Decoder(64, 1, 4)

        self.single_modal_fusion_r4r3 = single_fusion(256)
        self.single_modal_fusion_r3r2 = single_fusion(128)
        self.single_modal_fusion_r2r1 = single_fusion(64)

        self.single_modal_fusion_d4d3 = single_fusion(256)
        self.single_modal_fusion_d3d2 = single_fusion(128)
        self.single_modal_fusion_d2d1 = single_fusion(64)

        self.fusion1 = fusion(256, 256)
        self.fusion2 = fusion1(256, 256)
        self.fusion3 = fusion1(128, 128)
        self.fusion4 = fusion1(64, 64)

        self.c_2048_1024_r = nn.Sequential(nn.Conv2d(512, 256, 1),
                                         nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.c_1024_512_r = nn.Sequential(nn.Conv2d(256, 128, 1),
                                         nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.c_512_256_r = nn.Sequential(nn.Conv2d(128, 64, 1),
                                         nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

        self.c_2048_1024_d = nn.Sequential(nn.Conv2d(512, 256, 1),
                                         nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.c_1024_512_d = nn.Sequential(nn.Conv2d(256, 128, 1),
                                         nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.c_512_256_d = nn.Sequential(nn.Conv2d(128, 64, 1),
                                         nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

        self.c_512_256_f = nn.Sequential(nn.Conv2d(128, 64, 1),
                                         nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.c_1024_512_f = nn.Sequential(nn.Conv2d(256, 128, 1),
                                         nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))


    def forward(self, r, d):
        d = torch.cat([d, d, d], dim=1)
        r1, r2, r3, r4 = self.backboner(r)
        d1, d2, d3, d4 = self.backbonet(d)

        r4_c = self.c_2048_1024_r(r4)
        # add3_r = r4_c + r3
        add3_r = self.single_modal_fusion_r4r3(r4_c, r3)
        r3_c = self.c_1024_512_r(add3_r)
        # add2_r = r3_c + r2
        add2_r = self.single_modal_fusion_r3r2(r3_c, r2)
        r2_c = self.c_512_256_r(add2_r)
        # add1_r = r2_c + r1
        add1_r = self.single_modal_fusion_r2r1(r2_c, r1)
        out_r = self.r_decoder(add1_r)

        d4_c = self.c_2048_1024_d(d4)
        # add3_d = d4_c + d3
        add3_d = self.single_modal_fusion_d4d3(d4_c, d3)
        d3_c = self.c_1024_512_d(add3_d)
        # add2_d = d3_c + d2
        add2_d = self.single_modal_fusion_d3d2(d3_c, d2)
        d2_c = self.c_512_256_d(add2_d)
        # add1_d = d2_c + d1
        add1_d = self.single_modal_fusion_d2d1(d2_c, d1)
        out_d = self.d_decoder(add1_d)

        # s1 = r4_c + d4_c
        s1 = self.fusion1(r4_c, d4_c)
        # s2 = s1 + add3_r + add3_d
        s2_1 = self.fusion2(s1, add3_r, add3_d)
        s2 = self.c_1024_512_f(s2_1)
        # s3 = s2 + add2_r + add2_d
        s3_1 = self.fusion3(s2, add2_r, add2_d)
        s3 = self.c_512_256_f(s3_1)
        # s4 = s3 + add1_r + add1_d
        s4 = self.fusion4(s3, add1_r, add1_d)

        s1_out = self.s1_decoder(s1)
        s2_out = self.s2_decoder(s2)
        s3_out = self.s3_decoder(s3)
        s4_out = self.s4_decoder(s4)


        return s4_out, s3_out, s2_out, s1_out, out_r, out_d
        # return s4_out, s3_out, s2_out, s1_out, out_r, out_d, add3_r, add3_d, add2_r, add2_d, add1_r, add1_d, r4, d4, r3, d3, r2, d2, r1, d1
        # return out_r

    def load_pre(self, pre_model):
        # save_model = torch.load(pre_model)
        # model_dict_r = self.resnet_r.state_dict()
        # state_dict_r = {k: v for k, v in save_model.items() if k in model_dict_r.keys()}
        # self.resnet_r.load_state_dict(state_dict_r)
        # self.resnet_d.load_state_dict(state_dict_r)
        # print('self.resnet_r, resnet_d loading')
        ##################################################################################
        # from collections import OrderedDict
        # new_state_dict3 = OrderedDict()
        # state_dict = torch.load(pre_model)['state_dict']
        # for k, v in state_dict.items():
        #     name = "backbone" + k
        #     new_state_dict3[name] = v
        # self.mit_r.load_state_dict(new_state_dict3, strict=True)
        # self.mit_d.load_state_dict(new_state_dict3, strict=False)
        # print('self.mit_r loading', 'self.mit_d loading')
        ######################################################################################
        # state_dict = torch.load(pre_model)['state_dict']
        # for key in list(state_dict.keys()):
        #     state_dict[key.replace('module.', '')] = state_dict.pop(key)
        #
        # self.resnet_r.load_state_dict(state_dict)
        # self.resnet_d.load_state_dict(state_dict)
        # print('self.rgb_uniforr loading', 'self.depth_unifor loading')
        #######################################################################################
        save_model = torch.load(pre_model)
        self.backboner.load_state_dict(save_model)
        self.backbonet.load_state_dict(save_model)
        print('self.backboner loading', 'self.backboner loading')


if __name__ == '__main__':
    a = torch.randn(2, 3, 224, 224).cuda()
    b = torch.randn(2, 1, 224, 224).cuda()
    model =fourmodel_student().cuda()

    from FLOP import CalParams
    CalParams(model, a, b)

    print('Total params % .2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    # total_params = sum(p.numel() for p in model.parameters())
    # print('Parameters:', total_params / 1000000)

    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        # net = GTLW(n_classes=9).cuda()
        net = fourmodel_teacher().cuda()
        flops, params = get_model_complexity_info(net, (3, 256, 256), as_strings=True, print_per_layer_stat=False)
        print('Flops:  ' + flops)
        print('Params: ' + params)