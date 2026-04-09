import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.resnet import resnet50
from nets.vgg import VGG16

# ========== 原有的辅助函数（保留不动） ==========
#把特征图的某些通道顺序重新打乱
def shuffle_mid(mid):
    mid_s = torch.ones_like(mid)
    channels = mid.shape[1]
    if channels <= 16:
        return mid
    mid_s[:] = mid[:]
    groups = int(channels / 16)
    index = [i for i in range(2*groups)]
    tmp = 0
    for i in range(groups):
        for j in range(2):
            index[tmp] = i + j*groups
            tmp = tmp + 1
    for i in range(len(index)):
        j = index[i]
        mid[:,i*8:i*8+8,:,:] = mid_s[:,j*8:j*8+8,:,:]
    return mid

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x

class BaseConv(nn.Module):
    def __init__(self, in_channel, out_channel, groups, ratio=1, last=0):
        super(BaseConv, self).__init__()
        self.groups = groups
        self.ratio = ratio
        self.last = last
        self.cheap_conv = nn.Conv2d(in_channel, in_channel*ratio, kernel_size=3, stride=1, padding=1, groups=in_channel, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel*ratio)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channel*(ratio+1), out_channel, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.cheap_conv(x)
        out = self.bn1(out)
        out = self.relu1(out)
        mid = torch.cat((out, x), dim=1)
        mid = shuffle_mid(mid)
        out = self.conv(mid)
        out = self.bn2(out)
        if self.last == 0:
            out = self.relu2(out)
        if self.groups > 1:
            out = channel_shuffle(out, self.groups)
        return out

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    def forward(self, x):
        return self.conv(x)

class AttentionGate(nn.Module):
    def __init__(self, g_channels, x_channels, inter_channels):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UnetUp(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout_rate=0.25):
        super(UnetUp, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.att = AttentionGate(
            g_channels=in_channels,
            x_channels=skip_channels,
            inter_channels=max(1, min(in_channels, skip_channels) // 2)
        )
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.att(x1, x2)
        x = torch.cat([x2, x1], dim=1)
        x = self.dropout(x)
        return self.conv(x)

# ========== 新增：CBAM 和 Unet_DNA（融合 DNANet 思想） ==========
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class DoubleConvWithCBAM(nn.Module):
    def __init__(self, in_channels, out_channels, use_residual=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()
        self.use_residual = use_residual
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        if self.use_residual:
            out = out + residual
        out = self.relu(out)
        return out

class DenseUpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels_list, out_channels):
        super().__init__()
        total_channels = in_channels + sum(skip_channels_list)
        self.conv = DoubleConvWithCBAM(total_channels, out_channels, use_residual=False)
    def forward(self, x, skip_feats):
        cat_list = [x] + skip_feats
        cat = torch.cat(cat_list, dim=1)
        return self.conv(cat)

class Unet_DNA(nn.Module):
    def __init__(self, num_classes=1, deep_supervision=True):
        super(Unet_DNA, self).__init__()
        self.deep_supervision = deep_supervision
        # Encoder
        self.enc1 = DoubleConvWithCBAM(1, 64, use_residual=False)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConvWithCBAM(64, 128, use_residual=False)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConvWithCBAM(128, 256, use_residual=False)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConvWithCBAM(256, 512, use_residual=False)
        # Decoder with dense skip connections
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.dec3 = DenseUpBlock(in_channels=256, skip_channels_list=[256, 128, 64], out_channels=128)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec2 = DenseUpBlock(in_channels=128, skip_channels_list=[128, 64], out_channels=64)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec1 = DenseUpBlock(in_channels=64, skip_channels_list=[64], out_channels=64)
        # Multi-scale aggregation
        self.final_conv = nn.Sequential(
            nn.Conv2d(64 + 64 + 128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        # Deep supervision
        if self.deep_supervision:
            self.ds3 = nn.Conv2d(128, num_classes, kernel_size=1)
            self.ds2 = nn.Conv2d(64, num_classes, kernel_size=1)
            self.ds1 = nn.Conv2d(64, num_classes, kernel_size=1)
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        # Decoder 3
        up4 = self.up4(e4)
        e2_down = F.interpolate(e2, size=up4.shape[2:], mode='bilinear', align_corners=True)
        e1_down = F.interpolate(e1, size=up4.shape[2:], mode='bilinear', align_corners=True)
        d3 = self.dec3(up4, [e3, e2_down, e1_down])
        # Decoder 2
        up3 = self.up3(d3)
        e1_down2 = F.interpolate(e1, size=up3.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.dec2(up3, [e2, e1_down2])
        # Decoder 1
        up2 = self.up2(d2)
        d1 = self.dec1(up2, [e1])
        # Multi-scale aggregation
        d2_up = F.interpolate(d2, size=d1.shape[2:], mode='bilinear', align_corners=True)
        d3_up = F.interpolate(d3, size=d1.shape[2:], mode='bilinear', align_corners=True)
        concat_multi = torch.cat([d1, d2_up, d3_up], dim=1)
        final = self.final_conv(concat_multi)
        if self.deep_supervision:
            ds3 = F.interpolate(self.ds3(d3), size=final.shape[2:], mode='bilinear', align_corners=False)
            ds2 = F.interpolate(self.ds2(d2), size=final.shape[2:], mode='bilinear', align_corners=False)
            ds1 = self.ds1(d1)
            return final, ds1, ds2, ds3
        else:
            return final

# ========== 修改后的 Unet 类（保持接口不变） ==========
class Unet(nn.Module):
    def __init__(self, num_classes=1, pretrained=False, backbone='myunet'):
        super(Unet, self).__init__()
        if backbone not in ['vgg', 'resnet50', 'myunet']:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50, myunet.'.format(backbone))

        if backbone == 'myunet':
            # 使用融合 DNANet 思想的 Unet_DNA
            self.net = Unet_DNA(num_classes=num_classes, deep_supervision=True)
        else:
            # 保留原有的 VGG 和 ResNet50 分支
            if backbone == 'vgg':
                self.vgg = VGG16(pretrained=pretrained)
                decoder_in_channels = [128, 256, 512, 512]
                skip_channels = [64, 128, 256, 512]
            elif backbone == 'resnet50':
                self.resnet = resnet50(pretrained=pretrained)
                decoder_in_channels = [128, 256, 512, 2048]
                skip_channels = [64, 256, 512, 1024]
            out_filters = [64, 128, 256, 512]

            self.up_concat4 = UnetUp(
                in_channels=decoder_in_channels[3],
                skip_channels=skip_channels[3],
                out_channels=out_filters[3]
            )
            self.up_concat3 = UnetUp(
                in_channels=decoder_in_channels[2],
                skip_channels=skip_channels[2],
                out_channels=out_filters[2]
            )
            self.up_concat2 = UnetUp(
                in_channels=decoder_in_channels[1],
                skip_channels=skip_channels[1],
                out_channels=out_filters[1]
            )
            self.up_concat1 = UnetUp(
                in_channels=decoder_in_channels[0],
                skip_channels=skip_channels[0],
                out_channels=out_filters[0]
            )
            if backbone == 'resnet50':      # ← 这里缩进应与上面的 if/elif 对齐（少缩进一级）
                self.up_conv = nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                    nn.ReLU(),
                )
            else:
                self.up_conv = None

            self.final = nn.Conv2d(out_filters[0], num_classes, 1)

            self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == 'myunet':
            return self.net(inputs)
        else:
            if self.backbone == 'vgg':
                [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
            elif self.backbone == 'resnet50':
                [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)


            up4 = self.up_concat4(feat5, feat4)
            up3 = self.up_concat3(up4, feat3)
            up2 = self.up_concat2(up3, feat2)
            up1 = self.up_concat1(up2, feat1)

            if self.up_conv is not None:
                up1 = self.up_conv(up1)

            final = self.final(up1)
            return final

    def freeze_backbone(self):
        if self.backbone == 'myunet':
            for param in self.net.parameters():
                param.requires_grad = False
            # 让最终分类层可训练
            for param in self.net.final_conv.parameters():
                param.requires_grad = True
            if self.net.deep_supervision:
                for param in self.net.ds1.parameters():
                    param.requires_grad = True
                for param in self.net.ds2.parameters():
                    param.requires_grad = True
                for param in self.net.ds3.parameters():
                    param.requires_grad = True
        elif self.backbone == 'vgg':
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == 'resnet50':
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == 'myunet':
            for param in self.net.parameters():
                param.requires_grad = True
        elif self.backbone == 'vgg':
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == 'resnet50':
            for param in self.resnet.parameters():
                param.requires_grad = True


class NoiseSuppressionBranch(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, out_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.down = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Conv2d(base_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder(x)
        b = self.bottleneck(self.down(e1))
        u = self.up(b)
        return self.head(torch.cat([u, e1], dim=1))


class DualBranchUnet(nn.Module):
    def __init__(
        self,
        num_classes=1,
        det_backbone='myunet',
        det_pretrained=False,
        use_unet_dna=True,
        fusion_alpha_init=1.0
    ):
        super().__init__()
        if use_unet_dna:
            self.det_branch = Unet_DNA(num_classes=num_classes, deep_supervision=True)
        else:
            self.det_branch = Unet(num_classes=num_classes, pretrained=det_pretrained, backbone=det_backbone)
        self.noise_branch = NoiseSuppressionBranch(in_channels=1, base_channels=32, out_channels=num_classes)
        self.fusion_alpha = nn.Parameter(torch.tensor(float(fusion_alpha_init)))
        self.noise_act = nn.Sigmoid()

    def _fuse(self, det_logit, noise_logit):
        return det_logit - self.fusion_alpha * self.noise_act(noise_logit)

    def forward(self, x):
        det_outputs = self.det_branch(x)
        if isinstance(det_outputs, tuple):
            det_main, *det_aux = det_outputs
        else:
            det_main, det_aux = det_outputs, []

        noise_main = self.noise_branch(x)
        final_main = self._fuse(det_main, noise_main)

        if not det_aux:
            return final_main, final_main, final_main, final_main

        fused_aux = []
        for aux_logit in det_aux:
            resized_noise = F.interpolate(noise_main, size=aux_logit.shape[2:], mode='bilinear', align_corners=False)
            fused_aux.append(self._fuse(aux_logit, resized_noise))
        return (final_main, *fused_aux)
# class unetUp(nn.Module):
#     def __init__(self, in_size, out_size):
#         super(unetUp, self).__init__()
#         self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
#         self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
#         self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
#         self.relu   = nn.ReLU(inplace = True)

#     def forward(self, inputs1, inputs2):
#         outputs = torch.cat([inputs1, self.up(inputs2)], 1)
#         outputs = self.conv1(outputs)
#         outputs = self.relu(outputs)
#         outputs = self.conv2(outputs)
#         outputs = self.relu(outputs)
#         return outputs

# class Unet(nn.Module):
#     def __init__(self, num_classes = 21, pretrained = False, backbone = 'vgg'):
#         super(Unet, self).__init__()
#         if backbone == 'vgg':
#             self.vgg    = VGG16(pretrained = pretrained)
#             in_filters  = [192, 384, 768, 1024]
#         elif backbone == "resnet50":
#             self.resnet = resnet50(pretrained = pretrained)
#             in_filters  = [192, 512, 1024, 3072]
#         else:
#             raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
#         out_filters = [64, 128, 256, 512]

#         # upsampling
#         # 64,64,512
#         self.up_concat4 = unetUp(in_filters[3], out_filters[3])
#         # 128,128,256
#         self.up_concat3 = unetUp(in_filters[2], out_filters[2])
#         # 256,256,128
#         self.up_concat2 = unetUp(in_filters[1], out_filters[1])
#         # 512,512,64
#         self.up_concat1 = unetUp(in_filters[0], out_filters[0])

#         if backbone == 'resnet50':
#             self.up_conv = nn.Sequential(
#                 nn.UpsamplingBilinear2d(scale_factor = 2), 
#                 nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
#                 nn.ReLU(),
#                 nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
#                 nn.ReLU(),
#             )
#         else:
#             self.up_conv = None

#         self.final = nn.Conv2d(out_filters[0], num_classes, 1)

#         self.backbone = backbone

#     def forward(self, inputs):
#         if self.backbone == "vgg":
#             [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
#         elif self.backbone == "resnet50":
#             [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

#         up4 = self.up_concat4(feat4, feat5)
#         up3 = self.up_concat3(feat3, up4)
#         up2 = self.up_concat2(feat2, up3)
#         up1 = self.up_concat1(feat1, up2)

#         if self.up_conv != None:
#             up1 = self.up_conv(up1)

#         final = self.final(up1)
        
#         return final

#     def freeze_backbone(self):
#         if self.backbone == "vgg":
#             for param in self.vgg.parameters():
#                 param.requires_grad = False
#         elif self.backbone == "resnet50":
#             for param in self.resnet.parameters():
#                 param.requires_grad = False

#     def unfreeze_backbone(self):
#         if self.backbone == "vgg":
#             for param in self.vgg.parameters():
#                 param.requires_grad = True
#         elif self.backbone == "resnet50":
#             for param in self.resnet.parameters():
#                 param.requires_grad = True
