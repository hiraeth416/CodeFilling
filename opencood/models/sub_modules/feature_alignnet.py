
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import DeformConv2dPack as dconv2d
from timm.models.layers import DropPath
from opencood.models.sub_modules.cbam import BasicBlock
from opencood.models.sub_modules.feature_alignnet_modules import SCAligner, Res1x1Aligner, \
    Res3x3Aligner, Res3x3Aligner, CBAM, ConvNeXt, FANet, SDTAAgliner
import numpy as np



class DeformAlignNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        model_name = args['core_method']
        if model_name == "resnet1x1":
            self.model = Resnet1x1(args['args'], deform = True)
        elif model_name == "resnet3x3":
            self.model = Resnet3x3(args['args'], deform = True)
        elif model_name == "cbam":
            self.model = CBAM(args['args'])
        elif model_name == "sdta":
            self.model = SDTA(args['args'], deform=True)
        elif model_name == "identity":
            self.model = nn.Identity()

    def forward(self, x):
        return self.model(x)
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class XCA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # -------------------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        # ------------------
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}


class ConvEncoder(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=4, kernel_size=1, deformable=False):
        super().__init__()
        if not deformable:
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        else:
            self.dwconv = dconv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class SDTAEncoder(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=4,
                 use_pos_emb=False, num_heads=4, qkv_bias=True, attn_drop=0., drop=0., num_conv=2, deformable=False):
        super().__init__()
        width = dim
        convs = []
        if not deformable:
            for i in range(num_conv):
                convs.append(nn.Conv2d(dim, dim, kernel_size=1, padding=0, groups=width))
                # convs.append(nn.BatchNorm2d(dim))
                convs.append(nn.ReLU())
        else:
            for i in range(num_conv):
                convs.append(dconv2d(dim, dim, kernel_size=1, padding=0, groups=width))
                # convs.append(nn.BatchNorm2d(dim))
                convs.append(nn.ReLU())
        self.convs = nn.Sequential(*convs)


        self.norm_xca = LayerNorm(dim, eps=1e-6)
        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        self.xca = XCA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()  # TODO: MobileViT is using 'swish'
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        x = self.convs(x)

        # XCA
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        x = x + self.drop_path(self.gamma_xca * self.xca(self.norm_xca(x)))
        x = x.reshape(B, H, W, C)

        # Inverted Bottleneck
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)

        return x



class SDTA(nn.Module):
    def __init__(self, args, deform):
        super().__init__()
        in_ch = args['in_ch']
        self.model = nn.ModuleList()

        for i in range(args['layer_num']):
            self.model.append(ConvEncoder(dim=in_ch, deformable=deform))
            self.model.append(SDTAEncoder(dim=in_ch, deformable=deform))
            
    def forward(self, x):
        for m in self.model:
            x = m(x)
        return x


class ResidualBlock(nn.Module):  
    def __init__(self, in_channels, out_channels, use_1x1conv=False, kernel_size=3, deform=False):
        super(ResidualBlock, self).__init__()
        if kernel_size == 3:
            padding = 1
            stride = 1
        elif kernel_size == 1:
            padding = 0
            stride = 1
        else:
            raise("Not Supported")

        if not deform:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        else:
            self.conv1 = dconv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
            self.conv2 = dconv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)

        # 1x1conv来升维
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


class Resnet3x3(nn.Module):
    def __init__(self, args, deform=False):
        super().__init__()
        in_ch = args['in_ch']
        layernum = args['layer_num']
        model_list = nn.ModuleList()
        for _ in range(layernum):
            model_list.append(ResidualBlock(in_ch, in_ch, kernel_size=3, deform=deform))
 
        self.model = nn.Sequential(*model_list)

    def forward(self, x):
        return self.model(x)


class Resnet1x1(nn.Module):
    def __init__(self, args, deform=False):
        super().__init__()
        in_ch = args['in_ch']
        layernum = args['layer_num']
        model_list = nn.ModuleList()
        for _ in range(layernum):
            model_list.append(ResidualBlock(in_ch, in_ch, kernel_size=1, deform=deform))
 
        self.model = nn.Sequential(*model_list)

    def forward(self, x):
        return self.model(x)

class CBAM(nn.Module):
    def __init__(self, args):
        super().__init__()
        in_ch = args['in_ch']
        layernum = args['layer_num']
        model_list = nn.ModuleList()
        for _ in range(layernum):
            model_list.append(BasicBlock(in_ch, in_ch))
 
        self.model = nn.Sequential(*model_list)

    def forward(self, x):
        return self.model(x)


class AlignNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        model_name = args['core_method']
        
        if model_name == "scaligner":
            self.channel_align = SCAligner(args['args'])
        elif model_name == "resnet1x1":
            self.channel_align = Res1x1Aligner(args['args'])
        elif model_name == "resnet3x3":
            self.channel_align = Res3x3Aligner(args['args'])
        elif model_name == "sdta":
            self.channel_align = SDTAAgliner(args['args'])
        elif model_name == "cbam":
            self.channel_align = CBAM(args['args'])
        elif model_name == "convnext":
            self.channel_align = ConvNeXt(args['args'])
        elif model_name == "fanet":
            self.channel_align = FANet(args['args'])
        elif model_name == 'identity':
            self.channel_align = nn.Identity()

        self.spatial_align_flag = args.get("spatial_align", False)
        if self.spatial_align_flag:
            warpnet_indim = args['args']['warpnet_indim']
            dim = args['args']['dim']
            self.teacher = args['args']['teacher']
            setattr(self, "warpnet", 
                nn.Sequential(
                nn.Conv2d(warpnet_indim, warpnet_indim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(warpnet_indim),
                nn.ReLU(),
                nn.Conv2d(warpnet_indim, dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                nn.Conv2d(dim, 2, kernel_size=3, stride=1, padding=1),
                )
            )
            self.theta_identity = torch.tensor([[[1.,0.,0.],[0.,1.,0.]]])

        self.count = 0 # debug

    def forward(self, x):
        return self.channel_align(x)

    def spatail_align(self, student_feature, teacher_feature, physical_dist):
        physical_offset = self.warpnet(torch.cat([student_feature, teacher_feature], dim=1)).permute(0,2,3,1) # N, H, W, 2, unit is meter.
        mask = torch.any(teacher_feature != 0, dim=1)
        physical_offset *= mask.unsqueeze(-1)
        relative_offset = physical_offset * torch.tensor([2./physical_dist[0], 2./physical_dist[1]], device=physical_offset.device)  # N, H, W, 2
        warp_field = relative_offset + \
            torch.nn.functional.affine_grid(self.theta_identity.expand(student_feature.shape[0], 2, 3), student_feature.shape).to(relative_offset.device)
        spataial_aligned_feature = torch.nn.functional.grid_sample(student_feature, warp_field)

        # self.visualize_offset(physical_offset, warp_field, student_feature, spataial_aligned_feature, teacher_feature)
        return spataial_aligned_feature

    def visualize_offset(self, physical_offset, warp_field, feature_before, feature_after, teacher_feature):
        """
        physical_offset: shape [N, H, W, 2]
        warp_field: shape [N, H, W, 2]
        feaure_before: [N, C, H, W]
        feature_after: [N, C, H, W]
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        import os
        N = physical_offset.shape[0]
        print(physical_offset.shape)
        
        save_path = "opencood/logs_HEAL/vislog"
        file_idx = self.count
        self.count += 1

        physical_offsets_save_path = os.path.join(save_path, "physical_offsets")
        vmax = physical_offset.max()
        print(f"physical offset max: {vmax}")
        if not os.path.exists(physical_offsets_save_path):
            os.mkdir(physical_offsets_save_path)
        physical_offset = physical_offset.detach().cpu().numpy()
        warp_field = warp_field.detach().cpu().numpy()
        for i in range(N):
            sns.heatmap(physical_offset[i,:,:,0], cmap="vlag", vmin=-vmax*0.8, vmax=vmax*0.8, square=True)
            plt.axis('off')
            plt.savefig(os.path.join(physical_offsets_save_path, "{}_{}_physical_x.png".format(file_idx, i)), dpi=500)
            plt.close()

            sns.heatmap(physical_offset[i,:,:,1], cmap="vlag", vmin=-vmax*0.8, vmax=vmax*0.8, square=True)
            plt.axis('off')
            plt.savefig(os.path.join(physical_offsets_save_path, "{}_{}_physical_y.png".format(file_idx, i)), dpi=500)
            plt.close()

            sns.heatmap(warp_field[i,:,:,0], cmap="vlag", square=True)
            plt.axis('off')
            plt.savefig(os.path.join(physical_offsets_save_path, "{}_{}_warpfield_x.png".format(file_idx, i)), dpi=500)
            plt.close()

            sns.heatmap(warp_field[i,:,:,1], cmap="vlag", square=True)
            plt.axis('off')
            plt.savefig(os.path.join(physical_offsets_save_path, "{}_{}_warpfield_y.png".format(file_idx, i)), dpi=500)
            plt.close()

        spatial_feature_save_path = os.path.join(save_path, "spatial_feature")
        if not os.path.exists(spatial_feature_save_path):
            os.mkdir(spatial_feature_save_path)
        feature_before = feature_before.detach().cpu().numpy()
        feature_after = feature_after.detach().cpu().numpy()
        teacher_feature = teacher_feature.detach().cpu().numpy()
        for i in range(N):
            channel = np.random.randint(64)
            plt.imshow(feature_before[i, channel])
            plt.axis("off")
            plt.colorbar()
            plt.savefig(os.path.join(spatial_feature_save_path, "{}_{}_before.png".format(file_idx, i)), dpi=500)
            plt.close()

            plt.imshow(feature_after[i, channel])
            plt.axis("off")
            plt.colorbar()
            plt.savefig(os.path.join(spatial_feature_save_path, "{}_{}_spaligned.png".format(file_idx, i)), dpi=500)
            plt.close()
            
            plt.imshow(teacher_feature[i, channel])
            plt.axis("off")
            plt.colorbar()
            plt.savefig(os.path.join(spatial_feature_save_path, "{}_{}_teacher.png".format(file_idx, i)), dpi=500)
            plt.close()