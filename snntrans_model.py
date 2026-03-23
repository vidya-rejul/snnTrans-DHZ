# SNN libraries
from spikingjelly.activation_based import neuron, surrogate, functional, layer

# torch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# import general libraries

import numpy as np

import kornia.color as K

np.int = int

# model parameters
batch_size = 1
input_height = 512
input_width = 512

# neuron parameters
tau=2.0
v_reset=0.0
v_threshold=0.25
surro_grad=surrogate.Sigmoid(alpha=4.0)
timesteps=4

# Adaptive LIF for spike coding
class ALIFNode(neuron.BaseNode):
    def __init__(self, tau: float | torch.Tensor = 2., v_threshold: float = 1., *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.v_threshold = nn.Parameter(torch.tensor(v_threshold, dtype=torch.float, requires_grad=True))
        tau = torch.tensor(tau, dtype=torch.float)
        self.register_buffer("tau", tau)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + (x - (self.v - self.v_reset)) / self.tau
    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)

# spike coding
class spikecode_rgb(nn.Module):
  def __init__(self,timesteps):
    super().__init__()
    self.timesteps=timesteps

    # spike conversion
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    self.lif1 = ALIFNode(tau=tau,v_threshold=v_threshold)
    self.down1= nn.MaxPool2d(2,2) # kernel=2,stride=2,Padding=0

  def forward(self, x):
    batch_size, _, height, width = x.shape # B,3,H,W
    # encoder
    conv1_out = self.conv1(x)
    conv1_out_reshaped=conv1_out.unsqueeze(0).repeat(self.timesteps,1,1,1,1) #incorporated the temporal dimension

    lif1_out = []

    for t in range(self.timesteps):
      self.lif1.neuronal_charge(conv1_out_reshaped[t])  # Charge neuron
      spike = self.lif1.neuronal_fire()  # Fire spike
      lif1_out.append(spike)

    lif1_out = torch.stack(lif1_out, dim=1)  # B, T, 16, H, W
    down1_out = self.down1(lif1_out.view(self.timesteps * batch_size, -1, height, width)) # B*T, 16, H/2, W/2
    down1_out = down1_out.view(self.timesteps, batch_size, -1, height // 2, width // 2) # T, B, 16, H/2, W/2 #unfold to temporal

    return down1_out

class spikecode_lab(nn.Module):
  def __init__(self,timesteps):
    super().__init__()
    self.timesteps=timesteps

    # spike conversion
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    self.lif1 = ALIFNode(tau=tau,v_threshold=v_threshold)
    self.down1= nn.MaxPool2d(2,2) # kernel=2,stride=2,Padding=0

  def forward(self, x):
    batch_size, _, height, width = x.shape # B,3,H,W
    # encoder
    conv1_out = self.conv1(x)
    conv1_out_reshaped=conv1_out.unsqueeze(0).repeat(self.timesteps,1,1,1,1) #incorporated the temporal dimension

    lif1_out = []

    for t in range(self.timesteps):
      self.lif1.neuronal_charge(conv1_out_reshaped[t])  # Charge neuron
      spike = self.lif1.neuronal_fire()  # Fire spike
      lif1_out.append(spike)

    lif1_out = torch.stack(lif1_out, dim=1)  # B, T, 16, H, W
    down1_out = self.down1(lif1_out.view(self.timesteps * batch_size, -1, height, width)) # B*T, 16, H/2, W/2
    down1_out = down1_out.view(self.timesteps, batch_size, -1, height // 2, width // 2) # T, B, 16, H/2, W/2 #unfold to temporal

    return down1_out

# Background light estimation
class BL_est_conv(nn.Module):
    def __init__(self, timesteps):
        super().__init__()
        self.timesteps = timesteps

        self.conv1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.lif1 = ALIFNode(tau=tau, v_threshold=v_threshold)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.lif2 = ALIFNode(tau=tau, v_threshold=v_threshold)

        self.convf = nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=2, stride=2)  # same as input size
        self.BL_liff = neuron.LIFNode(step_mode='m', tau=tau, decay_input=True, v_reset=v_reset, v_threshold=np.inf,
                                      surrogate_function=surro_grad, backend='cupy')

    def forward(self, x):
        T, batch_size, _, height, width = x.shape  # T, B, 32, H/2, W/2 (downsampled feature map)

        conv1_out = self.conv1(x.view(T * batch_size, -1, height, width))  # T*B, 16, H/2, W/2
        conv1_out_reshaped = conv1_out.view(T, batch_size, -1, height, width)  # T, B, 16, H/2, W/2
        lif1_out = []
        for t in range(self.timesteps):
            self.lif1.neuronal_charge(conv1_out_reshaped[t])  # Charge neuron
            spike = self.lif1.neuronal_fire()  # Fire spike
            lif1_out.append(spike)
        lif1_out = torch.stack(lif1_out, dim=1)  # B, T, 16, H/2, W/2

        conv2_out = self.conv2(lif1_out.view(T * batch_size, -1, height, width))  # T*B, 16, H/2, W/2
        conv2_out_reshaped = conv2_out.view(T, batch_size, -1, height, width)  # T, B, 16, H/2, W/2
        lif2_out = []
        for t in range(self.timesteps):
            self.lif2.neuronal_charge(conv2_out_reshaped[t])  # Charge neuron
            spike = self.lif2.neuronal_fire()  # Fire spike
            lif2_out.append(spike)
        lif2_out = torch.stack(lif2_out, dim=1)  # B, T, 16, H/2, W/2

        convf_out = self.convf(lif2_out.view(T * batch_size, -1, height, width))  # T*B, 3, H, W
        BL_liff_out = self.BL_liff(
            convf_out.view(self.timesteps, batch_size, -1, 2 * height, 2 * width))  # T, B, 3, H, W

        return BL_liff_out, self.BL_liff.v

# Spiking Transformer-based H(x) Estimation
class BNAndPadLayer(nn.Module):
    def __init__(self, pad_pixels, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (self.bn.bias.detach() - self.bn.running_mean * self.bn.weight.detach()/ torch.sqrt(self.bn.running_var + self.bn.eps))
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)

            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0 : self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels :, :] = pad_values
            output[:, :, :, 0 : self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels :] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps

class RepConv(nn.Module):
    def __init__(self, in_channel, out_channel, bias=False):
        super().__init__()
        # hidden_channel = in_channel
        conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=1)
        bn = BNAndPadLayer(pad_pixels=1, num_features=in_channel)
        conv3x3 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 0, groups=in_channel, bias=False),
                                nn.Conv2d(in_channel, out_channel, 1, 1, 0, groups=1, bias=False),
                                nn.BatchNorm2d(out_channel))
        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)

class attn(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
        super().__init__()
        assert (dim % num_heads == 0), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        self.head_lif = ALIFNode(tau=tau,v_threshold=v_threshold)

        self.q_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        self.k_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        self.v_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        self.q_lif = ALIFNode(tau=tau,v_threshold=v_threshold)

        self.k_lif = ALIFNode(tau=tau,v_threshold=v_threshold)

        self.v_lif = ALIFNode(tau=tau,v_threshold=v_threshold)

        self.attn_lif = ALIFNode(tau=tau,v_threshold=v_threshold)

        self.proj_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W

        head_lif_out = []
        for t in range(T):
            self.head_lif.neuronal_charge(x[t])
            spike =  self.head_lif.neuronal_fire()
            head_lif_out.append(spike)
        x = torch.stack(head_lif_out, dim=0) # T, B, C, H, W

        q = self.q_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        k = self.k_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        v = self.v_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)

        q_out = []
        for t in range(T):
            self.q_lif.neuronal_charge(q[t])
            spike =  self.q_lif.neuronal_fire()
            q_out.append(spike)
        q = torch.stack(q_out, dim=0) # T, B, C, H, W
        q = q.flatten(3)
        q = (q.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads)
              .permute(0, 1, 3, 2, 4).contiguous())

        k_out = []
        for t in range(T):
            self.k_lif.neuronal_charge(k[t])
            spike =  self.k_lif.neuronal_fire()
            k_out.append(spike)
        k = torch.stack(k_out, dim=0) # T, B, C, H, W
        k = k.flatten(3)
        k = (k.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads)
              .permute(0, 1, 3, 2, 4).contiguous())

        v_out = []
        for t in range(T):
            self.v_lif.neuronal_charge(v[t])
            spike =  self.v_lif.neuronal_fire()
            v_out.append(spike)
        v = torch.stack(v_out, dim=0) # T, B, C, H, W
        v = v.flatten(3)
        v = (v.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads)
              .permute(0, 1, 3, 2, 4).contiguous())

        x = k.transpose(-2, -1) @ v
        #x = (q @ x) * self.scale
        x = (q @ x)

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        attn_lif_out = []
        for t in range(T):
            self.attn_lif.neuronal_charge(x[t])
            spike =  self.attn_lif.neuronal_fire()
            attn_lif_out.append(spike)
        x = torch.stack(attn_lif_out, dim=0) # T, B, C, N
        x = x.reshape(T, B, C, H, W)
        x = x.flatten(0, 1)
        x = self.proj_conv(x).reshape(T, B, C, H, W)

        return x

class mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0, layer=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_conv = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(out_features)
        self.fc1_lif = ALIFNode(tau=tau,v_threshold=v_threshold)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W

        x= x.flatten(3)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_output, H, W).contiguous()

        fc1_lif_out = []
        for t in range(T):
            self.fc1_lif.neuronal_charge(x[t])
            spike =  self.fc1_lif.neuronal_fire()
            fc1_lif_out.append(spike)
        x = torch.stack(fc1_lif_out,dim=0)

        return x

class snn_trans(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None,
               drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()

        self.attn = attn(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                     attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x

class H_est(nn.Module):
    def __init__(self, timesteps):
        super().__init__()
        self.timesteps = timesteps

        ########################################################################################################
        # encoder_rgb
        self.r_conv1_Hx = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.r_lif1_Hx = ALIFNode(tau=tau, v_threshold=v_threshold)
        self.r_down1_Hx = nn.MaxPool2d(2, 2)  # kernel=2,stride=2,Padding=0

        self.r_conv2_Hx = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.r_lif2_Hx = ALIFNode(tau=tau, v_threshold=v_threshold)
        self.r_down2_Hx = nn.MaxPool2d(2, 2)  # kernel=2,stride=2,Padding=0

        # latent space_rgb
        self.r_snn_trans_Lx = nn.ModuleList([snn_trans(dim=64, num_heads=8) for i in range(6)])  # Lx=4

        ########################################################################################################
        # encoder_lab
        self.l_conv1_Hx = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.l_lif1_Hx = ALIFNode(tau=tau, v_threshold=v_threshold)
        self.l_down1_Hx = nn.MaxPool2d(2, 2)  # kernel=2,stride=2,Padding=0

        self.l_conv2_Hx = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.l_lif2_Hx = ALIFNode(tau=tau, v_threshold=v_threshold)
        self.l_down2_Hx = nn.MaxPool2d(2, 2)  # kernel=2,stride=2,Padding=0

        # latent space_lab
        self.l_snn_trans_Lx = nn.ModuleList([snn_trans(dim=64, num_heads=8) for i in range(6)])  # Lx=4

        # decoder_common
        ########################################################################################################
        self.up1_Hx = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=2, stride=2)
        self.lif_up1_Hx = ALIFNode(tau=tau, v_threshold=v_threshold)

        self.up2_Hx = nn.ConvTranspose2d(in_channels=96, out_channels=16, kernel_size=2, stride=2)
        self.lif_up2_Hx = ALIFNode(tau=tau, v_threshold=v_threshold)

        self.up3_Hx = nn.ConvTranspose2d(in_channels=48, out_channels=3, kernel_size=2, stride=2)
        self.Hx_liff = neuron.LIFNode(step_mode='m', tau=tau, decay_input=True, v_reset=v_reset, v_threshold=np.inf,
                                      surrogate_function=surro_grad, backend='cupy')

    def forward(self, x_rgb, x_lab):
        T, B, C, H, W = x_rgb.shape

        #################################################################################################################################
        # RGB

        r_conv1_Hx_out = self.r_conv1_Hx(x_rgb.view(T * B, -1, H, W))  # T*B, 32, H, W
        r_conv1_Hx_out_reshaped = r_conv1_Hx_out.view(T, B, -1, H, W)  # T, B, 32, H, W
        r_lif1_Hx_out = []
        for t in range(self.timesteps):
            self.r_lif1_Hx.neuronal_charge(r_conv1_Hx_out_reshaped[t])  # Charge neuron
            spike = self.r_lif1_Hx.neuronal_fire()  # Fire
            r_lif1_Hx_out.append(spike)
        r_lif1_Hx_out = torch.stack(r_lif1_Hx_out, dim=0)  # T, B, 32, H, W
        r_down1_Hx_out = self.r_down1_Hx(r_lif1_Hx_out.view(T * B, -1, H, W))  # T*B, 32, H/2, W/2
        r_down1_Hx_out_reshaped = r_down1_Hx_out.view(T, B, -1, H // 2,
                                                      W // 2)  # T, B, 32, H/2, W/2 # To concat_features1

        r_conv2_Hx_out = self.r_conv2_Hx(r_down1_Hx_out)  # T*B, 64, H/2, W/2
        r_conv2_Hx_out_reshaped = r_conv2_Hx_out.view(T, B, -1, H // 2, W // 2)
        r_lif2_Hx_out = []
        for t in range(self.timesteps):
            self.r_lif2_Hx.neuronal_charge(r_conv2_Hx_out_reshaped[t])  # Charge neuron
            spike = self.r_lif2_Hx.neuronal_fire()  # Fire
            r_lif2_Hx_out.append(spike)
        r_lif2_Hx_out = torch.stack(r_lif2_Hx_out, dim=0)
        r_down2_Hx_out = self.r_down2_Hx(r_lif2_Hx_out.view(T * B, -1, H // 2, W // 2))  # T*B, 64, H/4, W/4
        r_down2_Hx_out_reshaped = r_down2_Hx_out.view(T, B, -1, H // 4, W // 4)  # T, B, 64, H/4, W/4

        for blk in self.r_snn_trans_Lx:
            r_snn_trans_Lx_out = blk(r_down2_Hx_out_reshaped)

        ###################################################################################################################################
        # LAB

        l_conv1_Hx_out = self.l_conv1_Hx(x_lab.view(T * B, -1, H, W))  # T*B, 32, H, W
        l_conv1_Hx_out_reshaped = l_conv1_Hx_out.view(T, B, -1, H, W)  # T, B, 32, H, W
        l_lif1_Hx_out = []
        for t in range(self.timesteps):
            self.l_lif1_Hx.neuronal_charge(l_conv1_Hx_out_reshaped[t])  # Charge neuron
            spike = self.l_lif1_Hx.neuronal_fire()  # Fire
            l_lif1_Hx_out.append(spike)
        l_lif1_Hx_out = torch.stack(l_lif1_Hx_out, dim=0)  # T, B, 32, H, W
        l_down1_Hx_out = self.l_down1_Hx(l_lif1_Hx_out.view(T * B, -1, H, W))  # T*B, 32, H/2, W/2
        l_down1_Hx_out_reshaped = l_down1_Hx_out.view(T, B, -1, H // 2,
                                                      W // 2)  # T, B, 32, H/2, W/2 # To concat_features1

        l_conv2_Hx_out = self.l_conv2_Hx(l_down1_Hx_out)  # T*B, 64, H/2, W/2
        l_conv2_Hx_out_reshaped = l_conv2_Hx_out.view(T, B, -1, H // 2, W // 2)
        l_lif2_Hx_out = []
        for t in range(self.timesteps):
            self.l_lif2_Hx.neuronal_charge(l_conv2_Hx_out_reshaped[t])  # Charge neuron
            spike = self.l_lif2_Hx.neuronal_fire()  # Fire
            l_lif2_Hx_out.append(spike)
        l_lif2_Hx_out = torch.stack(l_lif2_Hx_out, dim=0)
        l_down2_Hx_out = self.l_down2_Hx(l_lif2_Hx_out.view(T * B, -1, H // 2, W // 2))  # T*B, 64, H/4, W/4
        l_down2_Hx_out_reshaped = l_down2_Hx_out.view(T, B, -1, H // 4, W // 4)  # T, B, 64, H/4, W/4

        for blk in self.l_snn_trans_Lx:
            l_snn_trans_Lx_out = blk(l_down2_Hx_out_reshaped)

        ####################################################################################################################################
        # combined
        concat_features_trans = torch.cat([r_snn_trans_Lx_out, l_snn_trans_Lx_out], dim=2)  # T, B, 128, H/4, W/4
        down1_reshaped = torch.cat([r_down1_Hx_out_reshaped, l_down1_Hx_out_reshaped], dim=2)  # T, B, 64, H/2, W/2
        x_concat = torch.concat([x_rgb, x_lab], dim=2)  # T, B, 32, H, W

        up1_Hx_out = self.up1_Hx(concat_features_trans.view(T * B, -1, H // 4, W // 4))  # T*B, 32, H/2, W/2
        up1_Hx_out_reshaped = up1_Hx_out.view(T, B, -1, H // 2, W // 2)  # T, B, 32, H/2, W/2
        lif_up1_Hx_out = []
        for t in range(self.timesteps):
            self.lif_up1_Hx.neuronal_charge(up1_Hx_out_reshaped[t])  # Charge neuron
            spike = self.lif_up1_Hx.neuronal_fire()  # Fire
            lif_up1_Hx_out.append(spike)
        lif_up1_Hx_out = torch.stack(lif_up1_Hx_out, dim=0)  # T, B, 32, H/2, W/2

        concat_features1 = torch.cat([down1_reshaped, lif_up1_Hx_out], dim=2)  # T, B, 96, H/2, W/2

        up2_Hx_out = self.up2_Hx(concat_features1.view(T * B, -1, H // 2, W // 2))  # T*B, 16, H, W
        up2_Hx_out_reshaped = up2_Hx_out.view(T, B, -1, H, W)  # T, B, 16, H, W
        lif_up2_Hx_out = []
        for t in range(self.timesteps):
            self.lif_up2_Hx.neuronal_charge(up2_Hx_out_reshaped[t])  # Charge neuron
            spike = self.lif_up2_Hx.neuronal_fire()  # Fire
            lif_up2_Hx_out.append(spike)
        lif_up2_Hx_out = torch.stack(lif_up2_Hx_out, dim=0)  # T, B, 16, H, W

        concat_features2 = torch.cat([x_concat, lif_up2_Hx_out], dim=2)  # T, B, 48, H, W

        up3_Hx_out = self.up3_Hx(concat_features2.view(T * B, -1, H, W))  # T*B, 3, 2H, 2W (same as initial image size)
        Hx_liff_out = self.Hx_liff(up3_Hx_out.view(T, B, -1, 2 * H, 2 * W))  # T, B, 3, 2H, 2W

        return Hx_liff_out, self.Hx_liff.v

# Image Reconstruction
class SIR_est(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, I_x, H_x, B):
        I_x = I_x
        J_x = (H_x * I_x) - (H_x * B) + I_x # Reconstruction based on IFM 
        return J_x

class dhz_net(nn.Module):
  def __init__(self,timesteps):
    super().__init__()
    self.timesteps=timesteps

    self.spikecode_rgb=spikecode_rgb(timesteps) #spike coding
    self.spikecode_lab=spikecode_lab(timesteps) #spike coding
    self.H_est=H_est(timesteps) # H estimator
    self.BL_est_conv=BL_est_conv(timesteps) # background light estimator
    self.SIR_est=SIR_est() # soft image reconstruction

  def forward(self, x):
    x_rgb = x
    x_lab_cnvrt = K.rgb_to_lab(x_rgb)
    xlab_l = x_lab_cnvrt[:, 0:1, :, :]/100
    xlab_a = (x_lab_cnvrt[:, 1:2, :, :] + 128)/100
    xlab_b = (x_lab_cnvrt[:, 2:3, :, :] + 128)/100
    x_lab = torch.cat((xlab_l, xlab_a, xlab_b), dim=1)

    spikecode_rgb = self.spikecode_rgb(x_rgb)
    spikecode_lab = self.spikecode_lab(x_lab)
    spikecode_2BLest = torch.cat((spikecode_rgb, spikecode_lab), dim=2)
    H_s, H_v=self.H_est(spikecode_rgb, spikecode_lab)
    BL_s, BL_v=self.BL_est_conv(spikecode_2BLest)
    J=self.SIR_est(x,H_v,BL_v)
    return J
