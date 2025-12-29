import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
import time
from einops import rearrange

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)
    
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)

class DMSA(nn.Module): # scale的取值最大的理论值是10
    def __init__(self, c, num_heads=4, scale = 4, bias=False, attn_dropout=0., residual_scale_init=0.0):
        super(DMSA, self).__init__()

        self.num_heads = num_heads
        # self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.log_temperature = nn.Parameter(torch.zeros(num_heads, 1, 1))

        self.scale = scale
        k_size = 2 * scale + 1

        self.AvgDownSample = nn.Conv2d(c,c,kernel_size=k_size, stride=scale, padding=k_size//2, groups=c)
        
        # 针对下采样之后的特征
        self.to_qk = nn.Conv2d(c, 2*c, kernel_size=1, bias=bias)
        self.qk_dwconv = nn.Conv2d(c*2, c*2, kernel_size=3, stride=1, padding=1, groups=c*2, bias=bias)
        
        # 针对原特征大小
        self.to_v = nn.Conv2d(c, c, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(c, c, kernel_size=1, bias=bias)

        # regularization
        self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity()

        # residual scaling factor (start from 0 optionally)
        self.res_scale = nn.Parameter(torch.ones(1) * residual_scale_init)
    
    def forward(self, x):
        b,c,h,w = x.shape
        v = self.to_v(x)

        x_d = self.AvgDownSample(x)
        qk = self.to_qk(x_d)
        qk = self.qk_dwconv(qk)
        q, k = qk.chunk(2, dim=1)

        q = rearrange(q, 'b (heads hdim) h w -> b heads hdim (h w)', heads=self.num_heads) # dim=c*head_num 
        k = rearrange(k, 'b (heads hdim) h w -> b heads hdim (h w)', heads=self.num_heads)
        v = rearrange(v, 'b (heads hdim) h w -> b heads hdim (h w)', heads=self.num_heads) # q, k, v: [b, heads, hdim, hw]

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1) # q、k：[b, num, c, hw]

        temperature = torch.exp(self.log_temperature).view(1, self.num_heads, 1, 1)

        attn = ( q @ k.transpose(-2, -1)) * temperature # [b, num,c, c]
        attn = attn.softmax(dim=-1) 
        attn = self.attn_dropout(attn)

        out = (attn @ v ) # [b, num, c, hw]
        
        out = rearrange(out, 'b head hdim (h w) -> b (head hdim) h w', head=self.num_heads, h=h, w=w) # [b, c, h, w]

        out = self.project_out(out) # [b, dim, h, w]
        out = x + self.res_scale * out
        return out
    
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        return x1 * x2
        
class DAB(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, heads=4, scale=4, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand

        self.conv1 = nn.Conv2d(c, dw_channel, 1, padding=0, stride=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, stride=1, groups=dw_channel,bias=True)
        self.conv3 = nn.Conv2d(c, c, 1, stride=1, bias=True)
        self.attn = DMSA(c,num_heads=heads,scale=scale)
        self.sg = SimpleGate()

        ffn_channel = c * FFN_Expand
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, stride=1, bias=True)
        self.conv5 = nn.Conv2d(c, c, 1, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        ############GCAM--start#####################
        x = self.norm1(x)
        x = self.conv1(x) #
        x = self.conv2(x)
        x = self.sg(x) # 激活函数引入非线性
        x = self.attn(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        ###########GCAM------end#####################
        ###########FFN----start######################
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma
        ##########FFN--END###################
class CFAM(nn.Module):
    def __init__(self, c, width=32, layernums = 3):
        super().__init__()
        self.branch_convs = nn.ModuleList()
        
        dim = width
        for _ in range(layernums):
            self.branch_convs.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim//2, kernel_size=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(dim//2, dim//2, kernel_size=3, padding=1, groups=dim//2, bias=False),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(dim//2, dim, 1, bias=False),
                    nn.Sigmoid()
                )
            )
            dim = dim * 2
        
        # 1 2 4 8
        k = 0    
        for idx in range(layernums):
            k = k + 2**idx
        
        self.conv1x1 = nn.Conv2d(k * width, c, 1)
        self.conv_final = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, 1),
        )

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, enfea_list, size=None):
        branches = []
        for i, f in enumerate(enfea_list):
            f_up = F.interpolate(f, size=size, mode='bilinear', align_corners=False)
            b = self.branch_convs[i](f_up) * f_up 
            branches.append(b)

        x = torch.cat(branches, dim=1)

        x = self.conv1x1(x)
        x = self.lrelu(x)
        x = self.conv_final(x)
        x = x * self.sca(x)
        return x

class Enhance_model(nn.Module):

    def __init__(self, img_channel=3, 
                 width=40, 
                 middle_blk_num=1, 
                 enc_blk_nums=[1, 1,4], 
                 dec_blk_nums=[1,1,2], 
                 layernums = 3,
                 num_heads=4,
                 scale=4,
                 drop_out_rate=0.):
        
        super().__init__()

        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, stride=1, groups=1,bias=True)

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.encfea = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            blocks = nn.ModuleList([DAB(c=chan, heads=num_heads, scale=scale, drop_out_rate=drop_out_rate) for _ in range(num)])
            self.encoders.append(blocks)    
            self.downs.append( nn.Conv2d(chan, 2*chan, 2, 2) )
            chan = chan * 2

        self.middle_blks = nn.ModuleList([DAB(c=chan, heads=num_heads, scale=scale, drop_out_rate=drop_out_rate) for _ in range(middle_blk_num)])
        
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )

            chan = chan // 2
            self.encfea.append(CFAM(chan, width=width, layernums = layernums))
            self.decoders.append(
                nn.ModuleList([DAB(c=chan, heads=num_heads, scale=scale, drop_out_rate=drop_out_rate) for _ in range(num)])
            )

        self.padder_size = 2 ** len(self.encoders)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1, stride=1, groups=1, bias=True)

        self.beta = nn.Parameter(torch.zeros((1, img_channel, 1, 1)), requires_grad=True)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = inp
        x = self.intro(x)
        encs = []
        for encoder_block, downx in zip(self.encoders, self.downs):
            for encoder in encoder_block:
                x = encoder(x)
            encs.append(x)
            x = downx(x)
                
        for blk in self.middle_blks:
            x = blk(x)
        
        for decoder_blocks, up, enahnce ,en_skip in zip(self.decoders, self.ups,self.encfea, encs[::-1]):
            x = up(x)
            _, _, h, w = x.shape
            x = x + enahnce(encs, size=(h,w)) + en_skip 
            for decoder in decoder_blocks:
                x = decoder(x)
        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

# dropout 一般取值0.05-0.2
    
if __name__ == '__main__':
    # 模型配置
    img_channel = 3
    width = 40
    enc_blks = [1, 1, 4]
    middle_blk_num = 1
    dec_blks = [1, 1, 2]
    layernums = 3
    heads = 4
    scale = 4

    # 实例化模型
    model = Enhance_model(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                         enc_blk_nums=enc_blks, dec_blk_nums=dec_blks, layernums=layernums,num_heads=heads,scale=scale).to('cuda')
    
    input = torch.rand(1, 3, 256, 256).to('cuda')  
    torch.cuda.synchronize()
    model.eval()
    time_start = time.time()
    _ = model(input)
    time_end = time.time()
    torch.cuda.synchronize()
    time_sum = time_end - time_start
    print(f"Time: {time_sum}")
    
    n_param = sum([p.nelement() for p in model.parameters()])  
    n_paras = f"n_paras: {(n_param/2**20)}M\n"
    print(n_paras)
    
    macs, params = profile(model, inputs=(input,)) 
    print(f'FLOPs:{macs/(2**30)}G')
