import paddle
import paddle.nn as nn
import math


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [(x // 2) for x in k]
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Layer):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2D(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=autopad(k, p),
            groups=g,
            bias_attr=False,
        )
        self.bn = nn.BatchNorm2D(num_features=c2)
        self.act = nn.Mish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Layer):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Layer):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2D(
            in_channels=c1, out_channels=c_, kernel_size=1, stride=1, bias_attr=False
        )
        self.cv3 = nn.Conv2D(
            in_channels=c_, out_channels=c_, kernel_size=1, stride=1, bias_attr=False
        )
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2D(num_features=2 * c_)
        self.act = nn.Mish()
        self.m = nn.Sequential(
            *[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)]
        )

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(paddle.concat(x=(y1, y2), axis=1))))


class BottleneckCSP2(nn.Layer):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super(BottleneckCSP2, self).__init__()
        c_ = int(c2)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2D(
            in_channels=c_, out_channels=c_, kernel_size=1, stride=1, bias_attr=False
        )
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2D(num_features=2 * c_)
        self.act = nn.Mish()
        self.m = nn.Sequential(
            *[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)]
        )

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(self.act(self.bn(paddle.concat(x=(y1, y2), axis=1))))


class VoVCSP(nn.Layer):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(VoVCSP, self).__init__()
        c_ = int(c2)
        self.cv1 = Conv(c1 // 2, c_ // 2, 3, 1)
        self.cv2 = Conv(c_ // 2, c_ // 2, 3, 1)
        self.cv3 = Conv(c_, c2, 1, 1)

    def forward(self, x):
        _, x1 = x.chunk(chunks=2, axis=1)
        x1 = self.cv1(x1)
        x2 = self.cv2(x1)
        return self.cv3(paddle.concat(x=(x1, x2), axis=1))


class SPP(nn.Layer):
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.LayerList(
            sublayers=[nn.MaxPool2D(kernel_size=x, stride=1, padding=x // 2) for x in k]
        )

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(paddle.concat(x=[x] + [m(x) for m in self.m], axis=1))


class SPPCSP(nn.Layer):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSP, self).__init__()
        c_ = int(2 * c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2D(
            in_channels=c1, out_channels=c_, kernel_size=1, stride=1, bias_attr=False
        )
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.LayerList(
            sublayers=[nn.MaxPool2D(kernel_size=x, stride=1, padding=x // 2) for x in k]
        )
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.bn = nn.BatchNorm2D(num_features=2 * c_)
        self.act = nn.Mish()
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(paddle.concat(x=[x1] + [m(x1) for m in self.m], axis=1)))
        y2 = self.cv2(x)
        return self.cv7(self.act(self.bn(paddle.concat(x=(y1, y2), axis=1))))


class MP(nn.Layer):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2D(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class Focus(nn.Layer):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        return self.conv(
            paddle.concat(
                x=[
                    x[..., ::2, ::2],
                    x[..., 1::2, ::2],
                    x[..., ::2, 1::2],
                    x[..., 1::2, 1::2],
                ],
                axis=1,
            )
        )


class Concat(nn.Layer):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return paddle.concat(x=x, axis=self.d)


class Flatten(nn.Layer):
    @staticmethod
    def forward(x):
        return x.reshape((x.shape[0], -1))


class Classify(nn.Layer):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2D(output_size=1)
        self.conv = nn.Conv2D(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=autopad(k, p),
            groups=g,
            bias_attr=False,
        )
        self.flat = Flatten()

    def forward(self, x):
        z = paddle.concat(
            x=[self.aap(y) for y in (x if isinstance(x, list) else [x])], axis=1
        )
        return self.flat(self.conv(z))


import os
import collections


class CombConvLayer(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel=1, stride=1, dropout=0.1, bias=False
    ):
        super().__init__()
        self.add_sublayer(
            name="layer1", sublayer=ConvLayer(in_channels, out_channels, kernel)
        )
        self.add_sublayer(
            name="layer2",
            sublayer=DWConvLayer(out_channels, out_channels, stride=stride),
        )

    def forward(self, x):
        return super().forward(x)


class DWConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super().__init__()
        out_ch = out_channels
        groups = in_channels
        kernel = 3
        self.add_sublayer(
            name="dwconv",
            sublayer=nn.Conv2D(
                in_channels=groups,
                out_channels=groups,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
                bias_attr=bias,
            ),
        )
        self.add_sublayer(name="norm", sublayer=nn.BatchNorm2D(num_features=groups))

    def forward(self, x):
        return super().forward(x)


class ConvLayer(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1, bias=False
    ):
        super().__init__()
        out_ch = out_channels
        groups = 1
        self.add_sublayer(
            name="conv",
            sublayer=nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_ch,
                kernel_size=kernel,
                stride=stride,
                padding=kernel // 2,
                groups=groups,
                bias_attr=bias,
            ),
        )
        self.add_sublayer(name="norm", sublayer=nn.BatchNorm2D(num_features=out_ch))
        self.add_sublayer(name="relu", sublayer=nn.ReLU6())

    def forward(self, x):
        return super().forward(x)


class HarDBlock(nn.Layer):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2**i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(
        self,
        in_channels,
        growth_rate,
        grmul,
        n_layers,
        keepBase=False,
        residual_out=False,
        dwconv=False,
    ):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0
        for i in range(n_layers):
            outch, inch, link = self.get_link(i + 1, in_channels, growth_rate, grmul)
            self.links.append(link)
            use_relu = residual_out
            if dwconv:
                layers_.append(CombConvLayer(inch, outch))
            else:
                layers_.append(Conv(inch, outch, k=3))
            if i % 2 == 0 or i == n_layers - 1:
                self.out_channels += outch
        self.layers = nn.LayerList(sublayers=layers_)

    def forward(self, x):
        layers_ = [x]
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = paddle.concat(x=tin, axis=1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)
        t = len(layers_)
        out_ = []
        for i in range(t):
            if i == 0 and self.keepBase or i == t - 1 or i % 2 == 1:
                out_.append(layers_[i])
        out = paddle.concat(x=out_, axis=1)
        return out


class BRLayer(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        self.add_sublayer(
            name="norm", sublayer=nn.BatchNorm2D(num_features=in_channels)
        )
        self.add_sublayer(name="relu", sublayer=nn.ReLU())

    def forward(self, x):
        return super().forward(x)


class HarDBlock2(nn.Layer):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2**i
            if layer % dv == 0:
                k = layer - dv
                link.insert(0, k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, dwconv=False):
        super().__init__()
        self.links = []
        conv_layers_ = []
        bnrelu_layers_ = []
        self.layer_bias = []
        self.out_channels = 0
        self.out_partition = collections.defaultdict(list)
        for i in range(n_layers):
            outch, inch, link = self.get_link(i + 1, in_channels, growth_rate, grmul)
            self.links.append(link)
            for j in link:
                self.out_partition[j].append(outch)
        cur_ch = in_channels
        for i in range(n_layers):
            accum_out_ch = sum(self.out_partition[i])
            real_out_ch = self.out_partition[i][0]
            conv_layers_.append(
                nn.Conv2D(
                    in_channels=cur_ch,
                    out_channels=accum_out_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias_attr=True,
                )
            )
            bnrelu_layers_.append(BRLayer(real_out_ch))
            cur_ch = real_out_ch
            if i % 2 == 0 or i == n_layers - 1:
                self.out_channels += real_out_ch
        self.conv_layers = nn.LayerList(sublayers=conv_layers_)
        self.bnrelu_layers = nn.LayerList(sublayers=bnrelu_layers_)

    def transform(self, blk, trt=False):
        in_ch = blk.layers[0][0].weight.shape[1]
        for i in range(len(self.conv_layers)):
            link = self.links[i].copy()
            link_ch = [
                (
                    blk.layers[k - 1][0].weight.shape[0]
                    if k > 0
                    else blk.layers[0][0].weight.shape[1]
                )
                for k in link
            ]
            part = self.out_partition[i]
            w_src = blk.layers[i][0].weight
            b_src = blk.layers[i][0].bias
            self.conv_layers[i].weight[0 : part[0], :, :, :] = w_src[:, 0:in_ch, :, :]
            self.layer_bias.append(b_src)
            if b_src is not None:
                if trt:
                    self.conv_layers[i].bias[1 : part[0]] = b_src[1:]
                    self.conv_layers[i].bias[0] = b_src[0]
                    self.conv_layers[i].bias[part[0] :] = 0
                    self.layer_bias[i] = None
                else:
                    self.conv_layers[i].bias = None
            else:
                self.conv_layers[i].bias = None
            in_ch = part[0]
            link_ch.reverse()
            link.reverse()
            if len(link) > 1:
                for j in range(1, len(link)):
                    ly = link[j]
                    part_id = self.out_partition[ly].index(part[0])
                    chos = sum(self.out_partition[ly][0:part_id])
                    choe = chos + part[0]
                    chis = sum(link_ch[0:j])
                    chie = chis + link_ch[j]
                    self.conv_layers[ly].weight[chos:choe, :, :, :] = w_src[
                        :, chis:chie, :, :
                    ]
            self.bnrelu_layers[i] = None
            if isinstance(blk.layers[i][1], nn.BatchNorm2D):
                self.bnrelu_layers[i] = nn.Sequential(
                    blk.layers[i][1], blk.layers[i][2]
                )
            else:
                self.bnrelu_layers[i] = blk.layers[i][1]

    def forward(self, x):
        layers_ = []
        outs_ = []
        xin = x
        for i in range(len(self.conv_layers)):
            link = self.links[i]
            part = self.out_partition[i]
            xout = self.conv_layers[i](xin)
            layers_.append(xout)
            xin = xout[:, 0 : part[0], :, :] if len(part) > 1 else xout
            if len(link) > 1:
                for j in range(len(link) - 1):
                    ly = link[j]
                    part_id = self.out_partition[ly].index(part[0])
                    chs = sum(self.out_partition[ly][0:part_id])
                    che = chs + part[0]
                    xin += layers_[ly][:, chs:che, :, :]
            xin = self.bnrelu_layers[i](xin)
            if i % 2 == 0 or i == len(self.conv_layers) - 1:
                outs_.append(xin)
        out = paddle.concat(x=outs_, axis=1)
        return out


class ConvSig(nn.Layer):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(ConvSig, self).__init__()
        self.conv = nn.Conv2D(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=autopad(k, p),
            groups=g,
            bias_attr=False,
        )
        self.act = nn.Sigmoid() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class ConvSqu(nn.Layer):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(ConvSqu, self).__init__()
        self.conv = nn.Conv2D(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=autopad(k, p),
            groups=g,
            bias_attr=False,
        )
        self.act = nn.Mish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))

    def fuseforward(self, x):
        return self.act(self.conv(x))


"""
class SE(nn.Module):
    # Squeeze-and-excitation block in https://arxiv.org/abs/1709.01507
    def __init__(self, c1, c2, n=1, shortcut=True, g=8, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(SE, self).__init__()
        c_ = int(c2)  # hidden channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cs = ConvSqu(c1, c1//g, 1, 1)
        self.cvsig = ConvSig(c1//g, c1, 1, 1)

    def forward(self, x):
        return x = x * self.cvsig(self.cs(self.avg_pool(x))).expand_as(x)
    
class SAM(nn.Module):
    # SAM block in yolov4
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(SAM, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cvsig = ConvSig(c1, c1, 1, 1)

    def forward(self, x):
        return x = x * self.cvsig(x)
    
class DNL(nn.Module):
    # Disentangled Non-Local block in https://arxiv.org/abs/2006.06668
    def __init__(self, c1, c2, k=3, s=1):
        super(DNL, self).__init__()
        c_ = int(c1)  # hidden channels
        
        # 
        self.conv_query = nn.Conv2d(c1, c_, kernel_size=1)
        self.conv_key = nn.Conv2d(c1, c_, kernel_size=1)
        
        self.conv_value = nn.Conv2d(c1, c1, kernel_size=1, bias=False)
        self.conv_out = None
        
        self.scale = math.sqrt(c_)
        self.temperature = 0.05
        
        self.softmax = nn.Softmax(dim=2)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.conv_mask = nn.Conv2d(c1, 1, kernel_size=1)
                
        self.cv = Conv(c1, c2, k, s)

    def forward(self, x):

        # [N, C, T, H, W]
        residual = x
        
        # [N, C, T, H', W']        
        input_x = x

        # [N, C', T, H, W]
        query = self.conv_query(x)
        
        # [N, C', T, H', W']
        key = self.conv_key(input_x)
        value = self.conv_value(input_x)

        # [N, C', H x W]
        query = query.view(query.size(0), query.size(1), -1)
        
        # [N, C', H' x W']
        key = key.view(key.size(0), key.size(1), -1)
        value = value.view(value.size(0), value.size(1), -1)
        
        # channel whitening
        key_mean = key.mean(2).unsqueeze(2)
        query_mean = query.mean(2).unsqueeze(2)
        key -= key_mean
        query -= query_mean

        # [N, T x H x W, T x H' x W']
        sim_map = torch.bmm(query.transpose(1, 2), key)
        sim_map = sim_map/self.scale
        sim_map = sim_map/self.temperature
        sim_map = self.softmax(sim_map)

        # [N, T x H x W, C']
        out_sim = torch.bmm(sim_map, value.transpose(1, 2))
        
        # [N, C', T x H x W]
        out_sim = out_sim.transpose(1, 2)
        
        # [N, C', T,  H, W]
        out_sim = out_sim.view(out_sim.size(0), out_sim.size(1), *x.size()[2:])
        out_sim = self.gamma * out_sim
        
        # [N, 1, H', W']
        mask = self.conv_mask(input_x)
        # [N, 1, H'x W']
        mask = mask.view(mask.size(0), mask.size(1), -1)
        mask = self.softmax(mask)
        # [N, C, 1, 1]
        out_gc = torch.bmm(value, mask.permute(0,2,1)).unsqueeze(-1)
        out_sim = out_sim+out_gc

        return self.cv(out_sim + residual)


class GC(nn.Module):
    # global context block in https://arxiv.org/abs/1904.11492
    def __init__(self, c1, c2, k=3, s=1):
        super(GC, self).__init__()
        c_ = int(c1)  # hidden channels
        
        #             
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(c1, c_, kernel_size=1),
            nn.LayerNorm([c_, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(c_, c1, kernel_size=1))
        
        self.conv_mask = nn.Conv2d(c_, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
                
        self.cv = Conv(c1, c2, k, s)
        
        
    def spatial_pool(self, x):
        
        batch, channel, height, width = x.size()
        
        input_x = x        
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):

        return self.cv(x + self.channel_add_conv(self.spatial_pool(x)))
"""
