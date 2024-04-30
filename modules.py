import torch
import torch.nn as nn

from attention import build_attention


def attention_padding_mask(q, k, padding_index=0):
    """Generate mask tensor for padding value

    Args:
        q (Tensor): (B, T_q)
        k (Tensor): (B, T_k)
        padding_index (int): padding index. Default: 0

    Returns:
        (torch.BoolTensor): Mask with shape (B, T_q, T_k). True element stands for requiring making.

    Notes:
        Assume padding_index is 0:
        k.eq(0) -> BoolTensor (B, T_k)
        k.eq(0).unsqueeze(1)  -> (B, 1, T_k)
        k.eq(0).unsqueeze(1).expand(-1, q.size(-1), -1) -> (B, T_q, T_k)

    """

    mask = k.eq(padding_index).unsqueeze(1).expand(-1, q.size(-1), -1)
    return mask


class BasicBlock(nn.Module):
    """
    膨胀卷积模块
    """

    def __init__(self, in_channel, out_first, out_second, kernel_size, dilation):
        super(BasicBlock, self).__init__()
        # 此处卷积作用相当于全连接层，用于压缩降维，减少参数量，out_channel可自己指定
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_first,
                               kernel_size=(1, 1), padding=0, bias=False)  # 使用bn层时不使用bias
        self.bn1 = nn.BatchNorm2d(out_first)  # BN层只修改channel通道
        # 空洞卷积单元
        self.conv2 = nn.Conv2d(in_channels=out_first, out_channels=out_second,
                               kernel_size=kernel_size, dilation=dilation,
                               padding=(kernel_size - 1) * dilation // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_second)  # BN层只修改channel通道

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通过第一个卷积单元
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sigmoid(out)
        # 通过第二个卷积单元
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sigmoid(out)

        return out


class PERCNet(nn.Module):
    """
    block:BasicBlock
    between_channels:1维数组，BasicBlock间的通道数，2个BasicBlock构成1个resblock
    kernel_sizes:2维数组，每个BasicBlock的kernel大小
    dilations:2维数组，每个BasicBlock的dilations大小
    input_dims:整个网络的输入维度
    output_dims:整个网络的输出维度
    """

    def __init__(self, block, in_channel, output_channel, out_first, out_second, kernel_size, dilation):
        super(PERCNet, self).__init__()
        self.output_channel = output_channel
        self.layer1 = block(in_channel, out_first[0], out_second[0], kernel_size[0], dilation[0])
        self.layer2 = block(in_channel, out_first[1], out_second[1], kernel_size[1], dilation[1])
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size[2], padding=1, stride=1)
        self.conv1 = nn.Conv2d(in_channels=out_first[2], out_channels=out_second[2],
                               kernel_size=(1, 1), padding=0, bias=False)  # 使用bn层时不使用bias
        self.bn1 = nn.BatchNorm2d(out_second[2])

        self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=out_first[3],
                               kernel_size=(1, 1), padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_first[3])

        self.sigmoid = nn.Sigmoid()

        for m in self.modules():  # 卷积层的初始化操作
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='sigmoid')

    def forward(self, inputs):
        layer1 = self.layer1(inputs)
        layer2 = self.layer2(inputs)

        maxpool = self.maxpool(inputs)
        conv1 = self.conv1(maxpool)
        conv1 = self.bn1(conv1)

        conv2 = self.conv2(inputs)
        conv2 = self.bn2(conv2)

        output = torch.cat([layer1, layer2, conv1, conv2], dim=1).squeeze(0)
        output = self.sigmoid(output)

        assert output.shape[0] == self.output_channel

        return output


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, dropout_rate=0.0, attention_type='scaled_dot'):
        super().__init__()

        assert model_dim % num_heads == 0, 'model_dim should be devided by num_heads'

        self.h_size = model_dim
        self.num_heads = num_heads
        self.head_h_size = model_dim // num_heads

        self.linear_q = nn.Linear(self.h_size, self.h_size)
        self.linear_k = nn.Linear(self.h_size, self.h_size)
        self.linear_v = nn.Linear(self.h_size, self.h_size)

        self.attention = build_attention(attention_type, q_dim=self.head_h_size, k_dim=self.head_h_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, attn_mask=None):
        batch_size = q.size(0)

        # Residual
        residual = q

        # Linear projection
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        # Form multi heads
        q = q.view(self.num_heads * batch_size, -1, self.head_h_size)  # (h * B, T_q, D / h)
        k = k.view(self.num_heads * batch_size, -1, self.head_h_size)  # (h * B, T_k, D / h)
        v = v.view(self.num_heads * batch_size, -1, self.head_h_size)  # (h * B, T_v, D / h)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(self.num_heads, 1, 1)  # (h * B, T_q, T_k)

        context, attention = self.attention(q, k, v, attn_mask=attn_mask)
        # context: (h * B, T_q, D_v) attention: (h * B, T_q, T_k)

        # Concatenate heads
        context = context.view(batch_size, -1, self.h_size)  # (B, T_q, D)

        # Dropout
        output = self.dropout(context)  # (B, T_q, D)

        return output, attention


def PERC(input_dims=768, output_dims=768):
    # 通过调整模型内部BasicBlock的数量和配置实现不同的ResNet
    # out_first = [96, 16, input_dims, 64]
    # out_second = [128, 32, 32]
    # kernel_size = [5, 3, 3]
    # dilations = [2, 2]
    out_first = [96, 16, input_dims, 192]
    out_second = [384, 96, 96]
    kernel_size = [5, 3, 3]
    dilations = [2, 2]
    return PERCNet(BasicBlock, input_dims, output_dims, out_first, out_second, kernel_size, dilations)


if __name__ == '__main__':
    def PERC(input_dims=768, output_dims=768):
        # 通过调整模型内部BasicBlock的数量和配置实现不同的ResNet
        # out_first = [96, 16, input_dims, 64]
        # out_second = [128, 32, 32]
        # kernel_size = [5, 3, 3]
        # dilations = [2, 2]
        out_first = [96, 16, input_dims, 192]
        out_second = [384, 96, 96]
        kernel_size = [5, 3, 3]
        dilations = [2, 2]
        return PERCNet(BasicBlock, input_dims, output_dims, out_first, out_second, kernel_size, dilations)


    batch_size = 32
    seq_length = 100
    embed_dim = 768
    text_data = torch.randn(1, embed_dim, batch_size, seq_length)
    num_classes = 768  # 假设有10个类别需要分类
    model = PERC(embed_dim, num_classes)
    outputs = model(text_data)
    print(outputs.shape)  # 应该输出 [batch_size, num_classes]
