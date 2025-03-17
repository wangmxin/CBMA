class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class LocalTemporalResponseEnhancement(nn.Module):
    def __init__(self, channels):
        super(LocalTemporalResponseEnhancement, self).__init__()
        # 主卷积路径
        self.main_conv_path = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm1d(channels),
            nn.Sigmoid()
        )

        self.secondary_conv_path = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        main_scale = self.main_conv_path(x)
        secondary_scale = self.secondary_conv_path(x)
            output = x * main_scale * secondary_scale
        return output

class TCN(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.5):
        super(TCN, self).__init__()

        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.ltre1 = LocalTemporalResponseEnhancement(n_outputs)
        self.elu1 = nn.ELU()

        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                    stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.ltre2 = LocalTemporalResponseEnhancement(n_outputs)
        self.elu2 = nn.ELU()

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.elu1,
            self.conv2, self.chomp2, self.elu2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.elu = nn.ELU()
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.conv1.weight, 0, 0.01)
        nn.init.normal_(self.conv2.weight, 0, 0.01)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        out = out + res
        out = self.elu(out)
        return out

class BiTCN(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, dilation, padding, dropout):
        super(BiTCN, self).__init__()
        self.temporal_block_fwd = TCN(input_size, output_size, kernel_size, stride=1,
                                                dilation=dilation, padding=padding, dropout=dropout)
        self.temporal_block_bwd = TCN(input_size, output_size, kernel_size, stride=1,
                                                dilation=dilation, padding=padding, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out_fwd = self.temporal_block_fwd(x)
        out_bwd = self.temporal_block_bwd(x.flip(dims=[2]))
        out_bwd = out_bwd.flip(dims=[2])
        out = torch.cat((out_fwd, out_bwd), dim=1)
        out = self.dropout(out)
        return out

class AdaptiveFeatureTransformation(nn.Module):
    def __init__(self, input_dim):
        super(AdaptiveFeatureTransformation, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = x + residual
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, query_dim, key_dim, value_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.dim_per_head = query_dim // num_heads
        self.query_linear = nn.Linear(query_dim, query_dim)
        self.key_linear = nn.Linear(key_dim, query_dim)
        self.value_linear = nn.Linear(value_dim, query_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(query_dim, query_dim)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.dim_per_head)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.dim_per_head)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context = torch.matmul(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, -1, self.num_heads * self.dim_per_head)
        output = self.output_linear(context)
        return output

class CBMAModel(nn.Module):
    def __init__(self, batch_size, cnn_input_dim, bitcn_input_dim, conv_archs, num_channels, kernel_size, output_dim,
                 dropout=0.5, num_heads=8):
        super().__init__()
        self.batch_size = batch_size
        self.conv_arch = conv_archs
        self.input_channels = cnn_input_dim
        self.spacefeatures = self.make_layers()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = bitcn_input_dim if i == 0 else num_channels[i - 1] * 2
            out_channels = num_channels[i]
            layers += [BiTCN(in_channels, out_channels, kernel_size, dilation=dilation_size,
                                       padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        self.BiTCNnetwork = nn.Sequential(*layers)
        self.multihead_attention = MultiHeadAttention(num_heads, conv_archs[-1][-1], num_channels[-1] * 2, num_channels[-1] * 2)
        # self.aft = AdaptiveFeatureTransformation(conv_archs[-1][-1])
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(conv_archs[-1][-1], output_dim)

    def CNN_layers(self):
        layers = []
        for (num_convs, out_channels) in self.conv_arch:
            for _ in range(num_convs):
                layers.append(nn.Conv1d(self.input_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ELU(inplace=True))
                self.input_channels = out_channels
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, input_seq):
        space_features = self.spacefeatures(input_seq)
        space_features = space_features.permute(0, 2, 1)
        input_seq = input_seq.view(self.batch_size, -1, space_features.size(1))
        bitcn_features = self.BiTCNnetwork(input_seq)
        bitcn_features = bitcn_features.permute(0, 2, 1)
        query = space_features
        key = bitcn_features
        value = bitcn_features
        cross_attention_features = self.multihead_attention(query, key, value)
        x = self.adaptive_pool(cross_attention_features.transpose(1, 2))
        flat_tensor = x.reshape(self.batch_size, -1)
        outputs = self.classifier(flat_tensor)
        return outputs

# parameters(THU DATASET)
batch_size = 32
cnn_input = 7  
spaceconv_arch = ((1, 32), (1, 64), (1, 128))
bitcn_input = 56
num_channels = [64, 128]
kernel_size = 3
dropout = 0.5
output_dim = 14
num_heads = 8
