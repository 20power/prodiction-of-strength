import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class Blendmapping(nn.Module):
    def __init__(self, d_model, hvi_num, comber_num, d_yc, d_y, N, heads, dropout, is_twist=False,
                 use_dirichlet=True):
        super().__init__()
        self.input = nn.Linear(hvi_num, d_model)
        self.comber_input = nn.Linear(comber_num, d_model)
        self.encoder = MatEncoder(d_model, N, heads, dropout)
        self.Linear1 = nn.Linear(d_yc, d_model)

        self.is_twist = is_twist
        self.use_dirichlet = use_dirichlet

        # 🆕 新增：比例先验注意力聚合器
        if self.use_dirichlet:
            self.mixer = DirichletAttentionMixer(d_model, use_base_concat=True)

        if self.is_twist:
            self.twist_linear = nn.Sequential(
                nn.Linear(2, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, d_model),
                nn.Sigmoid()
            )
            self.twist_output = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1),
            )

        self.output = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_y),
        )
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, prop, comber, sp_ps, twist=None, return_kl=False):
        """
        src    : [B, N, hvi_num]          原料属性
        prop   : [B, N]                   原料配比（建议先标准化到非负且每行和=1）
        comber : [B, N, comber_num]       梳棉工艺特征（逐原料或逐道工序对齐）
        sp_ps  : [B, d_yc]                纺纱控制参数
        twist  : [B, 2] or None           (股数, 捻度)；is_twist=True时有效
        """
        # 1) 原料/工艺编码
        out = F.relu(self.input(src))                       # [B, N, d]
        comber = torch.sigmoid(self.comber_input(comber))   # [B, N, d]  (替换 F.sigmoid 为 torch.sigmoid)
        out = torch.mul(out, comber)                        # 梳棉条件化
        out = self.encoder(out)                             # [B, N, d]

        # 2) 比例聚合（这里替换你的线性加权）
        if self.use_dirichlet:
            # 使用先验注意力：w = softmax(log p + s)
            if return_kl:
                out, kl = self.mixer(out, prop, return_kl=True)  # out: [B, d]
            else:
                out = self.mixer(out, prop)                      # [B, d]
        else:
            # 兼容旧版：线性加权
            out = torch.mul(out, prop.unsqueeze(2))  # [B,N,d]
            out = torch.sum(out, dim=1)             # [B,d]
            kl = None

        # 3) 与纺纱参数交互（保持不变）
        x = torch.sigmoid(self.Linear1(sp_ps))       # [B, d]
        out = torch.mul(out, x)                      # [B, d]
        out = self.dropout(out)

        # 4) is_twist 分支（保持不变）
        if self.is_twist and twist is not None:
            twist_out = self.twist_linear(twist)
            out = torch.mul(out, twist_out)
            out = self.twist_output(out)
        else:
            out = self.output(out)                   # [B, d_y]

        if return_kl and self.use_dirichlet:
            return out, kl
        return out


class MatEncoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.layers = get_clones(MatEncoderLayer(d_model, heads, dropout), N)
        self.d_model = d_model

    def forward(self, x):
        for i in range(self.N):
            x = self.layers[i](x)
        return x


class MatEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = Matricatt(heads, d_model, dropout)
        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm_1(x)
        x = x + self.attn(x, x, x)
        x = self.norm_2(x)
        x = self.dropout_1(x)
        return x


class Matricatt(nn.Module):
    def __init__(self, heads, d_model, dropout):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.v_linear = nn.Linear(d_model, d_model)

        self.convkq = nn.Conv2d(1, self.d_k, 1)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        bs = q.size(0)

        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        ckq = self.convkq(k.unsqueeze(1))
        ckq = torch.sum(ckq, dim=2).view(bs, -1, self.h, self.d_k)
        ckq = ckq.transpose(1, 2)

        v = v.transpose(1, 2)

        scores = mat_attention(ckq, v, self.d_k, self.dropout) + v

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)
        output = torch.relu(output)
        return output


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def mat_attention(qk, v, d_k, dropout=None):
    scores = qk.transpose(-2, -1) / math.sqrt(d_k)

    if dropout is not None:
        scores = dropout(scores)

    scores = F.softmax(scores, dim=-2)
    output = torch.matmul(v, scores)
    return output

class DirichletAttentionMixer(nn.Module):
    """
    ratio-aware 聚合： w = softmax(log p + s),  z_mix = Σ w_i * z_i
    返回: z_mix 以及 KL(w || p)（可加权加到总损失里）
    """
    def __init__(self, d_model, use_base_concat=True):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )
        in_dim = (3 * d_model) if use_base_concat else d_model
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU()
        )
        self.use_base_concat = use_base_concat

    def forward(self, Z_tokens, p, eps=1e-12, return_kl=False):
        """
        Z_tokens: [B, N, d_model]  每种原料的表示（来自编码+注意力后）
        p:        [B, N]           配比（要求非负且每行和=1）
        """
        B, N, D = Z_tokens.shape
        s = self.score(Z_tokens).squeeze(-1)             # [B, N] 学到的修正分数 s_i
        w = F.softmax(torch.log(p + eps) + s, dim=-1)    # [B, N] 先验修正后的权重 w_i

        z_mix  = torch.sum(w.unsqueeze(-1) * Z_tokens, dim=1)  # [B, D]

        if not self.use_base_concat:
            out = self.proj(z_mix)
            if return_kl:
                kl = torch.sum(w * (torch.log((w + eps) / (p + eps))), dim=-1).mean()
                return out, kl
            return out

        # 保留线性基线 & 偏移，增强稳定与可解释
        z_base = torch.sum(p.unsqueeze(-1) * Z_tokens, dim=1)  # [B, D]
        Z = torch.cat([z_base, z_mix, z_mix - z_base], dim=-1) # [B, 3D]
        out = self.proj(Z)

        if return_kl:
            kl = torch.sum(w * (torch.log((w + eps) / (p + eps))), dim=-1).mean()
            return out, kl
        return out
