import math

import torch
from torch import nn


class ScaledDotAttention(nn.Module):
    def __init__(self, dk):
        super().__init__()
        self.dk = dk

    def forward(self, q, k, v, mk=None):
        sc = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)
        if mk is not None:
            if mk.dim() == 2:
                mk = mk[:, None, :]
            elif mk.dim() == 3:
                mk = mk
            sc = sc.masked_fill(~mk, -1e9)
        wt = torch.softmax(sc, dim=-1)
        ot = torch.matmul(wt, v)
        return ot, wt


class MultiHeadAttention(nn.Module):
    def __init__(self, dm=128, h=4, dk=32, dv=32):
        super().__init__()
        self.dm = dm
        self.h = h
        self.dk = dk
        self.dv = dv
        self.wq = nn.ModuleList([nn.Linear(dm, dk) for _ in range(h)])
        self.wk = nn.ModuleList([nn.Linear(dm, dk) for _ in range(h)])
        self.wv = nn.ModuleList([nn.Linear(dm, dv) for _ in range(h)])
        self.wo = nn.Linear(h * dv, dm)
        self.attn = ScaledDotAttention(dk)

    def forward(self, x, mk=None):
        hs = []
        ws = []
        for i in range(self.h):
            q = self.wq[i](x)
            k = self.wk[i](x)
            v = self.wv[i](x)
            ot, wt = self.attn(q, k, v, mk)
            hs.append(ot)
            ws.append(wt)
        ot = torch.cat(hs, dim=-1)
        wt = torch.stack(ws, dim=1)
        return self.wo(ot), wt


class FFN(nn.Module):
    def __init__(self, dm=128, df=512):
        super().__init__()
        self.a = nn.Linear(dm, df)
        self.b = nn.Linear(df, dm)
        self.r = nn.ReLU()

    def forward(self, x):
        return self.b(self.r(self.a(x)))


class SinPE(nn.Module):
    def __init__(self, dm=128, mx=512):
        super().__init__()
        ps = torch.arange(mx, dtype=torch.float32).unsqueeze(1)
        ds = torch.arange(0, dm, 2, dtype=torch.float32)
        dv = torch.exp((-math.log(10000.0) / dm) * ds)
        pe = torch.zeros(mx, dm, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(ps * dv)
        pe[:, 1::2] = torch.cos(ps * dv)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        n = x.size(1)
        return x + self.pe[:, :n]


class EncoderBlock(nn.Module):
    def __init__(self, dm=128, h=4, dk=32, dv=32, df=512, dr=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dm)
        self.ln2 = nn.LayerNorm(dm)
        self.mh = MultiHeadAttention(dm=dm, h=h, dk=dk, dv=dv)
        self.ff = FFN(dm=dm, df=df)
        self.dp1 = nn.Dropout(dr)
        self.dp2 = nn.Dropout(dr)

    def forward(self, x, mk=None):
        a, w = self.mh(self.ln1(x), mk)
        x = x + self.dp1(a)
        f = self.ff(self.ln2(x))
        x = x + self.dp2(f)
        return x, w


class TransformerCls(nn.Module):
    def __init__(self, nv=10001, dm=128, h=4, dk=32, dv=32, df=512, nl=4, nc=5, mx=257, dr=0.1):
        super().__init__()
        self.dm = dm
        self.emb = nn.Embedding(nv, dm)
        self.cls = nn.Parameter(torch.zeros(1, 1, dm))
        self.pe = SinPE(dm=dm, mx=mx)
        self.dp = nn.Dropout(dr)
        self.bl = nn.ModuleList([EncoderBlock(dm=dm, h=h, dk=dk, dv=dv, df=df, dr=dr) for _ in range(nl)])
        self.ln = nn.LayerNorm(dm)
        self.hd = nn.Sequential(nn.Linear(dm, 64), nn.ReLU(), nn.Dropout(dr), nn.Linear(64, nc))
        nn.init.normal_(self.cls, mean=0.0, std=0.02)

    def forward(self, x, mk=None):
        b = x.size(0)
        z = self.emb(x)
        c = self.cls.expand(b, -1, -1)
        z = torch.cat([c, z], dim=1)
        if mk is not None:
            cm = torch.ones((b, 1), dtype=torch.bool, device=x.device)
            mk = torch.cat([cm, mk], dim=1)
        z = self.dp(self.pe(z))
        ws = []
        for bl in self.bl:
            z, w = bl(z, mk)
            ws.append(w)
        z = self.ln(z)
        lg = self.hd(z[:, 0])
        return lg, ws


if __name__ == "__main__":
    md = TransformerCls()
    x = torch.randint(0, 10001, (3, 256))
    mk = x.ne(0)
    y, ws = md(x, mk)
    print("logits shape:", tuple(y.shape))
    print("attn layers:", len(ws))
    print("attn shape:", tuple(ws[0].shape))
