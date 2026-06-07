import torch
import torch.nn as nn


class QHAudioLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps=eps
        self.scale=nn.Parameter(torch.ones(dim))
        self.bias=nn.Parameter(torch.zeros(dim))

    def forward(self,x):
        mu=x.mean(dim=-1,keepdim=True)
        variance=x.var(dim=-1,keepdim=True, unbiasd=False)
        x_norm=(x-mu)/torch.sqrt(variance+self.eps)
        return x_norm*self.scale+self.bias


# TODO
class QHAudioRMSNorm(nn.Module):
    def __init__(self,):
        super().__init__()


class QHAudioBatchNorm(nn.Module):
    def __init__(self,dim,eps=1e-5,momentum=0.1):
        super().__init__()
        self.eps=eps
        self.momentum=momentum
        self.scale=nn.Parameter(torch.ones(dim))
        self.bias=nn.Parameter(torch.zeros(dim))

        self.register_buffer("running_mean",torch.zeros(1,dim))
        self.register_buffer("running_var",torch.ones(1,dim))

    def forward(self,x):
        B,T,F=x.size()

        # [B, T, F] --> [B*T, F]
        x=x.view(-1, F)
        if self.training:
            mu=x.mean(dim=0,keepdim=True) # [1, F]
            variance=x.var(dim=0,keepdim=True,unbiased=False) # [1, F]

            self.running_mean=self.running_mean*(1-self.momentum)+mu*self.momentum
            self.running_var=self.running_var*(1-self.momentum)+variance*self.momentum
        else:
            mu=self.running_mean
            variance=self.running_var

        x_norm=(x-mu)/torch.sqrt(variance+self.eps)
        x_norm=x_norm.view(B,T,F)
        return x_norm*self.scale+self.bias


# 位置编码是一个面试重点
# TODO 继续理解位置编码，目前为止只能复现代码，但仍无法真正理解位置编码
class QHSinusoidalPE(nn.Module):
    def __init__(self, fdim, flen):
        super().__init__()
        self.fdim=fdim
        self.flen=flen
        assert fdim%2==0

        pos=torch.arange(self.flen).unsqueeze(-1) # [flen, 1]
        i=torch.arange(self.fdim//2).unsqueeze(0) # [1, fdim//2]
        x=pos/(10000**(2*i/self.fdim))
        sinPE=torch.sin(x)
        cosPE=torch.cos(x)
        # .unsqueeze(0) 是为了在与feature相加时，方便对batchsize维度进行广播
        PE=torch.cat([sinPE,cosPE],dim=-1).unsqueeze(0)
        # 一般来说，位置编码不需要梯度，因此注册为buffer
        self.register_buffer("PE",PE)

    def forward(self):
        return self.PE


# TODO 理解MLP中split操作的作用？这种类似于glu的操作有什么作用？在glu里面加入深度卷积有什么实际含义吗？
class QHcgMLP(nn.Module):
    def __init__(self,fdim,hdim,kernel_size=31):
        super().__init__()
        assert hdim%2==0
        self.pre_norm=QHAudioLayerNorm(fdim)
        self.up_proj=nn.Linear(fdim,hdim)
        self.gelu=nn.GELU()
        self.conv_norm=QHAudioLayerNorm(hdim//2)
        self.depth_wise_conv=nn.Conv1d(fdim,fdim,kernel_size,1,(kernel_size-1)//2,groups=fdim)
        self.down_proj=nn.Linear(hdim//2,fdim)
        self.dp=nn.Dropout(0.1)

    def forward(self,x):
        # x [B, T, F]
        x=self.pre_norm(x)
        x=self.gelu( self.up_proj(x) )
        x1,x2=torch.chunk(x, chunks=2, dim=-1)
        x2=self.conv_norm(x2)
        x2=self.depth_wise_conv(x2.transpose(1,2))
        x=x1*x2.transpose(1,2)
        x=self.dp( self.down_proj(x) )
        return x


# TODO 实现GQA，确保MHA是QGA的特例
class QHMultiHeadAttention(nn.Module):
    def __init__(self,fdim,num_heads):
        super().__init__()
        assert fdim%num_heads==0
        self.fdim=fdim
        self.num_heads=num_heads
        self.head_dim=fdim//num_heads
        self.q_proj=nn.Linear(fdim,fdim)
        self.k_proj=nn.Linear(fdim,fdim)
        self.v_proj=nn.Linear(fdim,fdim)
        self.o_proj=nn.Linear(fdim,fdim)

    def forward(self,x):
        B,T,F=x.size()
        q=self.q_proj(x).view(B, self.num_heads, T, self.head_dim) # q [B, nheads, T, hdim]
        k=self.k_proj(x).view(B, self.num_heads, self.head_dim, T) # k [B, nheads, hdim, T]
        v=self.v_proj(x).view(B, self.num_heads, T,  self.head_dim) # v [B, nheads, T, hdim]

        att_scores=nn.functional.softmax(q@k/torch.sqrt(self.head_dim)) # att_scores [B, nheads, T, T]
        outputs=att_scores@v # outputs [B, nheads, T, hdim]
        outputs=self.o_proj(outputs.view(B,T,F))
        return outputs


class QHEbranchformerEncoderLayer(nn.Module):
    def __init__(self,fdim,mhattn,cgmlp,droup_out=0.1):
        super().__init__()
        self.pre_att_norm=QHAudioLayerNorm(fdim)
        self.pre_cgmlp_norm=QHAudioLayerNorm(fdim)
        self.dp=nn.Dropout(droup_out)
        self.merge_proj=nn.Linear(fdim*2,fdim)
        self.mhattn=mhattn
        self.cgmlp=cgmlp

    def forward(self,x):

        # att branch
        x1=self.dp(
            self.mhattn(
                self.pre_att_norm(x)
            )
        )
        
        # conv branch
        x2=self.dp(
            self.cgmlp(
                self.pre_cgmlp_norm(x)
            )
        )

        # merge
        x_merge=torch.cat((x1,x2), dim=-1)
        x_merge=self.merge_proj(x_merge)

        return x+x_merge


class QHEbranchformerEncoder(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self,x):
        pass
