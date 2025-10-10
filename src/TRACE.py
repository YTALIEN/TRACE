# source code for TRACE 
import torch
import torch.nn as nn
from config import Config
from torch.nn import functional as F
import math



args=Config(print_flag=False)

class Embedding(nn.Module):
    def __init__(self, 
                 input_dim, 
                 embed_dim):
        super(Embedding, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim//2),  
            nn.GELU(),
            nn.Linear(embed_dim//2, embed_dim//4),  
            nn.GELU(),
            nn.Linear(embed_dim//4, embed_dim),  # Linear projection
            nn.LayerNorm(embed_dim)  # LayerNorm
        )
        
    def forward(self, window_seq,mask):
        # input: (B, seq_len, feature_dim)
        embed_out=self.embed(window_seq)  # (B, seq_len, embed_dim)
        embed_out=embed_out.masked_fill(mask.unsqueeze(-1),0.0)
        return embed_out

class Context_Attention_layer(nn.Module):
    def __init__(self, 
                 embed_dim:int, 
                 nhead:int ):
        super().__init__()
        self.embed_dim = embed_dim
        self.nheads = nhead
        assert embed_dim % nhead == 0, "embed_dim must be divisible by nheads"
        self.head_dim = embed_dim // nhead  
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_weights=None

    def forward(self, window_seq,mask):
        """
        input: 
            window_seq: (Batch,seqlen, EmbedDim)
            mask: (Batch,seqlen)
        output: (Batch, EmbedDim)
        """
        bs,seq_len,embed_dim=window_seq.size()

        context_token=window_seq[:,0,:].unsqueeze(1) # (B, 1, E)
        q = self.q_proj(context_token) # (B, 1, E)
        k = self.k_proj(window_seq)  # (B, S, E)
        v = self.v_proj(window_seq)  # (B, S, E)

        scores=torch.matmul(q,k.transpose(-2,-1))/(k.size(-1)**0.5)  # (B, 1, S)
        scores=scores.masked_fill(mask.unsqueeze(1),-float('inf'))

        attn_weights=F.softmax(scores,dim=-1)  # (B, 1, S)

        self.attn_weights = attn_weights.detach()
        attended=torch.matmul(attn_weights,v) # (B, 1, E)

        encoder_context_out=attended.expand(-1,seq_len,-1) # (B, S, E)
        return self.out_proj(encoder_context_out)  # (B, S, E)

class Context_Encoder(nn.Module):
    def __init__(self, 
                num_layers, 
                embed_dim, 
                nhead):
        super().__init__()
        self.layers = nn.ModuleList([
            Context_Attention_layer(embed_dim, nhead) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x  # (B, S, E)
        
class PositionalEncoding(nn.Module):
    def __init__(self, 
                 embed_dim):
        super().__init__()
        self.embed_dim=embed_dim
    def forward(self, x,mask):
        seq_len = x.size(1)
        position = torch.arange(seq_len, device=x.device).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2).float() * 
            (-math.log(10000.0) / self.embed_dim)
        ).to(x.device)
        pe = torch.zeros(seq_len, self.embed_dim, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pos_out=x + pe.unsqueeze(0)
        pos_out=pos_out.masked_fill(mask.unsqueeze(-1),0.0)
        return pos_out  
    

class TRACE(nn.Module):
    def __init__(self, 
                input_dim:int=7,
                max_window_num:int=21,
                embed_dim:int =128, 
                hidden_dim:int=256,
                nhead=4, 
                num_classes:int=None):
        super(TRACE,self).__init__()
        self.input_dim = input_dim
        self.max_window_num = max_window_num
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.nhead = nhead
        if num_classes is None:
            self.num_classes = max_window_num
        else:
            self.num_classes = num_classes
        
    
        self.embedding=Embedding(
            input_dim=self.input_dim,  
            embed_dim=embed_dim  
        )
        
        self.position_enc = PositionalEncoding(embed_dim) # fixed sin position_encoding
        
        self.encoder_ori_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim, 
            dropout=0.5,
            activation='relu',
            norm_first=True,
            batch_first=True  # (bs, window_num, embed_dim)
        )
        self.G_MSA = nn.TransformerEncoder(self.encoder_ori_layer, 
                                                num_layers=2,
                                                enable_nested_tensor=False )
        
        self.C_MSA=Context_Encoder(num_layers=2,
                                                embed_dim=embed_dim,
                                                nhead=nhead)
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim), # input: (bs, seq_len, embed_dim*2) → (bs, seq_len,embed_dim)
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(embed_dim, 1)
        )

            
    def forward(self, window_seq,mask):
        '''
        input: 
            window_seq: (bs, window_num, feature_dim): padding
            mask: (bs, window_num) 
        return: (bs, window_num)
        '''
        
        seq_clean=window_seq.clone() # after padding:(bs, window_num, feature_dim)
        embed_seq = self.embedding(seq_clean,mask)  # (bs, window_num, embed_dim)
        pos_embed_seq = self.position_enc(embed_seq,mask)

        encoder_ori_out = self.G_MSA(
            pos_embed_seq,
            src_key_padding_mask=mask
            )  # (bs, seq_len,embed_dim)

        # encoder_context_out=self.context_attention(encoder_ori_out,mask)  # (bs, seq_len,embed_dim)
        encoder_context_out=self.C_MSA(pos_embed_seq,mask)  # (bs, seq_len,embed_dim) 

        combined=torch.cat([encoder_ori_out,encoder_context_out],dim=-1)  # (bs, seq_len, embed_dim*2)
        class_out=self.classifier(combined).squeeze(-1)  # (bs, seq_len)
        class_out=class_out.masked_fill(mask, -float('inf'))  # (bs, seq_len)
        return class_out # (bs,seq_len)
        

def collate_fn(batch,max_len=args.max_seq_len):
    sequences,labels=zip(*batch) # seq:(seq_len,feature_dim), labels: int
    feature_dim=sequences[0].shape[-1]
    lens=torch.tensor([seq.size(0) for seq in sequences]) 

    padded_sequences=torch.zeros(
        (len(batch),max_len,feature_dim),
        dtype=torch.float,
    )
    for i,(seq,length) in enumerate(zip(sequences,lens)):
        padded_sequences[i,:length]=seq[:max_len]
    
    attention_mask=torch.arange(max_len)[None,:]>=lens[:,None] # bool:(bs,max_len)(valid:false,padding:True)

    labels=torch.stack(labels).long()

    return padded_sequences, labels, attention_mask # (bs, max_len, feature_dim), (bs,), (bs, max_len)
    

