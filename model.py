import torch 
import math
import torch.nn as nn

class MiniGptConfig:

    def __init__(
        self,
        vocab_size,
        max_position_embeddings = 512,
        n_layers = 6,
        n_heads = 6,
        d_model = 384,
        d_ff = 1536,
        dropout = 0.1,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff 
        self.d_model= d_model
        self.dropout = dropout
    
class Embeddings(nn.Module):

    def __init__(self,config:MiniGptConfig):
        super().__init__()

        self.token_embedding = nn.Embedding(config.vocab_size,config.d_model)
        self.position_embedding = nn.Embedding(config.max_position_embeddings,config.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self,input_ids):

        batch_size, seq_length = input_ids.size()
        
        positions = torch.arange(0,seq_length,device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size,seq_length)

        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)

        return self.dropout(token_emb + pos_emb)
    

class MultiHeadSelfAttention(nn.Module):

    def __init__(self,config:MiniGptConfig):
        super().__init__()

        assert config.d_model % config.n_heads == 0 

        self.n_heads = config.n_heads   
        self.head_dim = config.d_model // config.n_heads
        self.d_model = config.d_model
        
        self.qkv_proj = nn.Linear(self.d_model,3 * self.d_model)
        self.out_proj = nn.Linear(config.d_model,config.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self,x):
        batch_size,seq_length,d_model = x.size()
        qkv = self.qkv_proj(x)

        q,k,v = torch.chunk(qkv,3, dim=-1)

        q = q.view(batch_size,seq_length,self.n_heads,self.head_dim).transpose(1,2)
        k = k.view(batch_size,seq_length,self.n_heads,self.head_dim).transpose(1,2)
        v= v.view(batch_size,seq_length,self.n_heads,self.head_dim).transpose(1,2)
        
        attn_score = (q @ k.transpose(-2,-1)/ math.sqrt(self.head_dim))

        mask = torch.tril(torch.ones(seq_length,seq_length,device=x.device))
        mask = mask.unsqueeze(0).unsqueeze(0)

        attn_scores = attn_score.masked_fill(mask == 0, float("-inf"))
        attn_probs = torch.softmax(attn_scores,dim = -1)
        
        attn_probs = self.dropout(attn_probs)
        attn_output = attn_probs @ v

        attn_output = attn_output.transpose(1,2).contiguous()
        attn_output = attn_output.view(batch_size,seq_length,d_model)

        output = self.out_proj(attn_output)
        return output
    

class FeedForward(nn.Module):

    def __init__(self, config: MiniGptConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model,config.d_ff)
        self.activation = nn.GELU()

        self.fc2 = nn.Linear(config.d_ff,config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self,x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x




class TransformerBlock(nn.Module):
    def __init__(self, config:MiniGptConfig):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.d_model)
        self.attention = MultiHeadSelfAttention(config)

        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self,x):

        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x




class MiniGpt(nn.Module):

    def __init__(self,config:MiniGptConfig):
        super().__init__()
        
        self.config = config
        self.embeddings = Embeddings(config)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.ln_f = nn.LayerNorm(config.d_model)

        self.lm_head = nn.Linear(config.d_model,config.vocab_size,bias=False)

    def forward(self,input_ids):
        x = self.embeddings(input_ids)

        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)

        logits = self.lm_head(x)

        return logits


if __name__ == "__main__":

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    config = MiniGptConfig(vocab_size=tokenizer.vocab_size)
    model = MiniGpt(config)

    dummy_input = torch.randint(0,tokenizer.vocab_size,(2,16))

    logits = model(dummy_input)

    print(f"logits output shape:{logits.shape}")        
