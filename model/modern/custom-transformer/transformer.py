import torch
import torch.nn as nn
import math

class TransformerCls(nn.Module):

    def __init__(self, tokenizer, embedding_dim, nhead, num_layers, dropout, max_len=512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.tokenizer= tokenizer
        self.max_len = max_len
        self.embedding = nn.Embedding(num_embeddings=tokenizer.get_vocab_size(with_added_tokens=True), embedding_dim=embedding_dim, padding_idx=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, batch_first=True, dim_feedforward=2*embedding_dim, dropout=dropout)
        self.encoder =nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 3)
        )

        self.odd_dims = tuple(range(1, embedding_dim, 2))
        self.even_dims = tuple(range(0, embedding_dim, 2))

    def positional_encoding(self, x):

        seq_len, embedding_dim = x.shape[-2], x.shape[-1]

        position = torch.arange(seq_len, device=x.device).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, device=x.device)
            * (-math.log(10000.0) / embedding_dim)
        )

        pe = torch.zeros(seq_len, embedding_dim, device=x.device)

        pe[:, self.even_dims] = torch.sin(position * div_term)[:, : len(self.even_dims)]
        pe[:, self.odd_dims] = torch.cos(position * div_term)[:, : len(self.odd_dims)]
        return x + pe.to(dtype=x.dtype)

    def encode_text(self, x):
        if isinstance(x, str):
            x = [x]
        encodings = self.tokenizer.encode_batch(x)        # list of Encoding

        keep = None if self.max_len is None else self.max_len - 1
        ids = [torch.tensor([2] + e.ids[:keep], dtype=torch.long) for e in encodings]
        ids = nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=1)

        mask = ids != 1

        return ids, mask

    def forward(self, x):
        x, mask = self.encode_text(x)

        x = x.to(self.embedding.weight.device)
        mask = mask.to(self.embedding.weight.device)

        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder(x, src_key_padding_mask=~mask)
        x = x[:,0,:]

        x = self.classifier(x)
        return x