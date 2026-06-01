import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class WordMLMHead(nn.Module):
    def __init__(
            self,
            char_vocab_size: int,
            dim: int = 256,
            n_ctx_layers: int = 2,
            n_dec_layers: int = 1,
            n_heads: int = 4,
            max_sent_len: int = 32,
            max_word_len: int = 32,
            dropout: float = 0.1,
            mask_rate: float = 0.15,
            pad_id: int = 0,
            bos_id: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.mask_rate = mask_rate
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.max_word_len = max_word_len
        self.max_sent_len = max_sent_len
        self.char_vocab_size = char_vocab_size

        self.mask_token = nn.Parameter(torch.randn(dim) * 0.02)
        self.word_pos_embed = nn.Embedding(max_sent_len, dim)

        ctx_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(ctx_layer, num_layers=n_ctx_layers)

        self.char_embed = nn.Embedding(char_vocab_size, dim, padding_idx=pad_id)
        self.char_pos_embed = nn.Embedding(max_word_len, dim)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.char_decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec_layers)
        self.char_out_proj = nn.Linear(dim, char_vocab_size)

    def _sample_mask(
            self,
            attention_mask: torch.Tensor,
            override_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if override_mask is not None:
            return override_mask & attention_mask
        rnd = torch.rand_like(attention_mask, dtype=torch.float)
        return (rnd < self.mask_rate) & attention_mask

    def encode_context(
            self,
            word_embs: torch.Tensor,
            attention_mask: torch.Tensor,
            mask_positions: torch.Tensor,
    ) -> torch.Tensor:
        B, T, D = word_embs.shape
        device = word_embs.device

        mask_expanded = mask_positions.unsqueeze(-1)
        masked_embs = torch.where(
            mask_expanded,
            self.mask_token.view(1, 1, D),
            word_embs,
        )

        pos = torch.arange(T, device=device)
        masked_embs = masked_embs + self.word_pos_embed(pos)

        key_padding_mask = ~attention_mask
        ctx = self.context_encoder(masked_embs, src_key_padding_mask=key_padding_mask)
        return ctx

    def forward(
            self,
            word_embs: torch.Tensor,
            attention_mask: torch.Tensor,
            target_char_ids: torch.Tensor,
            override_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict]:
        B, T, D = word_embs.shape
        device = word_embs.device
        L = target_char_ids.size(-1)

        mask_pos = self._sample_mask(attention_mask, override_mask)
        n_masked = int(mask_pos.sum().item())

        if n_masked == 0:
            zero_loss = word_embs.sum() * 0.0
            return zero_loss, {"mlm_loss": 0.0, "n_masked": 0}

        ctx = self.encode_context(word_embs, attention_mask, mask_pos)

        ctx_at_mask = ctx[mask_pos]
        target_at_mask = target_char_ids[mask_pos]

        bos_col = torch.full(
            (n_masked, 1), self.bos_id,
            device=device, dtype=torch.long,
        )
        decoder_input_ids = torch.cat([bos_col, target_at_mask[:, :-1]], dim=1)

        char_emb = self.char_embed(decoder_input_ids)
        char_pos = torch.arange(L, device=device)
        char_emb = char_emb + self.char_pos_embed(char_pos)

        memory = ctx_at_mask.unsqueeze(1)

        causal = torch.triu(
            torch.ones(L, L, device=device, dtype=torch.bool),
            diagonal=1,
        )

        decoded = self.char_decoder(
            tgt=char_emb,
            memory=memory,
            tgt_mask=causal,
        )

        logits = self.char_out_proj(decoded)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_at_mask.reshape(-1),
            ignore_index=self.pad_id,
        )

        return loss, {"mlm_loss": loss.item(), "n_masked": n_masked}

    def parameter_count(self) -> dict:
        def count(m):
            return sum(p.numel() for p in m.parameters())

        return {
            "mask_token": self.mask_token.numel(),
            "word_pos_embed": count(self.word_pos_embed),
            "context_encoder": count(self.context_encoder),
            "char_embed": count(self.char_embed),
            "char_pos_embed": count(self.char_pos_embed),
            "char_decoder": count(self.char_decoder),
            "char_out_proj": count(self.char_out_proj),
            "total": count(self),
        }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, L, D, V = 4, 16, 32, 256, 75
    head = WordMLMHead(
        char_vocab_size=V,
        dim=D,
        n_ctx_layers=2,
        n_dec_layers=1,
        max_sent_len=T,
        max_word_len=L,
    ).to(device)
    print(f"Param count: {head.parameter_count()}")

    word_embs = torch.randn(B, T, D, device=device, requires_grad=True)
    attention_mask = torch.ones(B, T, dtype=torch.bool, device=device)
    attention_mask[:, 12:] = False
    target_char_ids = torch.randint(4, V, (B, T, L), device=device)

    head.train()
    loss, info = head(word_embs, attention_mask, target_char_ids)
    print(f"Train: loss={loss.item():.4f}, masked={info['n_masked']}")
    loss.backward()
    print(f"Backward OK (word_embs grad: {word_embs.grad is not None})")

    head.eval()
    forced_mask = torch.zeros_like(attention_mask)
    forced_mask[:, 2] = True
    forced_mask[:, 5] = True
    with torch.no_grad():
        loss_eval, info_eval = head(
            word_embs, attention_mask, target_char_ids,
            override_mask=forced_mask,
        )
    print(f"Eval (forced): loss={loss_eval.item():.4f}, masked={info_eval['n_masked']}")
