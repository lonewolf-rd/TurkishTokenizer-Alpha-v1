from src.model_development.utils.providers.logger_provider import global_logger
from src.model_development.utils.text_utils import turkish_lower
from src.model_development.model.boundary_detector import RotaryEmbedding
from typing import Tuple, Dict, Union, Optional, List
import torch.nn.functional as F
import torch.nn as nn
import torch


class CharEncoderHelper:

    _PAD_ID: int = 0
    _UNK_ID: int = 1
    _BOS_ID: int = 2
    _EOS_ID: int = 3
    _CHAR_OFFSET: int = 4

    _TURKISH_CHARS: str = (
        "abcçdefgğhıijklmnoöprsştuüvyz"
        "0123456789"
        " .,!?;:'\"-()[]{}/@#%&*+=<>~`^_\\"
        "\n\t"
    )

    def __init__(self):
        self.char_vocab: Union[None, Dict[str, int]] = None
        self.char_vocab_size: Union[None, int] = None

        self.char_vocab = self._build_char_vocab()
        self.char_vocab_size = len(self.char_vocab)

    def _build_char_vocab(self) -> Optional[Dict[str, int]]:
        try:
            vocab = {
                "<PAD>": self._PAD_ID,
                "<UNK>": self._UNK_ID,
                "<BOS>": self._BOS_ID,
                "<EOS>": self._EOS_ID,
            }
            for i, c in enumerate(self._TURKISH_CHARS):
                if c not in vocab:
                    vocab[c] = i + self._CHAR_OFFSET

            global_logger.info(
                f"[CharEncoderHelper](_build_char_vocab) Built char vocab (size={len(vocab)})"
            )
            return vocab

        except Exception as err:
            global_logger.error(f"[CharEncoderHelper](_build_char_vocab) Failed: {err}")
            return None

    def word_to_char_ids(
            self,
            word: str,
            max_len: Optional[int] = 32,
            add_bos: Optional[bool] = True,
            add_eos: Optional[bool] = True,
    ) -> Tuple[List[int], List[int], int]:
        id_list: List[int] = []
        case_list: List[int] = []

        if add_bos:
            id_list.append(self._BOS_ID)
            case_list.append(0)

        reserved = 2 if (add_bos and add_eos) else (1 if (add_bos or add_eos) else 0)
        for c in word[: max_len - reserved]:
            lower = turkish_lower(c)
            is_upper = 1 if lower != c else 0
            id_list.append(self.char_vocab.get(lower, self._UNK_ID))
            case_list.append(is_upper)

        if add_eos:
            id_list.append(self._EOS_ID)
            case_list.append(0)

        real_length = len(id_list)

        pad_count = max_len - real_length
        id_list += [self._PAD_ID] * pad_count
        case_list += [0] * pad_count

        return id_list[:max_len], case_list[:max_len], real_length


class MultiScaleCNN(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            kernels: Tuple[int, ...] = (2, 3, 4, 5, 6),
            dropout: float = 0.1,
    ):
        super().__init__()
        self.kernels = kernels
        per_kernel = out_dim // len(self.kernels)
        remainder = out_dim - per_kernel * len(self.kernels)
        self.convs = nn.ModuleList()

        for i, k in enumerate(self.kernels):
            channels = per_kernel + (remainder if i == len(self.kernels) - 1 else 0)
            self.convs.append(nn.Sequential(
                nn.Conv1d(
                    in_channels=in_dim,
                    out_channels=channels,
                    kernel_size=k,
                    padding=k // 2,
                ),
                nn.GELU(),
                nn.Dropout(dropout),
            ))

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)

        outs = []
        for conv in self.convs:
            out = conv(x_t)
            out = out[:, :, :x.size(1)]
            outs.append(out)

        concat = torch.cat(outs, dim=1)
        out = concat.transpose(1, 2)
        return self.norm(out)


class LocalSelfAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 4,
            dropout: float = 0.1,
            window_size: int = None,
            max_seq_len: int = 64,
    ):
        super().__init__()

        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

        self.rope = RotaryEmbedding(dim=self.head_dim, max_seq_len=max_seq_len)

    def forward(
            self,
            x: torch.Tensor,
            padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:

        B, S, D = x.shape
        residual = x

        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, S, self.num_heads, self.head_dim)
                   .transpose(1, 2) for t in qkv]

        q, k = self.rope(q, k)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if self.window_size is not None:
            window_mask = self._make_window_mask(S, self.window_size, x.device)
            scores = scores.masked_fill(window_mask, float("-inf"))

        if padding_mask is not None:
            pad_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(pad_mask, float("-inf"))

        attn_scores = F.softmax(scores, dim=-1)
        attn_scores = torch.nan_to_num(attn_scores, nan=0.0)

        if padding_mask is not None:
            query_mask = padding_mask.unsqueeze(1).unsqueeze(-1)
            attn_scores = attn_scores.masked_fill(query_mask, 0.0)

        attn_scores = self.dropout(attn_scores)

        out = torch.matmul(attn_scores, v)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.proj(out)

        return self.norm(out + residual)

    @staticmethod
    def _make_window_mask(
            seq_len: int,
            window_size: int,
            device: torch.device,
    ) -> torch.Tensor:
        idx = torch.arange(seq_len, device=device)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        mask = dist > window_size
        return mask.unsqueeze(0).unsqueeze(0)


class CharEncoder(nn.Module):
    def __init__(
            self,
            char_embed_dim: int = 56,
            case_embed_dim: int = 8,
            char_dim: int = 256,
            n_attn_layers: int = 2,
            num_heads: int = 4,
            kernels: tuple = (2, 3, 4, 5, 6),
            max_word_len: int = 32,
            dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder_helper = CharEncoderHelper()
        char_vocab_size = self.encoder_helper.char_vocab_size
        self.char_dim = char_dim
        self.max_word_len = max_word_len

        self.char_embedding = nn.Embedding(
            num_embeddings=char_vocab_size,
            embedding_dim=char_embed_dim,
            padding_idx=self.encoder_helper._PAD_ID,
        )

        self.case_embedding = nn.Embedding(
            num_embeddings=2,
            embedding_dim=case_embed_dim,
        )

        combined_dim = char_embed_dim + case_embed_dim
        self.embed_proj = nn.Linear(combined_dim, char_dim)

        self.cnn = MultiScaleCNN(
            in_dim=char_dim,
            out_dim=char_dim,
            kernels=kernels,
            dropout=dropout,
        )

        self.attn_layers = nn.ModuleList([
            LocalSelfAttention(
                dim=char_dim,
                num_heads=num_heads,
                dropout=dropout,
                window_size=8,
            )
            for _ in range(n_attn_layers)
        ])

        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(char_dim),
                nn.Linear(char_dim, char_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(char_dim * 2, char_dim),
                nn.Dropout(dropout),
            )
            for _ in range(n_attn_layers)
        ])

        self.final_norm = nn.LayerNorm(char_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            char_ids: torch.Tensor,
            case_flags: torch.Tensor,
            padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if padding_mask is None:
            padding_mask = (char_ids == self.encoder_helper._PAD_ID)

        char_e = self.char_embedding(char_ids)
        case_e = self.case_embedding(case_flags)
        x = torch.cat([char_e, case_e], dim=-1)
        x = self.dropout(x)
        x = self.embed_proj(x)

        x = x + self.cnn(x)

        for attn, ffn in zip(self.attn_layers, self.ffns):
            x = attn(x, padding_mask)
            x = x + ffn(x)

        x = self.final_norm(x)
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return x

    def parameter_count(self) -> dict:
        def count(m):
            return sum(p.numel() for p in m.parameters())

        return {
            "char_embedding": count(self.char_embedding),
            "case_embedding": count(self.case_embedding),
            "embed_proj": count(self.embed_proj),
            "cnn": count(self.cnn),
            "attn_layers": count(self.attn_layers),
            "ffns": count(self.ffns),
            "total": count(self),
        }


if __name__ == "__main__":
    helper = CharEncoderHelper()
    test_words = ["Muhasebeleştirme", "İstanbul", "evlerdekiler", "TBMM"]
    for w in test_words:
        ids, flags, rl = helper.word_to_char_ids(w, max_len=32)
        print(f"{w:25s} | real_len={rl} | flags={flags[:rl]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = CharEncoder(char_dim=256, n_attn_layers=2, dropout=0.1).to(device)
    print(f"\nParam count: {encoder.parameter_count()}")

    batch_ids, batch_flags, lengths = [], [], []
    for w in test_words:
        ids, flags, rl = helper.word_to_char_ids(w, max_len=32)
        batch_ids.append(ids)
        batch_flags.append(flags)
        lengths.append(rl)

    char_ids = torch.tensor(batch_ids, device=device)
    case_flags = torch.tensor(batch_flags, device=device)
    out = encoder(char_ids, case_flags)
    print(f"Output shape: {out.shape}")
