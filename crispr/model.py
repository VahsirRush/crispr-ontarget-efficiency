"""CNN + attention model for sgRNA on-target efficiency.

Layout follows the project spec:

    (30, C) one-hot [+ optional chromatin channel]
      -> Conv1d k=3, 64,  ReLU, BatchNorm
      -> Conv1d k=5, 128, ReLU, BatchNorm
      -> single-head self-attention over the 30 sequence positions
      -> attention-weighted pooling over positions (not max/avg)
      -> concat with the thermodynamic / gene-position vector
      -> Dense 64 -> 16 -> 1, dropout 0.3, sigmoid

The target (``score_drug_gene_rank``) is a rank scaled to (0, 1], so the head is
sigmoid rather than linear.

Two different weight sets are exposed for interpretability and they answer
different questions:

  ``pool_weights``  (N, 30)      how much each position contributes to the pooled
                                 representation that the head actually sees. This
                                 is the one to plot against sequence position.
  ``attn_weights``  (N, 30, 30)  the self-attention matrix, i.e. which positions
                                 each position reads from when building its
                                 contextualised feature.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Single-head scaled dot-product self-attention over sequence positions."""

    def __init__(self, dim: int, attn_dim: int = 64):
        super().__init__()
        self.q = nn.Linear(dim, attn_dim)
        self.k = nn.Linear(dim, attn_dim)
        self.v = nn.Linear(dim, dim)
        self.scale = attn_dim ** -0.5

    def forward(self, x):                      # x: (B, L, D)
        w = torch.softmax((self.q(x) @ self.k(x).transpose(1, 2)) * self.scale, dim=-1)
        return w @ self.v(x), w                # (B, L, D), (B, L, L)


class AttentionPool(nn.Module):
    """Learned scalar score per position -> softmax -> weighted sum over positions."""

    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x):                      # x: (B, L, D)
        w = torch.softmax(self.score(x).squeeze(-1), dim=-1)   # (B, L)
        return torch.einsum("bl,bld->bd", w, x), w


class CrisprCNN(nn.Module):
    """Channel widths are configurable because the spec's 64/128 stack has ~94k
    parameters for ~2.6k training guides, which memorises the training genes
    almost immediately. See ``scripts/03a_sweep_cnn.py`` for the selection."""

    def __init__(
        self,
        n_channels: int = 4,
        n_thermo: int = 10,
        dropout: float = 0.3,
        conv1_filters: int = 64,
        conv2_filters: int = 128,
        attn_dim: int = 64,
        conv_dropout: float = 0.0,
        positional: bool = True,
        seq_len: int = 30,
    ):
        super().__init__()
        c1, c2 = conv1_filters, conv2_filters
        self.conv1 = nn.Conv1d(n_channels, c1, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm1d(c1)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=5, padding="same")
        self.bn2 = nn.BatchNorm1d(c2)
        # Dropout1d zeroes whole feature maps, which regularises conv stacks far
        # better than element-wise dropout on strongly position-correlated features.
        self.cdrop = nn.Dropout1d(conv_dropout) if conv_dropout > 0 else nn.Identity()

        # Convolutions are translation-equivariant, so without this the network has
        # no way to tell position 4 from position 20. Rule Set 2 gets 58% of its
        # Gini importance from position-specific nucleotide identity, so absolute
        # position is most of the signal on this task.
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, c2)) if positional else None

        self.attn = SelfAttention(c2, attn_dim=attn_dim)
        self.pool = AttentionPool(c2)

        self.head = nn.Sequential(
            nn.Linear(c2 + n_thermo, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, seq, thermo, return_attn: bool = False):
        # seq: (B, L, C) -> conv wants (B, C, L)
        h = F.relu(self.bn1(self.conv1(seq.transpose(1, 2))))
        h = self.cdrop(h)
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.cdrop(h)
        h = h.transpose(1, 2)                  # (B, L, c2)
        if self.pos_emb is not None:
            h = h + self.pos_emb

        ctx, attn_w = self.attn(h)
        h = h + ctx                            # residual keeps the conv signal intact
        pooled, pool_w = self.pool(h)

        out = torch.sigmoid(self.head(torch.cat([pooled, thermo], dim=1))).squeeze(-1)
        if return_attn:
            return out, pool_w, attn_w
        return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _loader(seq, thermo, y, batch_size, shuffle):
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(seq), torch.from_numpy(thermo), torch.from_numpy(y)
    )
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_model(
    seq_tr, th_tr, y_tr,
    seq_val, th_val, y_val,
    n_channels: int = 4,
    epochs: int = 120,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
    seed: int = 0,
    verbose: bool = False,
    **model_kwargs,
):
    """Train with early stopping on validation Spearman.

    Validation Spearman (not MSE) is the stopping criterion because Spearman is
    the reported metric and the two do not always move together on a rank target.
    """
    from scipy import stats

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = CrisprCNN(n_channels=n_channels, n_thermo=th_tr.shape[1], **model_kwargs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    train_dl = _loader(seq_tr, th_tr, y_tr, batch_size, True)
    vs = torch.from_numpy(seq_val)
    vt = torch.from_numpy(th_val)

    best_score, best_state, best_epoch = -np.inf, None, -1
    for epoch in range(epochs):
        model.train()
        for xb, tb, yb in train_dl:
            opt.zero_grad()
            loss_fn(model(xb, tb), yb).backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            pv = model(vs, vt).numpy()
        score = stats.spearmanr(y_val, pv).statistic if np.std(pv) > 0 else -1.0

        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
        if verbose and epoch % 10 == 0:
            print(f"    epoch {epoch:3d}  val spearman {score:.4f}  (best {best_score:.4f})")

    model.load_state_dict(best_state)
    model.eval()
    return model, best_score, best_epoch


@torch.no_grad()
def predict(model, seq, thermo, batch_size: int = 512, return_attn: bool = False):
    model.eval()
    preds, pools, attns = [], [], []
    for i in range(0, len(seq), batch_size):
        s = torch.from_numpy(seq[i:i + batch_size])
        t = torch.from_numpy(thermo[i:i + batch_size])
        if return_attn:
            p, pw, aw = model(s, t, return_attn=True)
            pools.append(pw.numpy())
            attns.append(aw.numpy())
        else:
            p = model(s, t)
        preds.append(p.numpy())
    if return_attn:
        return np.concatenate(preds), np.concatenate(pools), np.concatenate(attns)
    return np.concatenate(preds)


class Standardizer:
    """Standardize the thermodynamic block using training statistics only."""

    def fit(self, X):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd < 1e-8] = 1.0
        return self

    def transform(self, X):
        return ((X - self.mu) / self.sd).astype(np.float32)
