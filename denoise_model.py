import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# Forward diffusion
def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


# Loss function for denoising
def p_losses(denoise_model, 
             x_start, 
             t, 
             cond, 
             sqrt_alphas_cumprod, 
             sqrt_one_minus_alphas_cumprod, 
             bert_emb=None,       # <--- Only BERT embedding here
             noise=None,
             loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    # Create noisy x
    x_noisy = q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=noise)

    # Pass x_noisy, cond, and bert_emb to the denoising model
    predicted_noise = denoise_model(x_noisy, t, cond, bert_emb=bert_emb)

    # Compute loss
    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

# Position embeddings
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# Denoising model
class DenoiseNN(nn.Module):
    """
    Minimal changes: We only concatenate cond (n_cond dims) 
    and a BERT embedding (bert_dim=768 by default).
    """
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 n_layers, 
                 n_cond, 
                 d_cond, 
                 bert_dim=768):
        super(DenoiseNN, self).__init__()
        self.n_layers  = n_layers
        self.n_cond    = n_cond
        self.bert_dim  = bert_dim

        self.cond_mlp = nn.Sequential(
            nn.Linear(n_cond + bert_dim, 4*d_cond),
            nn.ReLU(),
            nn.Linear(4*d_cond, d_cond),
        )

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.ReLU(), 
            nn.Linear(4*hidden_dim, hidden_dim),
        )

        mlp_layers = []
        mlp_layers.append(nn.Linear(input_dim + d_cond, hidden_dim))
        for i in range(n_layers - 2):
            mlp_layers.append(nn.Linear(hidden_dim + d_cond, hidden_dim))
        mlp_layers.append(nn.Linear(hidden_dim, input_dim))
        self.mlp = nn.ModuleList(mlp_layers)

        bn_layers = [nn.BatchNorm1d(hidden_dim) for _ in range(n_layers - 1)]
        self.bn = nn.ModuleList(bn_layers)

        self.relu = nn.ReLU()

    def forward(self, x, t, cond, bert_emb=None):
        """
        x:        [batch_size, input_dim]
        t:        [batch_size] (timesteps)
        cond:     [batch_size, n_cond]
        bert_emb: [batch_size, bert_dim]
        """
        if bert_emb is None:
            bert_emb = torch.zeros(x.size(0), self.bert_dim, device=x.device)

        # Concatenate cond + BERT
        cond_cat = torch.cat([cond, bert_emb], dim=1)
        cond_cat = torch.nan_to_num(cond_cat, nan=-100.0)
        cond_cat = self.cond_mlp(cond_cat)

        # Compute time embedding
        t_embed = self.time_mlp(t)

        # Pass through each MLP layer
        for i in range(self.n_layers - 1):
            # Concatenate x and the cond embedding
            x = torch.cat((x, cond_cat), dim=1)
            x = self.relu(self.mlp[i](x)) + t_embed
            x = self.bn[i](x)

        # Final layer
        x = self.mlp[self.n_layers - 1](x)
        return x

@torch.no_grad()
def p_sample(model, x, t, cond, t_index, betas, bert_emb=None):
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # model (noise predictor) -> predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t, cond, bert_emb=bert_emb) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, cond, timesteps, betas, shape, bert_emb=None):
    device = next(model.parameters()).device
    b = shape[0]
    # Start from pure noise
    img = torch.randn(shape, device=device)
    imgs = []

    for i in reversed(range(0, timesteps)):
        img = p_sample(
            model,
            x=img,
            t=torch.full((b,), i, device=device, dtype=torch.long),
            cond=cond,
            t_index=i,
            betas=betas,
            bert_emb=bert_emb
        )
        imgs.append(img)
    return imgs

@torch.no_grad()
def sample(model, cond, latent_dim, timesteps, betas, batch_size, bert_emb=None):
    """
    cond:     [batch_size, n_cond]
    bert_emb: [batch_size, bert_dim]
    """
    return p_sample_loop(model, cond, timesteps, betas, shape=(batch_size, latent_dim), bert_emb=bert_emb)
