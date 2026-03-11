import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
import gdown
from PIL import Image
import torchvision.transforms as transforms

# ============================================================
# Google Drive Model Download Config
# ============================================================
# HOW TO GET YOUR FILE ID:
# Your Google Drive share link looks like:
#   https://drive.google.com/file/d/XXXXXXXXX/view?usp=sharing
# The XXXXXXXXX part is your FILE_ID. Paste it below.

GDRIVE_FILE_ID = "1grmfto8OeeCV-EKK3SrxLhus4Bj0Ozxc"   # <-- REPLACE THIS
MODEL_FILENAME = "mae_checkpoint_epoch_55.pth"

# ============================================================
# Model Architecture (same classes from your notebook)
# ============================================================

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


def get_sinusoidal_positional_embedding(num_positions, embed_dim):
    position = torch.arange(num_positions).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
    pe = torch.zeros(num_positions, embed_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MAEEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.register_buffer('pos_embed', get_sinusoidal_positional_embedding(self.num_patches, embed_dim))
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        B, N, D = x.shape
        len_keep = mask.sum(dim=1)[0].item()
        noise = mask.float()
        ids_shuffle = torch.argsort(noise, dim=1, descending=True)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x, mask, ids_restore


class MAEDecoder(nn.Module):
    def __init__(self, num_patches=196, patch_size=16, in_channels=3,
                 encoder_embed_dim=768, decoder_embed_dim=384, depth=12, num_heads=6):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.register_buffer('decoder_pos_embed',
                             get_sinusoidal_positional_embedding(num_patches, decoder_embed_dim))
        self.blocks = nn.ModuleList([TransformerBlock(decoder_embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(decoder_embed_dim)
        self.pred = nn.Linear(decoder_embed_dim, patch_size * patch_size * in_channels)

    def forward(self, x, ids_restore):
        x = self.decoder_embed(x)
        B, num_visible, D = x.shape
        num_masked = self.num_patches - num_visible
        mask_tokens = self.mask_token.repeat(B, num_masked, 1)
        x_full = torch.cat([x, mask_tokens], dim=1)
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D))
        x_full = x_full + self.decoder_pos_embed
        for block in self.blocks:
            x_full = block(x_full)
        x_full = self.norm(x_full)
        pred = self.pred(x_full)
        return pred


class MaskedAutoencoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
                 decoder_embed_dim=384, decoder_depth=12, decoder_num_heads=6,
                 mask_ratio=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2
        self.in_channels = in_channels
        self.encoder = MAEEncoder(img_size, patch_size, in_channels, encoder_embed_dim, encoder_depth, encoder_num_heads)
        self.decoder = MAEDecoder(self.num_patches, patch_size, in_channels, encoder_embed_dim, decoder_embed_dim, decoder_depth, decoder_num_heads)

    def patchify(self, imgs):
        p = self.patch_size
        B, C, H, W = imgs.shape
        h = w = H // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, h * w, p * p * C)
        return x

    def unpatchify(self, x):
        p = self.patch_size
        h = w = int(self.num_patches ** 0.5)
        C = self.in_channels
        x = x.reshape(-1, h, w, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(-1, C, h * p, w * p)
        return x

    def generate_mask(self, batch_size, device, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        num_keep = int(self.num_patches * (1 - mask_ratio))
        # Ensure at least 1 patch visible
        num_keep = max(num_keep, 1)
        noise = torch.rand(batch_size, self.num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.zeros(batch_size, self.num_patches, device=device, dtype=torch.bool)
        mask.scatter_(1, ids_shuffle[:, :num_keep], True)
        return mask

    def forward(self, imgs, mask=None):
        if mask is None:
            mask = self.generate_mask(imgs.shape[0], imgs.device)
        latent, mask, ids_restore = self.encoder(imgs, mask)
        pred = self.decoder(latent, ids_restore)
        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * (~mask).float()).sum() / (~mask).float().sum()
        return loss, pred, mask


# ============================================================
# Streamlit App
# ============================================================

st.set_page_config(page_title="MAE Image Reconstruction", page_icon="🧠", layout="wide")

# Custom CSS for a polished look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #888;
        font-size: 1.05rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    .metric-card h3 {
        color: #667eea;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    .metric-card p {
        color: #fff;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
    }

    .image-label {
        text-align: center;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.5rem 0;
        color: #ccc;
    }

    .stSlider > div > div {
        color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🧠 Masked Autoencoder Reconstruction</h1>
    <p>Upload an image, choose a masking ratio, and watch the MAE reconstruct it</p>
</div>
""", unsafe_allow_html=True)

st.divider()


def download_model_if_needed():
    """Download model from Google Drive if not already present."""
    if not os.path.exists(MODEL_FILENAME):
        with st.spinner("📥 Downloading model weights from Google Drive (first time only)..."):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_FILENAME, quiet=False)
        if os.path.exists(MODEL_FILENAME):
            st.success("✅ Model downloaded successfully!")
        else:
            st.error("❌ Download failed. Check your Google Drive file ID and sharing settings.")
            st.stop()
    return MODEL_FILENAME


@st.cache_resource
def load_model():
    """Download (if needed) and load the MAE model (cached so it only loads once)."""
    model_path = download_model_if_needed()

    model = MaskedAutoencoder(
        img_size=224, patch_size=16, in_channels=3,
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=384, decoder_depth=12, decoder_num_heads=6,
        mask_ratio=0.75
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model, device


def preprocess_image(image):
    """Preprocess a PIL image for the model."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


def denormalize(tensor):
    """Undo ImageNet normalization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    return (tensor * std + mean).clamp(0, 1)


def reconstruct(model, image_tensor, mask_ratio, device):
    """Run reconstruction and return original, masked, and reconstructed images."""
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        mask = model.generate_mask(1, device, mask_ratio=mask_ratio / 100.0)
        latent, mask, ids_restore = model.encoder(image_tensor, mask)
        pred = model.decoder(latent, ids_restore)

    # Reconstructed image
    pred_img = model.unpatchify(pred)
    pred_img = denormalize(pred_img[0]).cpu().permute(1, 2, 0).numpy()

    # Original image
    orig_img = denormalize(image_tensor[0]).cpu().permute(1, 2, 0).numpy()

    # Masked image (black out masked patches)
    patches = model.patchify(image_tensor)
    masked_patches = patches.clone()
    masked_patches[~mask] = 0.0
    masked_img = model.unpatchify(masked_patches)
    masked_img = denormalize(masked_img[0]).cpu().permute(1, 2, 0).numpy()

    # Compute PSNR
    mse = np.mean((pred_img - orig_img) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')

    return orig_img, masked_img, pred_img, psnr


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    mask_ratio = st.slider(
        "🎭 Masking Ratio (%)",
        min_value=0,
        max_value=100,
        value=75,
        step=5,
        help="Percentage of image patches to mask before reconstruction"
    )

    st.divider()
    st.markdown("### 📊 Masking Info")
    num_patches = 196
    num_masked = int(num_patches * mask_ratio / 100)
    num_visible = num_patches - num_masked

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Visible", f"{num_visible}")
    with col2:
        st.metric("Masked", f"{num_masked}")

    st.caption(f"Image split into {num_patches} patches (14×14 grid of 16×16 px patches)")

# ============================================================
# Main Content
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Upload any image — it will be resized to 224×224"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Load model
    try:
        model, device = load_model()
        device_name = "GPU 🟢" if device.type == 'cuda' else "CPU 🟡"
        st.sidebar.markdown(f"**Device:** {device_name}")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

    # Preprocess and reconstruct
    image_tensor = preprocess_image(image)

    with st.spinner("🔄 Reconstructing..."):
        orig_img, masked_img, pred_img, psnr = reconstruct(model, image_tensor, mask_ratio, device)

    # Display results
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="image-label">📷 Original</div>', unsafe_allow_html=True)
        st.image(orig_img, use_container_width=True)

    with col2:
        st.markdown(f'<div class="image-label">🎭 Masked ({mask_ratio}%)</div>', unsafe_allow_html=True)
        st.image(masked_img, use_container_width=True)

    with col3:
        st.markdown('<div class="image-label">🔧 Reconstructed</div>', unsafe_allow_html=True)
        st.image(pred_img, use_container_width=True)

    # Metrics
    st.divider()
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>PSNR</h3>
            <p>{psnr:.2f} dB</p>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Masking Ratio</h3>
            <p>{mask_ratio}%</p>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Patches Visible</h3>
            <p>{num_visible} / {num_patches}</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # Empty state
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; color: #666;">
        <p style="font-size: 3rem; margin-bottom: 0.5rem;">🖼️</p>
        <p style="font-size: 1.2rem;">Upload an image to get started</p>
        <p style="font-size: 0.9rem;">The model will mask patches and attempt to reconstruct them</p>
    </div>
    """, unsafe_allow_html=True)
