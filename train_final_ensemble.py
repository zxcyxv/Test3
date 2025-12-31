import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import sys

# 표준 출력 버퍼링 해제 및 로그 함수
def log(msg):
    print(f"[LOG] {msg}", flush=True)

# 경로 및 설정
DATA_DIR = Path('/workspace/SoccerPredict/open_track1')
CSV_PATH = DATA_DIR / 'train_features_v2.csv'
FIELD_X, FIELD_Y = 105, 68
K_SEQ = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# 1. 모델 아키텍처 (Run 3 복구 및 MDN 강화)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)

class GatedTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.2):
        super().__init__()
        self.norm1 = RMSNorm(d_model); self.norm2 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model))
    def forward(self, x):
        n = self.norm1(x)
        attn_out, _ = self.attn(n, n, n)
        x = x + self.gate(n) * attn_out
        x = x + self.ffn(self.norm2(x))
        return x

class GeneralistModel(nn.Module):
    """Model A: 13.13m 사수를 위한 고정밀 Transformer"""
    def __init__(self, input_dim, d_model=128):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.randn(1, K_SEQ+1, d_model))
        self.blocks = nn.ModuleList([GatedTransformerBlock(d_model, 4) for _ in range(2)])
        # FiLM Generator
        self.cond_mlp = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, d_model*2))
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 4))
    def forward(self, x, pos):
        gamma_beta = self.cond_mlp(pos / torch.tensor([105., 68.]).to(DEVICE))
        g, b = gamma_beta.chunk(2, dim=-1)
        x = self.input_proj(x)
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1) + self.pos_emb
        for block in self.blocks:
            x = block(x)
            x = g.unsqueeze(1) * x + b.unsqueeze(1)
        return self.head(x[:, 0, :]).chunk(2, dim=-1)

class GoalLineSpecialistMDN(nn.Module):
    """Model B: res_id 상호작용 피처가 직접 주입된 MDN"""
    def __init__(self, input_dim=71, hidden_dim=128, K=2): # res_id 포함 71차원
        super().__init__()
        self.K = K
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.pi_head = nn.Linear(hidden_dim, K)
        self.mu_head = nn.Linear(hidden_dim, K*2)
        self.sigma_head = nn.Linear(hidden_dim, K*2)
        
        # Bias 초기화: 한 모드는 물리적 경계(105m)를 우선시하도록 유도
        with torch.no_grad():
            self.mu_head.bias[2] = 0.9 # Mode 2 (Out-of-play) x-coord
    def forward(self, shortcut):
        h = self.net(shortcut)
        pi = F.softmax(self.pi_head(h), dim=-1)
        mu = self.mu_head(h).view(-1, self.K, 2)
        sigma = F.softplus(self.sigma_head(h)) + 0.1
        return pi, mu, sigma.view(-1, self.K, 2)

class SpatialRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('B', torch.randn(6, 32) * 10.0) # Fourier Scale=10
        self.net = nn.Sequential(
            nn.Linear(6 + 64, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1)
        )
    def forward(self, pos):
        p_n = pos / torch.tensor([105., 68.]).to(DEVICE)
        d = torch.stack([pos[:,0]/105, (105-pos[:,0])/105, pos[:,1]/68, (68-pos[:,1])/68], dim=-1)
        feat = torch.cat([p_n, d], dim=-1)
        proj = 2 * np.pi * (feat @ self.B)
        fourier = torch.cat([feat, torch.cos(proj), torch.sin(proj)], dim=-1)
        return torch.sigmoid(self.net(fourier) / 0.05) # Hard Switching T=0.05

# =============================================================================
# 2. 데이터 및 손실 함수
# =============================================================================

def gmm_nll(y_true, pi, mu, sigma):
    y_ex = y_true.unsqueeze(1).expand_as(mu)
    log_gauss = -0.5 * (np.log(2*np.pi) + 2*torch.log(sigma) + ((y_ex-mu)/sigma)**2).sum(dim=-1)
    return -torch.logsumexp(torch.log(pi + 1e-10) + log_gauss, dim=-1).mean()

def train_pipeline():
    log("Pipeline Started. 12m Goal-line Challenge Mode.")
    df = pd.read_csv(CSV_PATH)
    
    # Feature indices
    base_feats = ['start_x', 'start_y', 'dt', 'ep_idx_norm', 'x_zone', 'lane', 'dist_to_goal', 'angle_to_goal', 'type_id', 'res_id']
    masked_feats = ['end_x', 'end_y', 'dx', 'dy']
    input_dim = len(base_feats) + len(masked_feats)
    res_id_idx = 9 # base_feats에서의 res_id 위치

    X_seq = np.zeros((len(df), K_SEQ, input_dim))
    for t in range(K_SEQ):
        for j, f in enumerate(base_feats): X_seq[:, t, j] = df[f'{f}_{t}'].fillna(0).values
        if t < K_SEQ - 1:
            for j, f in enumerate(masked_feats): X_seq[:, t, len(base_feats)+j] = df[f'{f}_{t}'].fillna(0).values

    y = df[['target_end_x', 'target_end_y']].values
    pos = df[['start_x_7', 'start_y_7']].values
    zones = np.zeros(len(df))
    zones[(df['target_end_x'] > 100) | (df['target_end_x'] < 5)] = 3
    
    scaler = StandardScaler()
    X_seq_norm = scaler.fit_transform(X_seq.reshape(-1, input_dim)).reshape(X_seq.shape)
    y_norm = (y - np.array([52.5, 34.0])) / np.array([52.5, 34.0])
    
    train_idx, val_idx = train_test_split(range(len(df)), test_size=0.2, random_state=42)
    def make_loader(idx):
        return DataLoader(TensorDataset(
            torch.FloatTensor(X_seq_norm[idx]), torch.FloatTensor(y_norm[idx]),
            torch.FloatTensor(pos[idx]), torch.LongTensor(zones[idx])
        ), batch_size=256, shuffle=True)
    train_loader, val_loader = make_loader(train_idx), make_loader(val_idx)

    # 모델 초기화
    model_a = GeneralistModel(input_dim).to(DEVICE)
    model_b = GoalLineSpecialistMDN(input_dim=71).to(DEVICE) # Fourier 70 + res_id 1
    router = SpatialRouter().to(DEVICE)
    
    # 스케줄러 추가 (Model A 정밀도 향상용)
    opt_a = optim.AdamW(model_a.parameters(), lr=1e-3, weight_decay=0.01)
    sch_a = optim.lr_scheduler.ReduceLROnPlateau(opt_a, 'min', patience=10, factor=0.5)
    opt_b_r = optim.AdamW(list(model_b.parameters()) + list(router.parameters()), lr=1e-3)

    # [Stage 1] Generalist A Training
    log("Stage 1: Generalist A Training (100 Epochs with Scheduler)...")
    for epoch in range(100):
        model_a.train(); l_sum = 0
        for x, y_t, p, z in train_loader:
            x, y_t, p = x.to(DEVICE), y_t.to(DEVICE), p.to(DEVICE)
            opt_a.zero_grad()
            mu, _ = model_a(x, p)
            # 인필드 샘플에 대해 손실 가중치 부여 (In-field 사수)
            loss = F.mse_loss(mu, y_t)
            loss.backward(); opt_a.step()
            l_sum += loss.item()
        sch_a.step(l_sum)
        if (epoch+1) % 20 == 0: log(f"Epoch {epoch+1}/100 - Loss: {l_sum/len(train_loader):.6f}")

    # [Stage 2] Specialist B & Router Training
    log("Stage 2: Specialist B (MDN) & Router Training...")
    for epoch in range(50):
        model_b.train(); router.train()
        l_mdn_sum, l_gate_sum = 0, 0
        for x, y_t, p, z in train_loader:
            x, y_t, p, z = x.to(DEVICE), y_t.to(DEVICE), p.to(DEVICE), z.to(DEVICE)
            
            # 1. Router 학습 (모든 샘플)
            opt_b_r.zero_grad()
            gate = router(p)
            loss_gate = F.binary_cross_entropy(gate, (z != 0).float().unsqueeze(1))
            
            # 2. Specialist B 학습 (Goal-line만, res_id 주입)
            mask = (z == 3)
            if mask.any():
                p_n = p[mask] / torch.tensor([105., 68.]).to(DEVICE)
                d = torch.stack([p[mask,0]/105, (105-p[mask,0])/105, p[mask,1]/68, (68-p[mask,1])/68], dim=-1)
                feat = torch.cat([p_n, d], dim=-1)
                proj = 2 * np.pi * (feat @ router.B)
                fourier = torch.cat([feat, torch.cos(proj), torch.sin(proj)], dim=-1)
                
                # res_id_7 직접 주입 (핵심!)
                res_id_7 = x[mask, -1, res_id_idx:res_id_idx+1]
                shortcut = torch.cat([fourier, res_id_7], dim=-1) # 70 + 1 = 71D
                
                pi, mu_b, sigma_b = model_b(shortcut)
                loss_mdn = gmm_nll(y_t[mask], pi, mu_b, sigma_b)
                (loss_mdn + loss_gate).backward()
                l_mdn_sum += loss_mdn.item()
            else:
                loss_gate.backward()
            opt_b_r.step(); l_gate_sum += loss_gate.item()
        if (epoch+1) % 10 == 0: log(f"Epoch {epoch+1}/50 - MDN Loss: {l_mdn_sum/len(train_loader):.4f}, Gate: {l_gate_sum/len(train_loader):.4f}")

    # [Stage 3] Final Evaluation
    log("Stage 3: Detailed Zone-wise Ensemble Evaluation...")
    model_a.eval(); model_b.eval(); router.eval()
    target_std = np.array([52.5, 34.0]); target_mean = np.array([52.5, 34.0])
    
    zone_errs = {0: [], 3: []} # 0: In-field, 3: Goal-line
    with torch.no_grad():
        for x, y_t, p, z in val_loader:
            x, y_t, p = x.to(DEVICE), y_t.to(DEVICE), p.to(DEVICE)
            mu_a, _ = model_a(x, p)
            
            # Specialist B Shortcut (res_id 포함)
            p_n = p / torch.tensor([105., 68.]).to(DEVICE)
            d = torch.stack([p[:,0]/105, (105-p[:,0])/105, p[:,1]/68, (68-p[:,1])/68], dim=-1)
            feat = torch.cat([p_n, d], dim=-1)
            proj = 2 * np.pi * (feat @ router.B)
            fourier = torch.cat([feat, torch.cos(proj), torch.sin(proj)], dim=-1)
            res_id_7 = x[:, -1, res_id_idx:res_id_idx+1]
            shortcut = torch.cat([fourier, res_id_7], dim=-1)
            
            pi, mu_b, _ = model_b(shortcut)
            pred_b = mu_b[torch.arange(len(mu_b)), pi.argmax(dim=1)]
            
            gate = router(p)
            # Safety Guard: 20 < x < 85 구역은 라우터 무시하고 Model A 강제
            gate[(p[:, 0] > 20) & (p[:, 0] < 85)] = 0.0
            final_pred = (1 - gate) * mu_a + gate * pred_b
            
            p_final = final_pred.cpu().numpy() * target_std + target_mean
            p_true = y_t.cpu().numpy() * target_std + target_mean
            dists = np.sqrt(((p_final - p_true)**2).sum(axis=1))
            
            for i, zid in enumerate(z.cpu().numpy()):
                if zid in zone_errs: zone_errs[zid].append(dists[i])
                
    log(f"In-field Result: {np.mean(zone_errs[0]):.4f}m")
    log(f"Goal-line Result: {np.mean(zone_errs[3]):.4f}m")
    overall = (np.mean(zone_errs[0]) * 1855 + np.mean(zone_errs[3]) * 178) / (1855 + 178)
    log(f"Estimated Overall Score: {overall:.4f}m")

if __name__ == "__main__":
    train_pipeline()