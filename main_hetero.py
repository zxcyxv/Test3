import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from train_transformer_heteroscedastic import (
    DATA_DIR,
    K,
    MAX_LEN,
    HeteroscedasticTransformer,
    create_boundary_labels,
    gaussian_nll_loss,
    log,
    prepare_sequence_data,
)


def main():
    log("=" * 70)
    log("Heteroscedastic Transformer (No Y-Flip)")
    log("  - Selective Fourier Mapping")
    log("  - Aleatoric Uncertainty")
    log("=" * 70)

    log("\n[1] 데이터 로드...")
    df_all = pd.read_csv(DATA_DIR / "train_features_v2.csv")
    (
        coords_seq,
        cont_seq,
        angles_seq,
        cat_seq,
        valid_mask_seq,
        cont_valid_mask_seq,
        cont_idx,
        angle_idx,
        cat_idx,
    ) = prepare_sequence_data(df_all, K)
    y_coords = df_all[["target_end_x", "target_end_y"]].values
    y_labels = create_boundary_labels(
        df_all["target_end_x"].values, df_all["target_end_y"].values
    )

    train_idx, val_idx = torch.utils.data.random_split(
        range(len(df_all)),
        [int(len(df_all) * 0.8), len(df_all) - int(len(df_all) * 0.8)],
        generator=torch.Generator().manual_seed(42),
    )
    train_idx = np.array(train_idx.indices)
    val_idx = np.array(val_idx.indices)

    coords_train = coords_seq[train_idx]
    cont_train = cont_seq[train_idx]
    angles_train = angles_seq[train_idx]
    cat_train = cat_seq[train_idx]
    valid_mask_train = valid_mask_seq[train_idx]
    cont_valid_mask_train = cont_valid_mask_seq[train_idx]

    coords_val = coords_seq[val_idx]
    cont_val = cont_seq[val_idx]
    angles_val = angles_seq[val_idx]
    cat_val = cat_seq[val_idx]
    valid_mask_val = valid_mask_seq[val_idx]
    cont_valid_mask_val = cont_valid_mask_seq[val_idx]

    y_coords_train = y_coords[train_idx]
    y_coords_val = y_coords[val_idx]
    y_labels_val = y_labels[val_idx]

    log(f"  Train: {len(coords_train)}, Val: {len(coords_val)}")

    padding_mask_train = ~valid_mask_train
    padding_mask_val = ~valid_mask_val

    log("\n[2] 정규화...")
    n_samples, n_steps, cont_dim = cont_train.shape
    flat_cont = cont_train.reshape(-1, cont_dim)
    flat_mask = cont_valid_mask_train.reshape(-1, cont_dim)
    counts = flat_mask.sum(axis=0)
    counts = np.where(counts == 0, 1, counts)
    cont_mean = (flat_cont * flat_mask).sum(axis=0) / counts
    cont_var = ((flat_cont - cont_mean) ** 2 * flat_mask).sum(axis=0) / counts
    cont_std = np.sqrt(cont_var)
    cont_std = np.where(cont_std < 1e-6, 1.0, cont_std)

    cont_train = (cont_train - cont_mean) / cont_std
    cont_train[~cont_valid_mask_train] = 0.0
    cont_val = (cont_val - cont_mean) / cont_std
    cont_val[~cont_valid_mask_val] = 0.0

    cont_train = np.concatenate([cont_train, angles_train], axis=-1)
    cont_val = np.concatenate([cont_val, angles_val], axis=-1)

    coord_scale = np.array([MAX_LEN, MAX_LEN])
    y_coords_train_norm = y_coords_train / coord_scale
    y_coords_val_norm = y_coords_val / coord_scale

    cat_cardinalities = tuple(
        int(cat_train[:, :, i].max()) + 1 for i in range(cat_train.shape[2])
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"  Device: {device}")

    coords_train_t = torch.FloatTensor(coords_train).to(device)
    cont_train_t = torch.FloatTensor(cont_train).to(device)
    cat_train_t = torch.LongTensor(cat_train).to(device)
    padding_mask_train_t = torch.BoolTensor(padding_mask_train).to(device)
    y_train_t = torch.FloatTensor(y_coords_train_norm).to(device)

    coords_val_t = torch.FloatTensor(coords_val).to(device)
    cont_val_t = torch.FloatTensor(cont_val).to(device)
    cat_val_t = torch.LongTensor(cat_val).to(device)
    padding_mask_val_t = torch.BoolTensor(padding_mask_val).to(device)
    y_val_t = torch.FloatTensor(y_coords_val_norm).to(device)

    train_dataset = TensorDataset(
        coords_train_t, cont_train_t, cat_train_t, padding_mask_train_t, y_train_t
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    model = HeteroscedasticTransformer(
        num_cont_features=cont_train.shape[2],
        cat_cardinalities=cat_cardinalities,
        d_model=128,
        nhead=4,
        num_layers=2,
        dropout=0.3,
        fourier_mapping_size=32,
        fourier_scale=5.0,
    ).to(device)

    log(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    w_nll = 1.0
    w_final = 5.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    best_dist = float("inf")
    best_state = None
    patience, patience_counter = 25, 0

    for epoch in range(150):
        model.train()
        total_nll, total_mse = 0, 0

        for coords_batch, cont_batch, cat_batch, mask_batch, y_batch in train_loader:
            optimizer.zero_grad()

            mu, log_var, sigma, residual, y_final = model(
                coords_batch, cont_batch, cat_batch, mask_batch, return_all=True
            )

            nll_base = gaussian_nll_loss(y_batch, mu, log_var)
            y_final = mu + sigma.detach() * residual
            mse_final = F.mse_loss(y_final, y_batch)
            loss = w_nll * nll_base + w_final * mse_final

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_nll += nll_base.item()
            total_mse += mse_final.item()

        model.eval()
        with torch.no_grad():
            mu, log_var, sigma, residual, y_final = model(
                coords_val_t, cont_val_t, cat_val_t, padding_mask_val_t, return_all=True
            )

            mu_orig = mu.cpu().numpy() * coord_scale
            mu_dist = np.sqrt(((mu_orig - y_coords_val) ** 2).sum(axis=1)).mean()

            final_orig = y_final.cpu().numpy() * coord_scale
            final_dist = np.sqrt(((final_orig - y_coords_val) ** 2).sum(axis=1)).mean()

        scheduler.step(final_dist)

        if final_dist < best_dist:
            best_dist = final_dist
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            n_batches = len(train_loader)
            avg_sigma = sigma.mean().item()
            log(
                f"  Epoch {epoch+1:3d}: NLL={total_nll/n_batches:.4f}, "
                f"μ={mu_dist:.2f}m, Final={final_dist:.2f}m, σ_avg={avg_sigma:.3f}"
            )

        if patience_counter >= patience:
            log(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    log(f"\n  Best Final Distance: {best_dist:.4f}m")

    log("\n[4] 최종 평가...")
    model.eval()
    with torch.no_grad():
        mu, log_var, sigma, residual, y_final = model(
            coords_val_t, cont_val_t, cat_val_t, padding_mask_val_t, return_all=True
        )

        mu_np = mu.cpu().numpy() * coord_scale
        sigma_np = sigma.cpu().numpy()
        residual_np = residual.cpu().numpy()
        final_np = y_final.cpu().numpy() * coord_scale

    mu_dist = np.sqrt(((mu_np - y_coords_val) ** 2).sum(axis=1)).mean()
    final_dist = np.sqrt(((final_np - y_coords_val) ** 2).sum(axis=1)).mean()

    log("\n" + "=" * 70)
    log("[Regression Performance]")
    log("=" * 70)
    log(f"  Base (μ) Distance:     {mu_dist:.4f}m")
    log(f"  Final (μ+σ⊙r) Distance: {final_dist:.4f}m")
    log(f"  Improvement:           {mu_dist - final_dist:.4f}m")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cont_mean": cont_mean,
            "cont_std": cont_std,
            "coord_scale": coord_scale,
            "cat_cardinalities": cat_cardinalities,
            "best_dist": best_dist,
        },
        DATA_DIR / "transformer_heteroscedastic_noflip.pt",
    )
    log(f"\n저장: {DATA_DIR / 'transformer_heteroscedastic_noflip.pt'}")

    log("\n" + "=" * 70)
    log("완료!")
    log("=" * 70)


if __name__ == "__main__":
    main()
