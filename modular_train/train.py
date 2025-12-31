import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import DATA_DIR, FIELD_X, FIELD_Y, K, MAX_LEN
from .data import (
    augment_data,
    compute_boundary_distance,
    create_boundary_labels,
    prepare_sequence_data,
)
from .losses import compute_loss
from .model import FiLMSpatialMoETransformer
from .schedule import TrainingScheduler
from .utils import log


def main():
    log("=" * 70)
    log("FiLM + MDN Spatial MoE Transformer")
    log("  - MDN: K=2 modes (In-play, Out-of-play)")
    log("  - FiLM: Magnitude Explosion mitigation")
    log("  - Spatial MoE: Sign Reversal mitigation")
    log("  - GMM NLL: Multi-modal prediction")
    log("=" * 70)

    log("\n[1] Data loading...")
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

    boundary_zone = create_boundary_labels(
        df_all["target_end_x"].values,
        df_all["target_end_y"].values,
    )
    boundary_dist = compute_boundary_distance(
        df_all["target_end_x"].values,
        df_all["target_end_y"].values,
    )

    train_idx, val_idx = torch.utils.data.random_split(
        range(len(df_all)),
        [int(len(df_all) * 0.8), len(df_all) - int(len(df_all) * 0.8)],
        generator=torch.Generator().manual_seed(42),
    )
    train_idx = np.array(train_idx.indices)
    val_idx = np.array(val_idx.indices)

    coords_train_orig = coords_seq[train_idx]
    cont_train_orig = cont_seq[train_idx]
    angles_train_orig = angles_seq[train_idx]
    cat_train_orig = cat_seq[train_idx]
    valid_mask_train_orig = valid_mask_seq[train_idx]
    cont_valid_mask_train_orig = cont_valid_mask_seq[train_idx]
    coords_val = coords_seq[val_idx]
    cont_val = cont_seq[val_idx]
    angles_val = angles_seq[val_idx]
    cat_val = cat_seq[val_idx]
    valid_mask_val = valid_mask_seq[val_idx]
    cont_valid_mask_val = cont_valid_mask_seq[val_idx]
    y_coords_train_orig = y_coords[train_idx]
    y_coords_val = y_coords[val_idx]
    boundary_dist_train_orig = boundary_dist[train_idx]
    boundary_dist_val = boundary_dist[val_idx]
    boundary_zone_train_orig = boundary_zone[train_idx]
    boundary_zone_val = boundary_zone[val_idx]

    log(f"  Train: {len(coords_train_orig)}, Val: {len(coords_val)}")
    log(f"  Zone distribution (Val): {np.bincount(boundary_zone_val)}")

    log("\n[2] Augmentation (Y-flip, 2x)...")
    padding_mask_train_orig = ~valid_mask_train_orig
    padding_mask_val = ~valid_mask_val

    (
        coords_train,
        cont_train,
        angles_train,
        cat_train,
        padding_mask_train,
        cont_valid_mask_train,
        y_coords_train,
        boundary_dist_train,
        boundary_zone_train,
    ) = augment_data(
        coords_train_orig,
        cont_train_orig,
        angles_train_orig,
        cat_train_orig,
        padding_mask_train_orig,
        cont_valid_mask_train_orig,
        y_coords_train_orig,
        boundary_dist_train_orig,
        boundary_zone_train_orig,
        cont_idx,
        angle_idx,
        cat_idx,
    )
    log(f"  Train (augmented): {len(coords_train)}")
    log(f"  Zone distribution (Train): {np.bincount(boundary_zone_train)}")

    log("\n[3] Normalization...")
    flat_cont = cont_train.reshape(-1, cont_train.shape[2])
    flat_mask = cont_valid_mask_train.reshape(-1, cont_train.shape[2])
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
        int(cat_train_orig[:, :, i].max()) + 1 for i in range(cat_train_orig.shape[2])
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"  Device: {device}")

    coords_train_t = torch.FloatTensor(coords_train).to(device)
    cont_train_t = torch.FloatTensor(cont_train).to(device)
    cat_train_t = torch.LongTensor(cat_train).to(device)
    padding_mask_train_t = torch.BoolTensor(padding_mask_train).to(device)
    y_train_t = torch.FloatTensor(y_coords_train_norm).to(device)
    boundary_dist_train_t = torch.FloatTensor(boundary_dist_train.reshape(-1, 1)).to(device)
    boundary_zone_train_t = torch.LongTensor(boundary_zone_train).to(device)

    coords_val_t = torch.FloatTensor(coords_val).to(device)
    cont_val_t = torch.FloatTensor(cont_val).to(device)
    cat_val_t = torch.LongTensor(cat_val).to(device)
    padding_mask_val_t = torch.BoolTensor(padding_mask_val).to(device)
    y_val_t = torch.FloatTensor(y_coords_val_norm).to(device)
    boundary_dist_val_t = torch.FloatTensor(boundary_dist_val.reshape(-1, 1)).to(device)
    boundary_zone_val_t = torch.LongTensor(boundary_zone_val).to(device)

    train_dataset = TensorDataset(
        coords_train_t,
        cont_train_t,
        cat_train_t,
        padding_mask_train_t,
        y_train_t,
        boundary_dist_train_t,
        boundary_zone_train_t,
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    log("\n[4] Training...")
    model = FiLMSpatialMoETransformer(
        num_cont_features=cont_train.shape[2],
        cat_cardinalities=cat_cardinalities,
        d_model=128,
        nhead=4,
        num_layers=2,
        dropout=0.2,
        fourier_mapping_size=32,
        fourier_scale=5.0,
        film_hidden_dim=64,
        router_hidden_dim=32,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15
    )
    training_scheduler = TrainingScheduler(warmup_epochs=10, anneal_epochs=90)

    best_dist = float("inf")
    best_state = None
    patience, patience_counter = 50, 0
    label_names = ["In-field", "Top-out", "Bottom-out", "Goal-line"]

    for epoch in range(200):
        model.train()
        temperature = training_scheduler.get_temperature(epoch)
        loss_weights = training_scheduler.get_loss_weights(epoch)

        total_losses = {"nll": 0, "mse_final": 0, "aux_dist": 0, "aux_zone": 0, "gate": 0}

        for coords_batch, cont_batch, cat_batch, mask_batch, y_batch, dist_batch, zone_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                coords_batch, cont_batch, cat_batch, mask_batch,
                temperature=temperature, return_all=True
            )
            loss, loss_dict = compute_loss(
                outputs, y_batch, dist_batch, zone_batch, **loss_weights
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for k in total_losses:
                if k in loss_dict:
                    total_losses[k] += loss_dict[k]

        model.eval()
        with torch.no_grad():
            outputs = model(
                coords_val_t, cont_val_t, cat_val_t, padding_mask_val_t,
                temperature=0.05, return_all=True
            )

            final_np = outputs["y_final"].cpu().numpy() * coord_scale
            avg_np = outputs["y_avg"].cpu().numpy() * coord_scale
            final_dist = np.sqrt(((final_np - y_coords_val) ** 2).sum(axis=1)).mean()
            avg_dist = np.sqrt(((avg_np - y_coords_val) ** 2).sum(axis=1)).mean()

            gate_np = outputs["gate"].cpu().numpy()
            gate_mean = gate_np.mean()
            gate_std = gate_np.std()

            pi_np = outputs["pi"].cpu().numpy()
            mode_selection = pi_np.argmax(axis=1)
            mode0_ratio = (mode_selection == 0).mean()
            mode1_ratio = (mode_selection == 1).mean()

            mu_A_np = outputs["mu_A"].cpu().numpy()
            mu_B_np = outputs["mu_B"].cpu().numpy()
            pi_A_np = outputs["pi_A"].cpu().numpy()
            pi_B_np = outputs["pi_B"].cpu().numpy()
            best_mode_A = pi_A_np.argmax(axis=1)
            best_mode_B = pi_B_np.argmax(axis=1)
            batch_idx = np.arange(len(y_coords_val))
            mu_A_best = mu_A_np[batch_idx, best_mode_A] * coord_scale
            mu_B_best = mu_B_np[batch_idx, best_mode_B] * coord_scale

            boundary_mask = boundary_zone_val != 0
            if boundary_mask.sum() > 0:
                mse_A_boundary = ((mu_A_best[boundary_mask] - y_coords_val[boundary_mask]) ** 2).sum(axis=1)
                mse_B_boundary = ((mu_B_best[boundary_mask] - y_coords_val[boundary_mask]) ** 2).sum(axis=1)
                b_better = (mse_B_boundary < mse_A_boundary).mean()
            else:
                b_better = 0.0

            zone_pred = outputs["aux_zone"].argmax(dim=1).cpu().numpy()
            zone_acc = (zone_pred == boundary_zone_val).mean()

        lr_scheduler.step(final_dist)

        if final_dist < best_dist:
            best_dist = final_dist
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            phase = "Warm-up" if epoch < 10 else ("Anneal" if epoch < 100 else "Fine-tune")
            log(
                f"  Epoch {epoch+1:3d} [{phase:8s}] T={temperature:.2f}: "
                f"Final={final_dist:.2f}m, Mode0={mode0_ratio:.1%}, "
                f"Gate={gate_mean:.3f}±{gate_std:.3f}"
            )

        if patience_counter >= patience:
            log(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    log(f"\n  Best Final Distance: {best_dist:.4f}m")

    log("\n[5] Final evaluation...")
    model.eval()
    with torch.no_grad():
        outputs = model(
            coords_val_t, cont_val_t, cat_val_t, padding_mask_val_t,
            temperature=0.05, return_all=True
        )

        pi_np = outputs["pi"].cpu().numpy()
        mu_np = outputs["mu"].cpu().numpy()
        sigma_np = outputs["sigma"].cpu().numpy()
        final_np = outputs["y_final"].cpu().numpy() * coord_scale
        avg_np = outputs["y_avg"].cpu().numpy() * coord_scale
        gate_np = outputs["gate"].cpu().numpy()
        gamma_np = outputs["gamma"].cpu().numpy()
        beta_np = outputs["beta"].cpu().numpy()

    best_mode = pi_np.argmax(axis=1)
    batch_idx = np.arange(len(y_coords_val))
    mu_best = mu_np[batch_idx, best_mode] * coord_scale

    best_mode_dist = np.sqrt(((mu_best - y_coords_val) ** 2).sum(axis=1)).mean()
    final_dist = np.sqrt(((final_np - y_coords_val) ** 2).sum(axis=1)).mean()
    avg_dist = np.sqrt(((avg_np - y_coords_val) ** 2).sum(axis=1)).mean()

    log("\n" + "=" * 70)
    log("[MDN Regression Performance]")
    log("=" * 70)
    log(f"  Max Mode (argmax pi) Distance: {final_dist:.4f}m")
    log(f"  Weighted Avg (sum pi mu) Distance: {avg_dist:.4f}m")
    log(f"  Improvement (Avg->Max): {avg_dist - final_dist:.4f}m")

    log("\n" + "=" * 70)
    log("[Mode Distribution]")
    log("=" * 70)
    mode0_ratio = (best_mode == 0).mean()
    mode1_ratio = (best_mode == 1).mean()
    log(f"  Mode 0 (In-play): {mode0_ratio:.1%}")
    log(f"  Mode 1 (Out-of-play): {mode1_ratio:.1%}")

    log("\n" + "=" * 70)
    log("[Zone-wise Performance]")
    log("=" * 70)
    log(f"  {'Zone':12s} | {'N':>5s} | {'MaxMode':>8s} | {'WgtAvg':>8s} | {'Gate':>6s} | {'Mode1%':>7s}")
    log("-" * 65)
    for i, name in enumerate(label_names):
        mask = boundary_zone_val == i
        if mask.sum() > 0:
            max_d = np.sqrt(((final_np[mask] - y_coords_val[mask]) ** 2).sum(axis=1)).mean()
            avg_d = np.sqrt(((avg_np[mask] - y_coords_val[mask]) ** 2).sum(axis=1)).mean()
            avg_gate = gate_np[mask].mean()
            mode1_pct = (best_mode[mask] == 1).mean()
            log(
                f"  {name:12s} | {mask.sum():5d} | {max_d:8.2f} | {avg_d:8.2f} | {avg_gate:6.3f} | {mode1_pct:6.1%}"
            )

    log("\n" + "=" * 70)
    log("[FiLM Statistics]")
    log("=" * 70)
    log(f"  gamma: mean={gamma_np.mean():.4f}, std={gamma_np.std():.4f}")
    log(f"  beta: mean={beta_np.mean():.4f}, std={beta_np.std():.4f}")

    log("\n" + "=" * 70)
    log("[Gate Distribution by Zone]")
    log("=" * 70)
    for i, name in enumerate(label_names):
        mask = boundary_zone_val == i
        if mask.sum() > 0:
            g = gate_np[mask]
            log(f"  {name:12s}: mean={g.mean():.3f}, std={g.std():.3f}, min={g.min():.3f}, max={g.max():.3f}")

    log("\n" + "=" * 70)
    log("[Mode Separation Analysis]")
    log("=" * 70)
    goalline_mask = boundary_zone_val == 3
    if goalline_mask.sum() > 0:
        mu_gl = mu_np[goalline_mask] * coord_scale[0]
        mode_diff = np.abs(mu_gl[:, 0, 0] - mu_gl[:, 1, 0]).mean()
        log(f"  Goal-line Mode Separation (x): {mode_diff:.2f}m")
        log("  (Expected: ~5m if modes are In-play vs Out-of-play)")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cont_mean": cont_mean,
            "cont_std": cont_std,
            "coord_scale": coord_scale,
            "cat_cardinalities": cat_cardinalities,
            "best_dist": best_dist,
        },
        DATA_DIR / "film_smoe_transformer.pt",
    )
    log(f"\nSaved: {DATA_DIR / 'film_smoe_transformer.pt'}")

    log("\n" + "=" * 70)
    log("Done!")
    log("=" * 70)


if __name__ == "__main__":
    main()
