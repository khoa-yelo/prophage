#!/usr/bin/env python3
"""
autoencoder_ts_with_val.py

Train a PyTorch MLP autoencoder on X (n×1152) with an 80/20 split.
Print train & test metrics (MSE and R²) each epoch.
"""

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class Autoencoder(nn.Module):
    def __init__(self, input_dim=1152, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(X_train, X_test,
                      latent_dim=64,
                      epochs=20,
                      batch_size=128,
                      lr=1e-3,
                      device='cuda'):
    """
    Trains on X_train, evaluates on X_test each epoch.
    Prints train/test MSE and R² per epoch.
    Returns the final model plus compression ratio.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Prepare DataLoaders
    X_tr_t = torch.from_numpy(X_train).float()
    train_ds = TensorDataset(X_tr_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_te_t = torch.from_numpy(X_test).float().to(device)

    model = Autoencoder(input_dim=X_train.shape[1], latent_dim=latent_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        # --- Training epoch ---
        model.train()
        running_train = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            rec = model(batch)
            loss = loss_fn(rec, batch)
            loss.backward()
            opt.step()
            running_train += loss.item() * batch.size(0)
        train_mse = running_train / len(train_ds)

        # --- Validation evaluation ---
        model.eval()
        with torch.no_grad():
            rec_test = model(X_te_t)
            test_mse = loss_fn(rec_test, X_te_t).item()
            # compute R² on test
            orig = X_test
            rec_np = rec_test.cpu().numpy()
            sse = np.sum((orig - rec_np) ** 2)
            sst = np.sum((orig - orig.mean(axis=0)) ** 2)
            test_r2 = 1 - sse / sst

        print(f"Epoch {epoch:02d} | Train MSE: {train_mse:.6f} | "
              f"Test MSE: {test_mse:.6f} | Test R²: {test_r2:.4f}")

    compression_ratio = latent_dim / X_train.shape[1]
    return model, compression_ratio


if __name__ == "__main__":
    # --- load or generate your data here ---
    np.random.seed(0)
    # Example placeholder data
    # X = np.random.randn(1000, 1152).astype(np.float32)
    # Or load your real embeddings:
    X = proteinEmbeddings.get_all_embeddings()["max_middle"]

    # 80/20 train-test split
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples\n")

    # run for a single latent_dim (or loop over several)
    latent_dim = 164
    model, comp_ratio = train_autoencoder(
        X_train, X_test,
        latent_dim=latent_dim,
        epochs=30,
        batch_size=256,
        lr=1e-4
    )

    print(f"\nFinal compression ratio: {latent_dim}/1152 = {comp_ratio*100:.2f}%")
    
    model.eval()
    compressed_dnafeat = model.encoder(torch.tensor(X).to("cuda")).detach().cpu().numpy()
    with h5py.File("compressed_dnafeat.h5", "w") as f:
        f.create_dataset("dnafeat", data = compressed_dnafeat)
        f.create_dataset("id", data = dnafeat_ids)
    model.eval()
    compressed_mean_middle = model.encoder(torch.tensor(\
                            proteinEmbeddings.get_all_embeddings()["mean_middle"]).to("cuda")).detach().cpu().numpy()
    model.eval()
    compressed_max = model.encoder(torch.tensor(\
                    proteinEmbeddings.get_all_embeddings()["max"]).to("cuda")).detach().cpu().numpy()
    model.eval()
    compressed_mean = model.encoder(torch.tensor(\
                    proteinEmbeddings.get_all_embeddings()["mean"]).to("cuda")).detach().cpu().numpy()
    model.eval()
    compressed_max_middle = model.encoder(torch.tensor(\
                    proteinEmbeddings.get_all_embeddings()["max_middle"]).to("cuda")).detach().cpu().numpy()

    with h5py.File("compressed_protein_embeddings.h5", 'w') as f:
        f.create_dataset('mean', data=compressed_mean)
        f.create_dataset('max', data=compressed_max)
        f.create_dataset('mean_middle', data=compressed_mean_middle)
        f.create_dataset('max_middle', data=compressed_max_middle)

        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('protein_ids',
                         data=proteinEmbeddings.protein_cluster_ids,
                         dtype=dt)
