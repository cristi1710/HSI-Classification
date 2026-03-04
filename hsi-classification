import os
import gc
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
import warnings

# General configuration
warnings.filterwarnings("ignore")
BASE_PATH = '/kaggle/input/hsi-classification'
SEEDS = [42, 1010, 2002]  # 3 seeds for stability and reproducibility
BATCH_SIZE = 64
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 0
NUM_CLASSES = 7


# Hybrid architecture with 3 branches: 1D spectral, 3D spatial, and 2D global
class HSI_Hybrid_Network(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        # 1D branch: extracts spectral features from the central pixel
        self.depth_features = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # 3D branch: captures joint spatial-spectral context from the full patch volume
        self.spatial_features = nn.Sequential(
            nn.Conv3d(1, 16, (5, 3, 3), padding=(0, 1, 1)), nn.BatchNorm3d(16), nn.ReLU(),
            nn.Conv3d(16, 32, (3, 3, 3), padding=(0, 1, 1)), nn.BatchNorm3d(32), nn.ReLU()
        )

        # 2D branch: refines spatial features after collapsing the spectral dimension
        self.global_features = nn.Sequential(
            nn.Conv2d(32 * 42, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        # Classification head: concatenates outputs from the spectral and spatial branches
        self.classifier_head = nn.Sequential(
            nn.Linear(128 + 512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Extract the central pixel for the spectral branch
        center_pixel = x[:, :, 9, 9].unsqueeze(1)
        d_out = self.depth_features(center_pixel).flatten(1)

        # Process the 3D volume and collapse the spectral dimension
        s_out = self.spatial_features(x.unsqueeze(1))
        b, c, d, h, w = s_out.shape
        s_out = s_out.view(b, c * d, h, w)
        s_out = self.global_features(s_out).flatten(1)

        return self.classifier_head(torch.cat([d_out, s_out], dim=1))


# MixUp augmentation: blends two samples for regularization
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# Dataset with geometric augmentations: horizontal/vertical flip and random rotation
class CustomTransformedDataset(Dataset):
    def __init__(self, data_tensor, targets=None, apply_transforms=False):
        self.data = data_tensor
        self.targets = targets
        self.apply_transforms = apply_transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx].copy()
        if self.apply_transforms:
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=2).copy()
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=1).copy()
            k = np.random.randint(0, 4)
            if k > 0:
                image = np.rot90(image, k=k, axes=(1, 2)).copy()
        tensor_x = torch.from_numpy(image)
        if self.targets is not None:
            return tensor_x, torch.tensor(self.targets[idx], dtype=torch.long)
        return tensor_x


# Pre-allocated RAM buffer for efficient loading of .npy patch files
class EfficientRAMLoader:
    def __init__(self, max_memory_gb=3.5):
        single_item_size = 48 * 19 * 19 * 4
        self.max_items = int((max_memory_gb * 1024 ** 3) / single_item_size)
        self.buffer = np.zeros((self.max_items, 48, 19, 19), dtype=np.float32)
        self.y_list, self.names_list, self.count = [], [], 0

    def push(self, array_data, label, filename=None):
        if self.count < self.max_items:
            self.buffer[self.count] = array_data
            if label is not None:
                self.y_list.append(label)
            if filename is not None:
                self.names_list.append(filename)
            self.count += 1

    def get_arrays(self):
        return self.buffer[:self.count], np.array(self.y_list), np.array(self.names_list)


def prepare_all_data():
    print(">>> Data Prep: Loading All 7 Classes...")
    df_train = pd.read_csv(os.path.join(BASE_PATH, 'labels.csv'))

    # Keep only the 7 valid classes and remap labels to indices 0-6
    valid_classes = [1, 2, 3, 4, 5, 6, 7]
    df_train = df_train[df_train['label'].isin(valid_classes)].reset_index(drop=True)
    mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}

    train_loader = EfficientRAMLoader(max_memory_gb=2.5)
    for _, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Reading Train"):
        path = os.path.join(BASE_PATH, 'train',
                            row['filename'] if row['filename'].endswith('.npy') else row['filename'] + '.npy')
        try:
            arr = np.load(path).astype(np.float32)
            if arr.shape[-1] == 48:
                arr = arr.transpose(2, 0, 1)
            train_loader.push(arr, mapping[row['label']])
        except:
            continue
    X_tr, y_tr, _ = train_loader.get_arrays()

    df_test = pd.read_csv(os.path.join(BASE_PATH, 'sample_submission.csv'))
    test_loader = EfficientRAMLoader(max_memory_gb=3.5)
    for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Reading Test"):
        path = os.path.join(BASE_PATH, 'test',
                            row['filename'] if row['filename'].endswith('.npy') else row['filename'] + '.npy')
        try:
            arr = np.load(path).astype(np.float32)
            if arr.shape[-1] == 48:
                arr = arr.transpose(2, 0, 1)
            test_loader.push(arr, -1, row['filename'])
        except:
            test_loader.push(np.zeros((48, 19, 19), dtype=np.float32), -1, row['filename'])
    X_ts, _, fnames_ts = test_loader.get_arrays()

    # Global normalization computed from training set statistics
    g_mean = X_tr.mean(axis=(0, 2, 3), keepdims=True)
    g_std = X_tr.std(axis=(0, 2, 3), keepdims=True)
    X_tr = (X_tr - g_mean) / (g_std + 1e-8)
    X_ts = (X_ts - g_mean) / (g_std + 1e-8)
    return X_tr, y_tr, X_ts, fnames_ts


def execute_training_pipeline():
    start_time = time.time()
    X_train_np, y_train_np, X_test_np, test_files = prepare_all_data()

    ds_testing = CustomTransformedDataset(X_test_np, apply_transforms=False)
    aggregated_probs = np.zeros((len(ds_testing), NUM_CLASSES))
    seeds_completed = 0

    # Target distribution used for post-processing calibration of predictions
    target_dist = {
        1: 900, 2: 3000, 3: 800,
        4: 1, 5: 1,
        6: 13500, 7: 2200
    }
    idx_to_label_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7}

    for seed_val in SEEDS:
        print(f"\n{'='*30}\n STARTING SEED: {seed_val}\n{'='*30}")
        torch.manual_seed(seed_val)
        np.random.seed(seed_val)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_val)

        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed_val)
        current_seed_probs = np.zeros((len(ds_testing), NUM_CLASSES))

        for fold_idx, (idx_tr, idx_val) in enumerate(splitter.split(np.zeros(len(y_train_np)), y_train_np)):
            elapsed = (time.time() - start_time) / 3600
            if elapsed > 8.5:
                break

            print(f"--- Seed {seed_val} | Fold {fold_idx + 1}/3 ---")

            dset_train = CustomTransformedDataset(X_train_np[idx_tr], y_train_np[idx_tr], apply_transforms=True)
            dset_val = CustomTransformedDataset(X_train_np[idx_val], y_train_np[idx_val], apply_transforms=False)

            # WeightedRandomSampler to balance imbalanced classes during training
            counts = np.bincount(y_train_np[idx_tr], minlength=NUM_CLASSES)
            weights = 1.0 / (counts + 1e-6)
            sampler = WeightedRandomSampler(weights[y_train_np[idx_tr]], len(idx_tr))

            loader_tr = DataLoader(dset_train, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
            loader_val = DataLoader(dset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
            loader_test = DataLoader(ds_testing, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

            net = HSI_Hybrid_Network(num_classes=NUM_CLASSES).to(DEVICE)
            opt = optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.05)
            lr_sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=0.001, steps_per_epoch=len(loader_tr), epochs=EPOCHS)

            # Classes 4 and 5 have near-zero weight because they are absent from the training set
            loss_weights = torch.tensor([1.0, 1.0, 1.0, 0.001, 0.001, 1.0, 1.0]).to(DEVICE)
            loss_fn = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=0.1)

            amp_scaler = torch.amp.GradScaler('cuda')

            for epoch in range(EPOCHS):
                net.train()
                epoch_loss = 0.0
                progress = tqdm(loader_tr, leave=False, desc=f"Ep {epoch + 1}", mininterval=3.0)

                for batch_x, batch_y in progress:
                    batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                    opt.zero_grad()
                    with torch.amp.autocast('cuda'):
                        # Apply MixUp with 50% probability per batch
                        if np.random.rand() < 0.5:
                            mixed_x, y_a, y_b, lam = mixup_data(batch_x, batch_y, alpha=1.0)
                            preds = net(mixed_x)
                            loss = lam * loss_fn(preds, y_a) + (1 - lam) * loss_fn(preds, y_b)
                        else:
                            preds = net(batch_x)
                            loss = loss_fn(preds, batch_y)
                    amp_scaler.scale(loss).backward()
                    amp_scaler.step(opt)
                    amp_scaler.update()
                    lr_sched.step()
                    epoch_loss += loss.item()

                net.eval()
                preds_list, truth_list = [], []
                with torch.no_grad():
                    for v_x, v_y in loader_val:
                        v_x = v_x.to(DEVICE)
                        preds_list.extend(net(v_x).argmax(1).cpu().numpy())
                        truth_list.extend(v_y.numpy())

                if (epoch + 1) in [1, 5, 10, EPOCHS]:
                    curr_f1 = f1_score(truth_list, preds_list, average='macro')
                    print(f"   Ep {epoch + 1}: Loss {epoch_loss / len(loader_tr):.4f} | Val F1 {curr_f1:.4f}")

            # Test Time Augmentation: average probabilities over 4 geometric transforms
            net.eval()
            fold_predictions = []
            with torch.no_grad():
                for t_x in loader_test:
                    t_x = t_x.to(DEVICE)
                    p1 = torch.softmax(net(t_x), 1)
                    p2 = torch.softmax(net(torch.flip(t_x, [3])), 1)
                    p3 = torch.softmax(net(torch.flip(t_x, [2])), 1)
                    p4 = torch.softmax(net(torch.rot90(t_x, 1, [2, 3])), 1)
                    p_avg = (p1 + p2 + p3 + p4) / 4.0
                    fold_predictions.append(p_avg.cpu().numpy())
            current_seed_probs += np.concatenate(fold_predictions)

        aggregated_probs += (current_seed_probs / 3.0)
        seeds_completed += 1

        print(f">>> Seed {seed_val} Complete. Saving...")
        temp_final_probs = aggregated_probs / seeds_completed
        class_priors = np.ones(NUM_CLASSES)

        # Iterative calibration: adjust priors so the predicted distribution matches the target
        for _ in range(50):
            temp_preds = np.argmax(temp_final_probs * class_priors, axis=1)
            temp_labels = [idx_to_label_map[k] for k in temp_preds]
            current_counts = pd.Series(temp_labels).value_counts().sort_index()
            for i, class_lbl in enumerate([1, 2, 3, 4, 5, 6, 7]):
                observed = current_counts.get(class_lbl, 1)
                target = target_dist.get(class_lbl, 1)
                class_priors[i] *= (target / observed) ** 0.5

        final_idx = np.argmax(temp_final_probs * class_priors, axis=1)
        final_lbls = [idx_to_label_map[k] for k in final_idx]

        sub = pd.DataFrame({'filename': test_files, 'label': final_lbls})
        sub.to_csv('submission.csv', index=False)
        print(">>> Updated 'submission.csv'")
        print("Classes 1-7:")
        counts = sub['label'].value_counts()
        all_classes = [1, 2, 3, 4, 5, 6, 7]
        print(counts.reindex(all_classes, fill_value=0))

        elapsed = (time.time() - start_time) / 3600
        if elapsed > 8.5:
            break


if __name__ == "__main__":
    execute_training_pipeline()