"""
One piece of audio = One piece of image
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import librosa
import numpy as np
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import accuracy_score
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from google.colab import drive
drive.mount('/content/drive')

BASE_DIR = Path("/content/drive/MyDrive/audio_to_mel_spectrogram/Kaggle_Data")
TRAIN_CSV = BASE_DIR / "metadata" / "kaggle_train.csv"
TEST_CSV = BASE_DIR / "metadata" / "kaggle_test.csv"
AUDIO_DIR = BASE_DIR / "audio"
TEST_AUDIO_DIR = AUDIO_DIR / "test"

class AudioDataset(Dataset):
    def __init__(self, csv_path, audio_dir, transform=None, duration=2.5, sr=22050):

        self.df = pd.read_csv(csv_path)
        self.audio_dir = Path(audio_dir)
        self.transform = transform
        self.duration = duration
        self.sr = sr
        self.n_mels = 128

        self.classes = sorted(self.df['class'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        file_name = row['slice_file_name']
        fold = row['fold']
        audio_path = self.audio_dir / f"fold{fold}" / file_name

        mel_spec = self.audio_to_mel_spectrogram(audio_path)

        mel_spec = torch.FloatTensor(mel_spec).unsqueeze(0)

        if self.transform:
            mel_spec = self.transform(mel_spec)

        class_name = row['class']
        label = self.class_to_idx[class_name]

        return mel_spec, label, file_name

    def audio_to_mel_spectrogram(self, audio_path):
        try:
            y, sr = librosa.load(str(audio_path), sr=self.sr)

            target_length = int(self.sr * self.duration)
            if len(y) > target_length:
                y = y[:target_length]
            elif len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), mode='constant')

            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=self.n_mels,
                n_fft=2048, hop_length=512, fmax=8000
            )

            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

            return mel_spec_norm

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return np.zeros((self.n_mels, 216))

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout(0.4)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, val_loader, num_epochs=20, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_val_acc = 0
    best_model_state = None
    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        if epoch < 3:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001 * (epoch + 1) / 3

        for batch_idx, (data, target, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()

        if epoch == 5:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.00001

        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for data, target, _ in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(target.cpu().numpy())

        val_acc = accuracy_score(val_targets, val_preds)
        avg_loss = running_loss / len(train_loader)

        train_losses.append(avg_loss)
        val_accuracies.append(val_acc)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_loss:.4f}')
        print(f'  Val Accuracy: {val_acc:.4f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')


    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'\nLoaded best model with val_acc: {best_val_acc:.4f}')

    return train_losses, val_accuracies, best_val_acc

def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_targets = []
    all_filenames = []

    with torch.no_grad():
        for data, target, filenames in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_filenames.extend(filenames)

    accuracy = accuracy_score(all_targets, all_preds)
    print(f'Test Accuracy: {accuracy:.4f}')

    return all_preds, all_targets, all_filenames, accuracy

def audio_to_mel_spectrogram_single(audio_path, sr=22050, n_mels=128, duration=2.5):
    try:
        y, original_sr = librosa.load(audio_path, sr=sr)

        target_length = int(sr * duration)
        if len(y) > target_length:
            y = y[:target_length]
        elif len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')

        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=n_mels,
            n_fft=2048,
            hop_length=512,
            fmax=8000
        )

        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

        return mel_spec_norm

    except Exception as e:
        print(f"Error in audio processing for {audio_path}: {e}")
        return np.zeros((n_mels, 216))

def load_test_csv(test_csv_path):
    try:
        df = pd.read_csv(test_csv_path, sep='\t')
        if len(df.columns) > 1:
            return df
    except Exception as e:
        print(f"Tab separator failed: {e}")

    try:
        df = pd.read_csv(test_csv_path, sep=',')
        if len(df.columns) > 1:
            return df
    except Exception as e:
        print(f"Comma separator failed: {e}")

    df = pd.read_csv(test_csv_path)
    return df

def create_submission_csv(model, test_csv_path, audio_dir, test_audio_dir, class_to_idx, output_path='submission.csv'):
    from google.colab import files

    device = torch.device('cpu')
    model = model.to(device)
    model.eval()

    test_df = load_test_csv(test_csv_path)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    column_map = {}
    for col in test_df.columns:
        col_lower = col.lower()
        if 'id' in col_lower:
            column_map['ID'] = col
        elif 'file' in col_lower or 'slice' in col_lower:
            column_map['slice_file_name'] = col

    id_column = column_map.get('ID', test_df.columns[0])
    filename_column = column_map.get('slice_file_name', test_df.columns[1])

    predictions = []
    found_files = 0
    successful_predictions = 0
    failed_predictions = 0

    with torch.no_grad():
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
            file_name = row[filename_column]
            file_id = row[id_column]

            audio_path = None

            if test_audio_dir and test_audio_dir.exists():
                audio_path = test_audio_dir / file_name
                if not audio_path.exists():
                    audio_path = None

            if audio_path is None:
                try:
                    parts = file_name.split('-')
                    if len(parts) >= 2:
                        fold = int(parts[1])
                    else:
                        fold = 1
                except:
                    fold = 1

                audio_path = audio_dir / f"fold{fold}" / file_name

                if not audio_path.exists():
                    for test_fold in range(1, 11):
                        test_path = audio_dir / f"fold{test_fold}" / file_name
                        if test_path.exists():
                            audio_path = test_path
                            break

            if audio_path and audio_path.exists():
                found_files += 1
                try:
                    mel_spec = audio_to_mel_spectrogram_single(str(audio_path))
                    mel_spec_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(device)

                    output = model(mel_spec_tensor)
                    pred_class = output.argmax(dim=1).item()

                    predictions.append({
                        'ID': file_id,
                        'TARGET': pred_class
                    })
                    successful_predictions += 1

                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
                    failed_predictions += 1
                    predictions.append({
                        'ID': file_id,
                        'TARGET': 2
                    })
            else:
                predictions.append({
                    'ID': file_id,
                    'TARGET': 2
                })

    submission_df = pd.DataFrame(predictions)
    submission_df = submission_df[['ID', 'TARGET']]
    submission_df = submission_df.sort_values('ID').reset_index(drop=True)

    submission_df.to_csv(output_path, index=False)

    target_counts = submission_df['TARGET'].value_counts().sort_index()
    print("\n")
    for target, count in target_counts.items():
        class_name = idx_to_class.get(target, f"Class_{target}")
        print(f"   {target} ({class_name}): {count} samples")

    files.download(output_path)
    print(f"\nSubmission file downloaded")

    return submission_df

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    full_dataset = AudioDataset(TRAIN_CSV, AUDIO_DIR)

    train_size = int(0.80 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    num_classes = len(full_dataset.classes)
    model = CNN(num_classes=num_classes).to(device)

    print("\nTraining Model")


    train_losses, val_accuracies, best_val_acc = train_model(
        model, train_loader, val_loader, num_epochs=20, device=device
    )

    val_preds, val_targets, val_filenames, val_accuracy = evaluate_model(model, val_loader, device=device)
    print(f"Final Validation Accuracy: {val_accuracy:.4f}")

    if TEST_CSV.exists():
        if TEST_AUDIO_DIR and TEST_AUDIO_DIR.exists():
            submission_df = create_submission_csv(
                model, TEST_CSV, AUDIO_DIR, TEST_AUDIO_DIR, full_dataset.class_to_idx
            )
    model_weights_path = BASE_DIR / "cnn_weights.pth"
    torch.save(model.state_dict(), model_weights_path)
    print(f"\nModel weights saved to: {model_weights_path}")

