# audio_to_mel_spectrogram

**Classifying urban sounds by turning audio into images.**

Each audio clip is converted into a mel spectrogram (a picture of how the sound's frequencies change over time), and a convolutional neural network classifies that picture into one of 10 sound categories. One clip in, one image out, one label predicted.

**Final Kaggle score: 82.735% accuracy** on the held-out test set (1,653 clips).

---

## The Task

Given a short recording of a city sound, predict which of these 10 classes it belongs to:

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | air_conditioner | 5 | engine_idling |
| 1 | car_horn | 6 | gun_shot |
| 2 | children_playing | 7 | jackhammer |
| 3 | dog_bark | 8 | siren |
| 4 | drilling | 9 | street_music |

The data follows the UrbanSound8K layout: clips are organized into `fold1/` … `fold10/` directories and described by a metadata CSV with `slice_file_name`, `fold`, and `class` columns.

---

## How It Works

```
 .wav clip ──► load at 22,050 Hz ──► trim / zero-pad to 2.5 s
                                              │
                                              ▼
                  128-band mel spectrogram (n_fft=2048, hop=512, fmax=8 kHz)
                                              │
                                              ▼
                     convert to dB ──► min-max normalize to [0, 1]
                                              │
                                              ▼
                       1 × 128 × 108 "image" ──► CNN ──► 10 class scores
```

### 1. Preprocessing

Every clip is resampled to 22,050 Hz and forced to exactly 2.5 seconds (longer clips are truncated, shorter ones are zero-padded), so every spectrogram has the same shape. The mel spectrogram uses 128 mel bands capped at 8 kHz, which covers the range where most urban sounds carry their identifying information. Power values are converted to decibels and then scaled to [0, 1] per clip.

### 2. Model

A VGG-style CNN with three convolutional blocks followed by a fully connected classifier:

| Stage | Layers | Output channels |
|-------|--------|-----------------|
| Block 1 | 2 × (Conv 3×3 → BatchNorm → ReLU), MaxPool 2×2, Dropout 0.2 | 32 |
| Block 2 | 2 × (Conv 3×3 → BatchNorm → ReLU), MaxPool 2×2, Dropout 0.3 | 64 |
| Block 3 | 2 × (Conv 3×3 → BatchNorm → ReLU), AdaptiveAvgPool → 4×4, Dropout 0.4 | 128 |
| Classifier | Linear 2048 → 512 → 256 → 10, with Dropout 0.5 / 0.3 | — |

Convolutional weights use Kaiming initialization. The adaptive pooling layer means the network accepts spectrograms of any width, not just the 108-frame default.

### 3. Training

| Setting | Value |
|---------|-------|
| Split | 80% train / 20% validation (random) |
| Batch size | 32 |
| Optimizer | Adam, weight decay 1e-4 |
| Loss | Cross-entropy |
| Epochs | 20 |
| Learning rate | Linear warm-up to 1e-3 over epochs 1–3, held through epoch 5, then dropped to 1e-5 |
| Gradient clipping | Max norm 1.0 |

---

## Results

### Training progress

| Epoch | Train Loss | Val Accuracy | Learning Rate |
|------:|-----------:|-------------:|--------------:|
| 1 | 1.9317 | 36.51% | 0.000333 |
| 2 | 1.4160 | 48.02% | 0.000667 |
| 3 | 1.1192 | 50.07% | 0.001000 |
| 4 | 0.9384 | 62.64% | 0.001000 |
| 5 | 0.8554 | 69.99% | 0.001000 |
| 6 | 0.7506 | 71.47% | 0.000010 |
| 7 | 0.6469 | 80.72% | 0.000010 |
| 8 | 0.5865 | 82.42% | 0.000010 |
| 10 | 0.5379 | 82.70% | 0.000010 |
| 12 | 0.5099 | 83.90% | 0.000010 |
| 15 | 0.4905 | 84.53% | 0.000010 |
| 18 | 0.4758 | 84.68% | 0.000010 |
| 20 | 0.4688 | **84.96%** | 0.000010 |

The biggest single jump (71% → 81%) came right after the learning rate dropped at the end of epoch 5. After about epoch 11 the model plateaus in the 83–85% range.

### Final scores

| Metric | Accuracy |
|--------|---------:|
| Validation (20% of training data) | 84.96% |
| **Kaggle test set** | **82.735%** |

The roughly 2-point gap between validation and test is expected: the validation set is a random split of the training data, so it shares recording sessions with the training clips, while the Kaggle test set is fully unseen.

### Test-set prediction distribution

| Class | Predicted count |
|-------|----------------:|
| 0 · air_conditioner | 229 |
| 1 · car_horn | 67 |
| 2 · children_playing | 209 |
| 3 · dog_bark | 171 |
| 4 · drilling | 190 |
| 5 · engine_idling | 98 |
| 6 · gun_shot | 93 |
| 7 · jackhammer | 237 |
| 8 · siren | 137 |
| 9 · street_music | 222 |

---

## Project Structure

```
audio_to_mel_spectrogram/
└── Kaggle_Data/
    ├── metadata/
    │   ├── kaggle_train.csv      # slice_file_name, fold, class, ...
    │   └── kaggle_test.csv       # ID, slice_file_name
    ├── audio/
    │   ├── fold1/ … fold10/      # training clips
    │   └── test/                 # test clips
    └── cnn_weights.pth           # saved model weights (created after training)
```

---

## Running It

The script is written for **Google Colab** with the dataset stored in Google Drive.

1. Upload the `Kaggle_Data` folder to `MyDrive/audio_to_mel_spectrogram/` in Google Drive.
2. Open the script in Colab. Switching the runtime to a GPU (Runtime → Change runtime type → GPU) is strongly recommended; see the note on training time below.
3. Run the script. It will mount Drive, train the model, evaluate on the validation split, generate `submission.csv` for the test set, download it, and save the weights to `cnn_weights.pth`.

**Dependencies:** `torch`, `librosa`, `numpy`, `pandas`, `scikit-learn`, `tqdm` (all preinstalled on Colab).

**Submission format:** a CSV with columns `ID` and `TARGET`, where `TARGET` is the class index (0–9).

### A note on training time

The reported run used a **CPU only**. Each epoch took about 12 minutes (the first took 39 minutes while Drive files were cached), so the full 20 epochs took roughly 4.5 hours, plus another 26 minutes to predict the test set. Most of that time is spent recomputing spectrograms from audio on every epoch. A GPU runtime, or precomputing the spectrograms once and caching them as `.npy` files, would cut this down dramatically.

---

## Known Limitations & Ideas for Improvement

A few things worth knowing about the current code, and where the next accuracy gains are likely to come from:

- **"Test Accuracy" in the logs is really validation accuracy.** `evaluate_model` is run on `val_loader`, which is why it prints the same 84.96% as epoch 20. The true test score is the Kaggle result.
- **Best-model checkpointing isn't wired up.** `best_model_state` is never assigned during training, so the final-epoch weights are used. This happened to be the best epoch here, but saving a checkpoint whenever validation accuracy improves would make this reliable.
- **The learning-rate drop is very steep.** Going straight from 1e-3 to 1e-5 works, but a cosine or step schedule (e.g. 1e-3 → 1e-4 → 1e-5) might reach a higher plateau.
- **Random split vs. folds.** UrbanSound8K is designed for fold-based cross-validation because clips from the same recording can land in different folds. Holding out entire folds for validation would give a more honest estimate of test performance.
- **No data augmentation.** SpecAugment (masking random time and frequency bands), time shifting, or mixing in background noise are standard ways to push urban-sound classifiers higher.
- **Fallbacks are silent.** If an audio file fails to load, a blank spectrogram is used during training, and a missing test file is predicted as class 2 (`children_playing`). Logging how often these happen would help catch data issues.
- **Duplicated preprocessing.** The mel spectrogram logic exists both in `AudioDataset` and in `audio_to_mel_spectrogram_single`; keeping a single shared function avoids the two drifting apart.

---

## Key Takeaway

Treating sound as an image works well: a fairly standard image CNN, with no pretrained weights and no augmentation, reaches about **83% accuracy on unseen urban sounds** across 10 classes.
