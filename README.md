# audio_to_mel_spectrogram
Given numerous video clips, identify which it belongs to. Basic ML project

Attached are results:

Mounted at /content/drive
Using device: cpu

Training Model
Epoch 1/20: 100%|██████████| 176/176 [39:10<00:00, 13.35s/it]
Epoch 1/20:
  Train Loss: 1.9317
  Val Accuracy: 0.3651
  Learning Rate: 0.000333
Epoch 2/20: 100%|██████████| 176/176 [11:58<00:00,  4.09s/it]
Epoch 2/20:
  Train Loss: 1.4160
  Val Accuracy: 0.4802
  Learning Rate: 0.000667
Epoch 3/20: 100%|██████████| 176/176 [12:08<00:00,  4.14s/it]
Epoch 3/20:
  Train Loss: 1.1192
  Val Accuracy: 0.5007
  Learning Rate: 0.001000
Epoch 4/20: 100%|██████████| 176/176 [12:13<00:00,  4.17s/it]
Epoch 4/20:
  Train Loss: 0.9384
  Val Accuracy: 0.6264
  Learning Rate: 0.001000
Epoch 5/20: 100%|██████████| 176/176 [12:11<00:00,  4.16s/it]
Epoch 5/20:
  Train Loss: 0.8554
  Val Accuracy: 0.6999
  Learning Rate: 0.001000
Epoch 6/20: 100%|██████████| 176/176 [11:54<00:00,  4.06s/it]
Epoch 6/20:
  Train Loss: 0.7506
  Val Accuracy: 0.7147
  Learning Rate: 0.000010
Epoch 7/20: 100%|██████████| 176/176 [12:20<00:00,  4.21s/it]
Epoch 7/20:
  Train Loss: 0.6469
  Val Accuracy: 0.8072
  Learning Rate: 0.000010
Epoch 8/20: 100%|██████████| 176/176 [12:06<00:00,  4.13s/it]
Epoch 8/20:
  Train Loss: 0.5865
  Val Accuracy: 0.8242
  Learning Rate: 0.000010
Epoch 9/20: 100%|██████████| 176/176 [12:09<00:00,  4.14s/it]
Epoch 9/20:
  Train Loss: 0.5551
  Val Accuracy: 0.8234
  Learning Rate: 0.000010
Epoch 10/20: 100%|██████████| 176/176 [12:06<00:00,  4.13s/it]
Epoch 10/20:
  Train Loss: 0.5379
  Val Accuracy: 0.8270
  Learning Rate: 0.000010
Epoch 11/20: 100%|██████████| 176/176 [11:54<00:00,  4.06s/it]
Epoch 11/20:
  Train Loss: 0.5068
  Val Accuracy: 0.8355
  Learning Rate: 0.000010
Epoch 12/20: 100%|██████████| 176/176 [12:08<00:00,  4.14s/it]
Epoch 12/20:
  Train Loss: 0.5099
  Val Accuracy: 0.8390
  Learning Rate: 0.000010
Epoch 13/20: 100%|██████████| 176/176 [12:14<00:00,  4.17s/it]
Epoch 13/20:
  Train Loss: 0.5022
  Val Accuracy: 0.8362
  Learning Rate: 0.000010
Epoch 14/20: 100%|██████████| 176/176 [12:16<00:00,  4.18s/it]
Epoch 14/20:
  Train Loss: 0.5085
  Val Accuracy: 0.8347
  Learning Rate: 0.000010
Epoch 15/20: 100%|██████████| 176/176 [12:06<00:00,  4.13s/it]
Epoch 15/20:
  Train Loss: 0.4905
  Val Accuracy: 0.8453
  Learning Rate: 0.000010
Epoch 16/20: 100%|██████████| 176/176 [12:02<00:00,  4.10s/it]
Epoch 16/20:
  Train Loss: 0.4818
  Val Accuracy: 0.8369
  Learning Rate: 0.000010
Epoch 17/20: 100%|██████████| 176/176 [12:09<00:00,  4.14s/it]
Epoch 17/20:
  Train Loss: 0.4666
  Val Accuracy: 0.8411
  Learning Rate: 0.000010
Epoch 18/20: 100%|██████████| 176/176 [12:19<00:00,  4.20s/it]
Epoch 18/20:
  Train Loss: 0.4758
  Val Accuracy: 0.8468
  Learning Rate: 0.000010
Epoch 19/20: 100%|██████████| 176/176 [12:10<00:00,  4.15s/it]
Epoch 19/20:
  Train Loss: 0.4702
  Val Accuracy: 0.8468
  Learning Rate: 0.000010
Epoch 20/20: 100%|██████████| 176/176 [12:08<00:00,  4.14s/it]
Epoch 20/20:
  Train Loss: 0.4688
  Val Accuracy: 0.8496
  Learning Rate: 0.000010
Test Accuracy: 0.8496
Final Validation Accuracy: 0.8496
100%|██████████| 1653/1653 [25:43<00:00,  1.07it/s]


   0 (air_conditioner): 229 samples
   1 (car_horn): 67 samples
   2 (children_playing): 209 samples
   3 (dog_bark): 171 samples
   4 (drilling): 190 samples
   5 (engine_idling): 98 samples
   6 (gun_shot): 93 samples
   7 (jackhammer): 237 samples
   8 (siren): 137 samples
   9 (street_music): 222 samples


   Model weights saved to: /content/drive/MyDrive/audio_to_mel_spectrogram/Kaggle_Data/cnn_weights.pth

   Final Classification Accuracy on Kaggle Dataset: 82.735%
