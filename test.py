import torchaudio_augmentations

spectrogram = torchaudio_augmentations.transforms.Spectrogram()
masking = torchaudio_augmentations.transforms.FrequencyMasking(freq_mask_param=80)