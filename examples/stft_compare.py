import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from stft import ISTFT, STFT
# from torchaudio_stft import ISTFT, STFT


x = torch.rand((64, 1, 8000), requires_grad=True, dtype=torch.float32).cuda()
n_fft = 320
hop_length = 160

# ##### Test torchaudio.transforms.Spectrogram on multi-gpus
# window_fn = torch.hann_window
# power = None
# spectrogram = torchaudio.transforms.Spectrogram(
#     n_fft=n_fft,
#     hop_length=hop_length,
#     window_fn=window_fn,
#     power=power
# )
# spectrogram = nn.DataParallel(spectrogram)
# spectrogram.cuda()
# out = spectrogram(input_data)

##### Test torch.stft and torch.istft
x = F.pad(x, pad=(0, n_fft//2), mode='constant', value=0)
stft_extractor = STFT(n_fft=n_fft, hop_length=hop_length, window='hann')
stft_extractor = nn.DataParallel(stft_extractor)
stft_extractor.cuda()
x_stft_real, x_stft_imag = stft_extractor(x)

istft_extractor = ISTFT(n_fft=n_fft, hop_length=hop_length, window='hann')
istft_extractor = nn.DataParallel(istft_extractor)
istft_extractor.cuda()
x_reconst = istft_extractor(x_stft_real, x_stft_imag, length=8000)
print(torch.max(torch.abs(x[..., :8000] - x_reconst)))