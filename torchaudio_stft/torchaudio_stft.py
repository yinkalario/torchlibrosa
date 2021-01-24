import torch
import torch.nn as nn


class STFT(nn.Module):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None,
        window='hann', center=True, pad_mode='reflect'):
        super().__init__()

        assert pad_mode in ['constant', 'reflect']

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        self.pad_mode = pad_mode

        # By default, use the entire frame
        if self.win_length is None:
            self.win_length = self.n_fft
        window_dict = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window
        }
        self.register_buffer('window', window_dict[window](self.win_length))

        # Set the default hop, if it's not already specified
        if self.hop_length is None:
            self.hop_length = int(self.win_length // 4)
    
    def forward(self, x):
        """
        input: 
            x (Tensor): (batch_size, channels, data_length)
        output: 
            x_shift_real (Tensor): (batch_size, channels, time_steps, freq_bins)
            x_shift_imag (Tensor): (batch_size, channels, time_steps, freq_bins)
        """
        x_stft = []
        for ch in range(x.shape[1]):
            x_stft.append(torch.stft(
                input=x[:, ch],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=self.center,
                pad_mode=self.pad_mode,
                return_complex=False
            ))
        x_stft = torch.stack(x_stft, dim=1) # (batch_size, channels, freq_bins, time_steps, ...)
        return x_stft[..., 0].permute(0, 1, 3, 2), x_stft[..., 1].permute(0, 1, 3, 2), 


class ISTFT(nn.Module):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None,
        window='hann', center=True, pad_mode='reflect', return_complex=False):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length =hop_length
        self.win_length = win_length
        self.center = center
        self.pad_mode = pad_mode
        self.return_complex = return_complex

        # By default, use the entire frame
        if self.win_length is None:
            self.win_length = self.n_fft
        window_dict = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window
        }
        self.register_buffer('window', window_dict[window](self.win_length))

        # Set the default hop, if it's not already specified
        if self.hop_length is None:
            self.hop_length = int(self.win_length // 4)
    
    def forward(self, x_stft_real, x_stft_imag, length):
        """
        input: 
            x_shift_real (Tensor): (batch_size, channels, time_steps, freq_bins)
            x_shift_imag (Tensor): (batch_size, channels, time_steps, freq_bins)
            length (int): length of the signal
        output: 
            x (Tensor): (batch_size, channels, data_length)
        """
        x_stft = torch.stack([x_stft_real, x_stft_imag], dim=4)
        x_stft = x_stft.permute(0, 1, 3, 2, 4) # (batch_size, channels, freq_bins, time_steps, 2)
        
        x = []
        for ch in range(x_stft.shape[1]):
            x.append(torch.istft(
                input=x_stft[:, ch],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=self.center,
                length=length,
                return_complex=False
            ))
        x = torch.stack(x, dim=1)
        return x

