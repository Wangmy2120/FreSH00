import torch
import torch.nn as nn

from layers.global_spilt_MoE import FrequencyExpertSystemSplitMoEWithGlobalExpert


class Model(nn.Module):
    """FreSH: frequency-segmented hierarchical multi-expert model.

    The forward pass follows the paper:
      1) apply per-variable FFT to the raw multivariate input X in R^{d x l};
      2) process the complex spectrum F in C^{d x s} through HiFiMoE;
      3) apply iFFT to obtain the reconstructed time-domain signal X*;
      4) classify X* with a single fully-connected layer.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name

        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.num_classes = configs.num_class

        self.Seg_num = configs.Seg_num
        self.SegE_num = configs.SegE_num
        self.GE_num = configs.GE_num

        self.hidden = getattr(configs, 'expert_hidden_dim', 64)
        self.dropout = nn.Dropout(getattr(configs, 'dropout', 0.1))

        self.freq_expert = FrequencyExpertSystemSplitMoEWithGlobalExpert(
            segment_num=self.Seg_num,
            experts_per_segment=self.SegE_num,
            hidden=self.hidden,
            global_experts_num=self.GE_num,
            channels=self.enc_in,
        )

        # Prediction layer described in Eq. (12): a single fully-connected
        # classifier maps the flattened reconstructed signal to class logits.
        self.fc = nn.Linear(self.enc_in * self.seq_len, self.num_classes)

    def classification(self, x_enc, x_mark_enc=None):
        # x_enc: [B, L, C]
        B, L, C = x_enc.shape

        # Per-variable FFT along the time axis. The input is transposed to
        # [B, C, L] so that the last dimension is time, matching the paper
        # notation F = FFT(X) in C^{d x s}, where s = floor(L / 2) + 1.
        xf = torch.fft.rfft(x_enc.permute(0, 2, 1), n=L, dim=-1, norm="ortho")

        # Frequency-segmented hierarchical MoE -> H* in C^{d x s}.
        h_star = self.freq_expert(xf)

        # Inverse FFT back to the time domain: X* in R^{d x L}.
        x_rec = torch.fft.irfft(h_star, n=L, dim=-1, norm="ortho")

        # Flatten and classify. Raw logits are returned; softmax is applied
        # only during evaluation (as is standard for cross-entropy training).
        x_rec = self.dropout(x_rec.reshape(B, -1))
        return self.fc(x_rec)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None
