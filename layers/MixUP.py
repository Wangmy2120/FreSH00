import torch
import numpy as np


def random_mixup_with_extra(batch_x, batch_y, alpha=0.4, mix_ratio=0.5):

    device = batch_x.device
    batch_y = batch_y.squeeze().long()
    batch_size = batch_x.size(0)
    num_mixed = int(batch_size * mix_ratio)

    index_a = torch.randperm(batch_size)[:num_mixed].to(device)
    index_b = torch.randperm(batch_size)[:num_mixed].to(device)
    lam = np.random.beta(alpha, alpha, size=(num_mixed,))


    lam_t = torch.from_numpy(lam).float().to(device).view(-1, 1, 1)
    mixed_x = lam_t * batch_x[index_a] + (1 - lam_t) * batch_x[index_b]


    y_a = batch_y[index_a]
    y_b = batch_y[index_b]
    mixed_y = [(float(lam[i]), y_a[i], y_b[i]) for i in range(num_mixed)]


    orig_y = [(1.0, y, y) for y in batch_y]


    all_x = torch.cat([batch_x, mixed_x], dim=0)
    all_y = orig_y + mixed_y

    return all_x, all_y
