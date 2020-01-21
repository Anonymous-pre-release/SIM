import numpy as np

from torch.utils.data import Sampler
from collections import defaultdict


class CrossModalityRandomSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        self.rgb_list = []
        self.ir_list = []
        for i, cam in enumerate(dataset.cam_ids):
            if cam in [3, 6]:
                self.ir_list.append(i)
            else:
                self.rgb_list.append(i)

    def __len__(self):
        return max(len(self.rgb_list), len(self.ir_list)) * 2

    def __iter__(self):
        sample_list = []
        rgb_list = np.random.permutation(self.rgb_list).tolist()
        ir_list = np.random.permutation(self.ir_list).tolist()

        rgb_size = len(self.rgb_list)
        ir_size = len(self.ir_list)
        if rgb_size >= ir_size:
            diff = rgb_size - ir_size
            reps = diff // ir_size
            pad_size = diff % ir_size
            for _ in range(reps):
                ir_list.extend(np.random.permutation(self.ir_list).tolist())
            ir_list.extend(np.random.choice(self.ir_list, pad_size, replace=False).tolist())
        else:
            diff = ir_size - rgb_size
            reps = diff // ir_size
            pad_size = diff % ir_size
            for _ in range(reps):
                rgb_list.extend(np.random.permutation(self.rgb_list).tolist())
            rgb_list.extend(np.random.choice(self.rgb_list, pad_size, replace=False).tolist())

        assert len(rgb_list) == len(ir_list)

        half_bs = self.batch_size // 2
        for start in range(0, len(rgb_list), half_bs):
            sample_list.extend(rgb_list[start:start + half_bs])
            sample_list.extend(ir_list[start:start + half_bs])

        return iter(sample_list)


class CrossModalityIdentitySampler(Sampler):
    def __init__(self, dataset, p_size, k_size, ir_cams=[3, 6]):
        self.dataset = dataset
        self.p_size = p_size
        self.k_size = k_size // 2
        self.ir_cams = ir_cams
        self.batch_size = p_size * k_size * 2

        self.id2idx_rgb = defaultdict(list)
        self.id2idx_ir = defaultdict(list)
        for i, identity in enumerate(dataset.ids):
            if dataset.cam_ids[i] in self.ir_cams:
                self.id2idx_ir[identity].append(i)
            else:
                self.id2idx_rgb[identity].append(i)

    def __len__(self):
        return self.dataset.num_ids * self.k_size * 2

    def __iter__(self):
        sample_list = []

        id_perm = np.random.permutation(self.dataset.num_ids)
        for start in range(0, self.dataset.num_ids, self.p_size):
            selected_ids = id_perm[start:start + self.p_size]

            sample = []
            for identity in selected_ids:
                replace = len(self.id2idx_rgb[identity]) < self.k_size
                s = np.random.choice(self.id2idx_rgb[identity], size=self.k_size, replace=replace)
                sample.extend(s)

            sample_list.extend(sample)

            sample.clear()
            for identity in selected_ids:
                replace = len(self.id2idx_ir[identity]) < self.k_size
                s = np.random.choice(self.id2idx_ir[identity], size=self.k_size, replace=replace)
                sample.extend(s)

            sample_list.extend(sample)

        return iter(sample_list)
