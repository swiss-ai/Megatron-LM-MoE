"""Exact Apertus SFT epochs with a zero-loss tail for each global batch."""

import numpy as np
import torch


class ApertusSFTEpochDataset(torch.utils.data.Dataset):
    """Permute one exhaustive source epoch, then pad its last global batch."""

    def __init__(self, dataset, epochs, global_batch_size, seed):
        if len(dataset) == 0 or epochs <= 0 or global_batch_size <= 0:
            raise ValueError("exact SFT epochs require nonempty data and positive sizes")
        self.dataset = dataset
        self.real_samples_per_epoch = len(dataset)
        self.samples_per_epoch = (
            (len(dataset) + global_batch_size - 1) // global_batch_size * global_batch_size
        )
        rng = np.random.RandomState(seed)
        self.order = np.stack([rng.permutation(len(dataset)) for _ in range(epochs)])
        self.split = getattr(dataset, "index_split", getattr(dataset, "split", None))

    def __len__(self):
        return len(self.order) * self.samples_per_epoch

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        epoch, position = divmod(index, self.samples_per_epoch)
        if position < self.real_samples_per_epoch:
            return self.dataset[int(self.order[epoch, position])]
        # BlendedDataset needs an integer index; its AP leaves accept None.
        if hasattr(self.dataset, "datasets"):
            return {"dataset_id": 0, **self.dataset.datasets[0][None]}
        return self.dataset[None]


def prepare_apertus_sft_epochs(provider, args):
    """Resolve training length before optimizer setup and retain the source epoch.

    All ranks participate in the distributed provider and length reduction. Later
    calls build evaluation data at its actual requested size and reuse training
    data, including for virtual pipeline stages.
    """
    kwargs = {"vp_stage": 0} if args.virtual_pipeline_model_parallel_size is not None else {}
    source, _, _ = provider((None, 0, 0), **kwargs)
    count = torch.tensor([len(source) if source is not None else 0],
                         dtype=torch.long, device="cuda")
    torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.MAX)
    real_count = count.item()
    if real_count == 0:
        raise ValueError("--ap-sft-epochs requires a nonempty training split")
    if source is not None and len(source) != real_count:
        raise ValueError("Apertus SFT training length differs across ranks")
    samples_per_epoch = (
        (real_count + args.global_batch_size - 1)
        // args.global_batch_size * args.global_batch_size
    )
    args.ap_sft_real_samples_per_epoch = real_count
    args.ap_sft_samples_per_epoch = samples_per_epoch
    args.train_iters = args.ap_sft_epochs * samples_per_epoch // args.global_batch_size
    training = (
        ApertusSFTEpochDataset(source, args.ap_sft_epochs, args.global_batch_size, args.seed)
        if source is not None else None
    )

    def epoch_provider(sizes, vp_stage=None):
        stage_kwargs = {"vp_stage": vp_stage} if kwargs else {}
        _, valid, test = provider((0, sizes[1], sizes[2]), **stage_kwargs)
        return training, valid, test

    epoch_provider.is_distributed = getattr(provider, "is_distributed", False)
    return epoch_provider
