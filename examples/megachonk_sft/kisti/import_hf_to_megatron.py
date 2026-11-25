"""Import an Apertus 2 HF checkpoint (megachonk, with the embedding norm) into a Megatron torch_dist checkpoint,
as NeMo-RL's community_import.import_model_from_hf_name does: Bridge provider from the HF config, the model spread
over TP1 x PP4 x EP8 (32 GPUs), HF weights loaded, saved with the parallel sizes reset to 1 (the checkpoint
reshards on load). Run with one process per GPU (srun), RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR set.
    python import_hf_to_megatron.py <hf folder> <output folder>
"""
import os
import sys

import torch

from megatron.bridge import AutoBridge


# The async checkpoint writer starts helper processes with "spawn", which re-imports this file: everything runs
# under the main guard, or each helper would re-run the import (job 1325238 died with EOFError at the save).
def main():
    hf, out = sys.argv[1], sys.argv[2]
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()

    bridge = AutoBridge.from_hf_pretrained(hf, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=True)
    assert getattr(provider, "embedding_norm", False), "the provider has no embedding norm: check the Bridge patch"
    orig = {k: getattr(provider, k) for k in (
        "tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size",
        "expert_model_parallel_size", "expert_tensor_parallel_size",
        "num_layers_in_first_pipeline_stage", "num_layers_in_last_pipeline_stage", "pipeline_dtype")}
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 4
    provider.context_parallel_size = 1
    provider.expert_model_parallel_size = 8
    provider.expert_tensor_parallel_size = 1
    provider.num_layers_in_first_pipeline_stage = 16
    provider.num_layers_in_last_pipeline_stage = 15
    provider.pipeline_dtype = torch.bfloat16
    provider.sequence_parallel = False
    provider.finalize()
    provider.initialize_model_parallel(seed=0)
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    if rank == 0:
        names = [n for n, _ in model[0].named_parameters() if "embedding_norm" in n]
        print("embedding norm parameters on rank 0:", names, flush=True)
    config = model[0].config
    for k, v in orig.items():
        setattr(config, k, v)
    bridge.save_megatron_model(model, out)
    torch.distributed.barrier()
    if rank == 0:
        print("saved", out, flush=True)


if __name__ == "__main__":
    main()
