## How to use ?

Add these flags to enable optimizer cpu offload in MCore.

```bash
--optimizer-cpu-offload
--optimizer-offload-fraction 1.0
--use-precision-aware-optimizer
```

## Optional optimizer-state pinning

Add `--pin-cpu-optimizer-state` to copy pageable CPU optimizer state, such as
AdamW moments, into CUDA host storage after the CPU update and when loading state.
The Python API option is `pin_cpu_optimizer_state=True`. This is disabled by default.

CUDA host storage does not migrate based on GPU access on ATS systems such as
GH200. This is a placement safeguard, not a claim that normal CPU AdamW updates
access their moments from the GPU. CUDA host storage does not necessarily mean
physically page-locked memory on ATS systems.

AdamW's allocation logic stays unchanged. Each new or replaced pageable state
tensor needs one copy; subsequent steps reuse pinned tensors. The copy temporarily
requires both old and new storage, and other references can keep old storage alive.
CUDA tensors, empty tensors, and the `master_param` working-parameter alias are
left unchanged. Parameter and gradient pinning options are separate.

## Configuration Recommendations

Gradient copy from GPU to CPU, CPU optimizer step, and subsequent parameter copy from CPU to GPU can be time-consuming operations, and it is recommended to use the flag `--overlap-cpu-optimizer-d2h-h2d` to execute them concurrently.
