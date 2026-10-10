"""FP8 functions interface for DeepGEMM calls in MoE layers."""

import torch
try:
    import deep_gemm
except ImportError:
    deep_gemm = None
import matplotlib.pyplot as plt

def diff_tensor_norm(t1, t2):
    t1f = t1.to(torch.float)
    t2f = t2.to(torch.float)
    return torch.norm(t1f - t2f).item() / torch.norm(t2f).item()

def plot_tensor_hist(tensor: torch.Tensor, name, bins=100, title="Tensor Value Distribution"):
    values = tensor.flatten().detach().cpu().float().numpy()
    mean = values.mean()
    var = values.var()
    std = values.std()

    plt.figure(figsize=(8, 4))
    plt.hist(values, bins=bins, edgecolor="black", alpha=0.75)

    # vertical lines for mean ± std
    plt.axvline(mean, color="red", linestyle="--", label=f"Mean: {mean:.4e}")
    plt.axvline(mean + std, color="orange", linestyle=":", label=f"+1 Std: {mean+std:.4e}")
    plt.axvline(mean - std, color="orange", linestyle=":", label=f"−1 Std: {mean-std:.4e}")

    # stats box in the corner
    stats_text = f"Var: {var:.4e}\nStd: {std:.4e}\nMean: {mean:.4e}"
    plt.text(0.97, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment="top", horizontalalignment="right",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend(loc="upper left")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"./{name}_histogram.png")

def m_grouped_fp8_gemm_nt_contiguous(
    tokens_per_expert: torch.Tensor,
    fp8_a: tuple[torch.Tensor, torch.Tensor],
    fp8_b: tuple[torch.Tensor, torch.Tensor],
    compute_stream: torch.cuda.Stream,
    output: torch.Tensor = None,
) -> torch.Tensor:
    """M-grouped contiguous FP8 GEMM for one MoE chunk.

    Computes Y[i] = A[i] @ B[expert_of(i)].T in a single kernel launch,
    replacing the per-expert multi-stream loop in multi_stream_fp8_gemm_nt.

    Args:
        tokens_per_expert: int tensor [G], rows assigned to each local expert.
            Each entry must be a multiple of
            deep_gemm.get_m_alignment_for_contiguous_layout() (128 on SM90).
            Permuted activations must be padded accordingly upstream.
        fp8_a: (a, sfa) where
            a:   [M, K]      fp8_e4m3, K-major contiguous
            sfa: [M, K//128] float32  (recipe (1, 128), per-token block-K scales)
        fp8_b: (b, sfb) where
            b:   [G, N, K]      fp8_e4m3, K-major contiguous (stacked weights)
            sfb: [G, N//128, K//128] float32  (recipe (128, 128) block scales)
        output: optional [M, N] bfloat16 destination. Allocated if None.
        m_indices: optional int32 [M] expert-id-per-row. Built from
            tokens_per_expert if not provided. Pass it in to avoid rebuilding
            across fc1/fc2 of the same chunk.

    Returns:
        output tensor [M, N] bfloat16.
    """
    assert deep_gemm is not None
    a, sfa = fp8_a
    b, sfb = fp8_b
    assert b.dim() == 3, f"b must be stacked [G, N, K], got shape {tuple(b.shape)}"
    assert sfb.dim() == 3, f"sfb must be stacked [G, N//128, K//128], got {tuple(sfb.shape)}"

    g, n, k = b.shape
    m = a.shape[0]
    assert a.shape[1] == k, f"K mismatch: a {a.shape[1]} vs b {k}"
    assert tokens_per_expert.numel() == g, \
        f"tokens_per_expert length {tokens_per_expert.numel()} != num_groups {g}"

    # align = deep_gemm.get_m_alignment_for_contiguous_layout()
    # assert (tokens_per_expert % align == 0).all(), \
    #     f"each tokens_per_expert entry must be a multiple of {align}"

    if output is None:
        output = torch.empty(m, n, dtype=torch.bfloat16, device=a.device)
    else:
        assert output.shape == (m, n) and output.dtype == torch.bfloat16

    with torch.cuda.stream(compute_stream):
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (a, sfa),
            (b, sfb),
            output,
            tokens_per_expert,
            recipe_a=(1, 128),
            recipe_b=(128, 128),
            disable_ue8m0_cast=True,
            use_psum_layout=True,
        )
    return output

def k_grouped_fp8_gemm_nt_contiguous(
    ks: list[int],
    ks_tensor: torch.Tensor,
    fp8_a: tuple[torch.Tensor, torch.Tensor],
    fp8_b: tuple[torch.Tensor, torch.Tensor],
    num_local_experts: int,
    compute_stream: torch.cuda.Stream,
    output: torch.Tensor,
):
    with torch.cuda.stream(compute_stream):
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
            fp8_a,
            fp8_b,
            output,
            ks,
            ks_tensor,
            output,
            recipe=(1, 1, 128),
        )

    return output

def multi_stream_fp8_gemm_nt_1d1d(
    batch_sizes: torch.Tensor,
    fp8_a: tuple[torch.Tensor, torch.Tensor],
    fp8_b: tuple[torch.Tensor, torch.Tensor],
    compute_streams: list[torch.cuda.Stream],
    output: list[torch.Tensor],
    accumulate: bool = False,
):
    assert deep_gemm is not None
    batch_sizes_list = batch_sizes.tolist()
    
    num_gemms = len(batch_sizes_list)
    a, a_scales = fp8_a
    b, b_scales = fp8_b

    sliced_a = torch.split(a, batch_sizes_list, dim=1)
    sliced_b = torch.split(b, batch_sizes_list, dim=1)

    slice_sizes = (batch_sizes//128).tolist()
    sliced_a_scales = torch.split(a_scales, slice_sizes, dim=1)
    sliced_b_scales = torch.split(b_scales, slice_sizes, dim=1)

    for i in range(num_gemms):
        stream = compute_streams[i%len(compute_streams)]
        with torch.cuda.stream(stream):
            deep_gemm.fp8_gemm_nt(
                (sliced_a[i].contiguous(), sliced_a_scales[i].contiguous()),
                (sliced_b[i].contiguous(), sliced_b_scales[i].contiguous()),
                output[i],
                c=None if not accumulate else output[i],
                recipe=None,
                recipe_a=(1, 128),
                recipe_b=(128, 128) if not accumulate else (1, 128),
                disable_ue8m0_cast=True,
            )
    return output
