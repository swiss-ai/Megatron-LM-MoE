"""Physical-layer execution schedule for SMELT (arXiv:2609.01343)."""


def smelt_layer_order(num_layers, loop_start=-1, loop_layers=0):
    """Return zero-based physical indices; repeat the whole span, not each layer.

    An odd number of outside layers leaves the extra layer in the suffix.
    Registered modules and checkpoint keys remain indexed by physical layer.
    """
    if num_layers < 1 or loop_layers < 0 or loop_layers > num_layers:
        raise ValueError("SMELT requires 0 <= loop_layers <= num_layers and num_layers > 0")
    if loop_start == -1:
        loop_start = (num_layers - loop_layers) // 2
    if loop_start < 0 or loop_start + loop_layers > num_layers:
        raise ValueError("SMELT loop span is outside the physical layer stack")
    end = loop_start + loop_layers
    return tuple(range(loop_start)) + tuple(range(loop_start, end)) * 2 + tuple(range(end, num_layers))
