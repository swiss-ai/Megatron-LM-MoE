"""Physical-layer execution schedule for SMELT (arXiv:2609.01343)."""


def smelt_layer_visits(config, layer_number, is_mtp_layer=False):
    """Visits per microbatch for a one-based physical decoder layer."""
    count = getattr(config, 'smelt_loop_layers', 0)
    if not count or is_mtp_layer:
        return 1
    if layer_number is None:
        raise ValueError("SMELT router metrics require a physical layer number")
    start = getattr(config, 'smelt_loop_start', -1)
    if start == -1:
        start = (config.num_layers - count) // 2
    return 2 if start <= layer_number - 1 < start + count else 1


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
