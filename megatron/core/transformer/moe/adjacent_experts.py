"""Pair physically adjacent MoE layers without sharing their routers or attention."""


def adjacent_expert_pairs(num_layers, moe_layer_freq):
    """Return zero-based pairs; leave an odd tail of each MoE run unpaired."""
    if isinstance(moe_layer_freq, int):
        if moe_layer_freq < 1:
            raise ValueError('moe_layer_freq must be positive')
        pattern = [int(i % moe_layer_freq == 0) for i in range(num_layers)]
    else:
        pattern = list(moe_layer_freq)
    if len(pattern) != num_layers or any(x not in (0, 1) for x in pattern):
        raise ValueError('Expected one binary MoE entry per physical layer')
    pairs = []
    i = 0
    while i + 1 < num_layers:
        if pattern[i] and pattern[i + 1]:
            pairs.append((i, i + 1))
            i += 2
        else:
            i += 1
    return pairs


def tie_adjacent_experts(layers, pairs):
    """Alias only routed expert modules before optimizer/DDP construction.

    Each MoE retains its own dispatcher, router/bias, latent projections, and
    always-on shared expert. EP ranks therefore share the same local expert IDs.
    """
    for first, second in pairs:
        owner, follower = layers[first].mlp, layers[second].mlp
        if owner.num_local_experts != follower.num_local_experts:
            raise ValueError('Adjacent expert pools must have equal local expert counts')
        follower.experts = owner.experts
