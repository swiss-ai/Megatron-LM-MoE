# Dynamic inference functional tests

## Prefix-cache coordinator routing

| Feature | CLI Flag | Default | Purpose |
|---|---|---|---|
| Coordinator routing | `--inference-dynamic-batching-prefix-caching-coordinator-policy {longest_prefix, first_prefix_block, load_balanced}` | `load_balanced` | Multi-rank request routing |

`load_balanced` selects the rank with the fewest in-flight requests, ignoring
prefix affinity. Ties use the lowest rank index. `first_prefix_block` and
`longest_prefix` retain prefix-affinity/load scoring; when prefix caching is
disabled or a request has no hashes, routing falls back to the least-loaded
rank. The retired `round_robin` policy is not an alias for load balancing.

The H100 coordinator recipe
`tests/test_utils/recipes/h100/gpt-dynamic-inference-with-coordinator.yaml`
includes `gpt_dynamic_inference_tp1_pp1_dp8_583m_prefix_caching_load_balanced_zmq`.
Its model configuration explicitly selects `load_balanced`; the upstream golden
values are unchanged by the routing-policy rename.

This functional case requires eight inference ranks and the checkpoint/tokenizer
assets referenced by its model configuration. Presence of the configuration and
golden fixture does not establish a successful run on another GPU platform.
