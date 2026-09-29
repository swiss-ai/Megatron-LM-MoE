"""CPU-only policy tests; extract the pool to avoid Megatron/CUDA imports.

Run directly with Python. CUDA copy correctness still requires a GPU integration run.
"""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


source = Path(__file__).resolve().parents[4] / 'megatron/core/transformer/moe/moe_offload.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
pool_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MoEOffloadMemoryPool')
module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), pool_node], type_ignores=[])
scope = {'os': os}
exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), scope)
Pool = scope['MoEOffloadMemoryPool']


class Buffer:
    device = SimpleNamespace(type='cpu')

    def __init__(self, size):
        self.size = size

    def numel(self):
        return self.size


class Event:
    def __init__(self, ready=False):
        self.ready = ready

    def query(self):
        return self.ready


class CachePolicyTest(unittest.TestCase):
    def pool(self, limit, *sizes):
        pool = Pool(limit)
        buffers = [Buffer(s) for s in sizes]
        pool._all_cpu = buffers.copy()
        pool._total_bytes_cpu = sum(sizes)
        return pool, buffers

    def test_evicts_free_but_preserves_live_buffers(self):
        pool, (old, recent, live) = self.pool(10, 8, 8, 20)
        pool.free(old)
        pool.free(recent)
        self.assertEqual(pool._all_cpu, [recent, live])
        self.assertEqual(pool.stats()['cpu'], (2, 28, 8))

    def test_pending_copy_is_retained_until_completion(self):
        pool, (buf,) = self.pool(0, 20)
        event = Event()
        pool.free(buf, event)
        self.assertEqual(pool._all_cpu, [buf])
        event.ready = True
        pool.trim_cpu_cache()
        self.assertEqual(pool.stats()['cpu'], (0, 0, 0))

    def test_pending_buffer_does_not_prevent_other_eviction(self):
        pool, (pending, ready) = self.pool(8, 8, 8)
        pool.free(pending, Event())
        pool.free(ready, Event(True))
        self.assertEqual(pool._all_cpu, [pending])

    def test_unlimited_and_environment(self):
        pool, (buf,) = self.pool(-1, 100)
        pool.free(buf)
        self.assertEqual(pool.stats()['cpu'], (1, 100, 100))
        for setting, expected in [('0', 0), ('16', 16 * 1024**3), ('-1', -1)]:
            with patch.dict(os.environ, MOE_ACT_CPU_CACHE_LIMIT_GIB=setting):
                self.assertEqual(Pool().cpu_cache_limit_bytes, expected)
        with patch.dict(os.environ, MOE_ACT_CPU_CACHE_LIMIT_GIB='-2'):
            with self.assertRaises(ValueError):
                Pool()


if __name__ == '__main__':
    unittest.main()
