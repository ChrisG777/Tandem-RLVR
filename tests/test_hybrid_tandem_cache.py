"""Exercise patched cache delegation without importing CUDA libraries."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


def manager_type():
    patch = (Path(__file__).resolve().parents[1] / 'third_party/vllm-tandem.patch').read_text()
    section = patch.split('+++ b/vllm/v1/worker/tandem.py\n', 1)[1]
    section = section.split('\ndiff --git ', 1)[0]
    source = '\n'.join(line[1:] for line in section.splitlines() if line.startswith('+'))
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    methods = {'initialize_frozen_kv_cache', '_adapt_attn_metadata', '_adapt_slot_mapping'}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    namespace = {'Any': object, 'is_frozen_layer': lambda n: n.startswith('tandem_frozen.'),
                 'torch': SimpleNamespace(device=lambda kind, index: (kind, index),
                                          cuda=SimpleNamespace(current_device=lambda: 0))}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<tandem cache methods>', 'exec'), namespace)
    return namespace['TandemModelManager']


class HybridCache(unittest.TestCase):
    def manager(self):
        m = manager_type()()
        m.frozen_model = object()
        m.frozen_device = ('cuda', 0)
        m.primary_vllm_config = SimpleNamespace(model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(layer_types=['linear_attention', 'full_attention'])))
        m._native_hybrid_cache = False
        return m

    def test_delegates_all_cache_groups_and_preserves_native_metadata(self):
        m = self.manager()
        groups = [SimpleNamespace(layer_names=['primary.attn', 'tandem_frozen.attn']),
                  SimpleNamespace(layer_names=['primary.linear', 'tandem_frozen.linear'])]
        m.initialize_frozen_kv_cache(SimpleNamespace(kv_cache_groups=groups), object())
        self.assertEqual(m.frozen_layer_names, ['tandem_frozen.attn', 'tandem_frozen.linear'])
        metadata = {'tandem_frozen.attn': object(), 'tandem_frozen.linear': object()}
        self.assertIs(m._adapt_attn_metadata(metadata), metadata)
        self.assertIs(m._adapt_slot_mapping(metadata), metadata)

    def test_rejects_missing_registration_and_separate_device(self):
        m = self.manager()
        config = SimpleNamespace(kv_cache_groups=[])
        with self.assertRaisesRegex(ValueError, 'missing'):
            m.initialize_frozen_kv_cache(config, object())
        m.frozen_device = ('cuda', 1)
        with self.assertRaisesRegex(ValueError, 'colocated'):
            m.initialize_frozen_kv_cache(config, object())


if __name__ == '__main__':
    unittest.main()
