"""Small CPU fixtures for the exact contracts used by V3; no model/GPU imports."""
import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np

from lora_state_v3 import (EXPECTED_TRAINABLE, RUNTIME_SHA256, checkpoint_metadata_flat,
                           lora_factor, parameter_report, validate_dense_contract)
from v3_config import HEADS, parameter_group, paths, phase_steps, trainable_path

HERE = Path(__file__).resolve().parent


@dataclass
class Meta:
    shape: tuple
    dtype: object = np.dtype('float32')


class V3Contract(unittest.TestCase):
    def test_dense_tree_rejects_dropped_extra_shape_dtype(self):
        path = ('PaliGemma', 'llm', 'layers', 'attn', 'q_einsum', 'w')
        a = path[:-1] + ('lora_a',)
        expected = {path: Meta((3, 4)), a: Meta((3, 2))}
        validate_dense_contract({path: np.ones((3, 4), np.float32)}, expected)
        for bad in ({}, {path: np.ones((3, 4), np.float32), ('extra',): np.ones(1, np.float32)},
                    {path: np.ones((4, 3), np.float32)}, {path: np.ones((3, 4), np.float16)}):
            with self.assertRaises(ValueError):
                validate_dense_contract(bad, expected)

    def test_ffn_and_attention_b_factor_coverage(self):
        for leaf in ('lora_b', 'gating_einsum_lora_b', 'linear_lora_b'):
            self.assertEqual(lora_factor(('PaliGemma', 'llm', leaf)), 'b')
        for leaf in ('lora_a', 'gating_einsum_lora_a', 'linear_lora_a'):
            self.assertEqual(lora_factor(('PaliGemma', 'llm', leaf)), 'a')
        with self.assertRaises(ValueError):
            lora_factor(('PaliGemma', 'llm', 'unknown_lora_factor'))

    def test_zero_b_preserves_dense_and_allows_first_gradient(self):
        rng = np.random.default_rng(42)
        x, w, a = rng.normal(size=(2, 3)), rng.normal(size=(3, 4)), rng.normal(size=(3, 2))
        b = np.zeros((2, 4))
        np.testing.assert_array_equal(x @ w + x @ a @ b, x @ w)
        grad_b = (x @ a).T @ np.ones((2, 4))
        self.assertTrue(np.any(grad_b != 0))
        self.assertTrue(np.any(a != 0))

    def test_whitelist_freezes_vision_and_dense_without_substring_leaks(self):
        self.assertFalse(trainable_path(('PaliGemma', 'img', 'lora_b')))
        self.assertFalse(trainable_path(('PaliGemma', 'llm', 'layers', 'attn', 'q_einsum', 'w')))
        self.assertFalse(trainable_path(('unknown_action_in_proj', 'kernel')))
        self.assertTrue(trainable_path(('PaliGemma', 'llm', 'layers', 'mlp_1', 'linear_lora_b')))
        self.assertEqual(parameter_group(('PaliGemma', 'llm', 'layers', 'mlp_1', 'linear_lora_b')), 'action_expert_lora')

    def test_report_matches_independent_expected_counts_and_rejects_base_training(self):
        flat = {('PaliGemma', 'img', 'kernel'): Meta((10,)), ('PaliGemma', 'llm', 'embed'): Meta((20,)),
                ('PaliGemma', 'llm', 'attn', 'lora_a'): Meta((27869184,)),
                ('PaliGemma', 'llm', 'attn_1', 'lora_a'): Meta((22118400,))}
        sizes = (33792, 32800, 1049600, 1049600)
        for name, size in zip(HEADS, sizes):
            flat[(name, 'kernel')] = Meta((size,))
        keys = {path for path in flat if trainable_path(path)}
        report = parameter_report(flat, keys)
        self.assertEqual(report['trainable_parameters'], EXPECTED_TRAINABLE['total'])
        self.assertEqual(report['groups']['vit']['trainable'], 0)
        with self.assertRaises(ValueError):
            parameter_report(flat, keys | {('PaliGemma', 'img', 'kernel')})

    def test_orbax_metadata_normalization_keeps_every_dense_leaf(self):
        meta = {'params': {'PaliGemma': {'llm': {'w': {'value': Meta((2, 3))}}},
                           'action_out_proj': {'bias': {'value': Meta((32,))}}}}
        flat = checkpoint_metadata_flat(meta)
        self.assertEqual(set(flat), {('PaliGemma', 'llm', 'w'), ('action_out_proj', 'bias')})
        self.assertEqual(flat[('PaliGemma', 'llm', 'w')].shape, (2, 3))

    def test_actual_parent_metadata_all_paths_are_fp32_target_candidates(self):
        file = HERE.parents[2] / 'reference/parent_best1000_METADATA.json'
        if not file.exists():
            self.skipTest('Reference metadata is not bundled in deployed scripts')
        metadata = json.loads(file.read_text())['tree_metadata']
        names = [tuple(part['key'] for part in value['key_metadata'])[1:-1] for value in metadata.values()]
        self.assertEqual(len(names), len(set(names)))
        self.assertFalse(any('lora' in '/'.join(path) for path in names))
        self.assertTrue(any(parameter_group(path) == 'vit' for path in names))
        self.assertTrue(any(parameter_group(path) == 'gemma_base' for path in names))

    def test_no_call_to_official_initializer_and_no_frozen_cast(self):
        entry = ast.parse((HERE / 'train_lora_v3.py').read_text())
        calls = [node for node in ast.walk(entry) if isinstance(node, ast.Call)]
        self.assertFalse(any(isinstance(node.func, ast.Attribute) and node.func.attr == 'init_train_state' for node in calls))
        module = ast.parse((HERE / 'lora_state_v3.py').read_text())
        initializer = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'init_lora_train_state')
        self.assertNotIn('bfloat16', ast.unparse(initializer))
        self.assertIn('zeros_like', ast.unparse(initializer))

    def test_no_execute_default_is_cpu_safe_and_correct_global_phase(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu')
        result = subprocess.run([sys.executable, str(HERE / 'train_lora_v3.py')], env=env, text=True, capture_output=True, check=True)
        payload = result.stdout[:result.stdout.rfind('\nCONFIGURATION ONLY.')]
        plan = json.loads(payload)
        self.assertEqual(plan['phase_steps'], 19000)
        self.assertEqual(plan['source_step'], 1000)
        self.assertEqual(plan['peak_lr'], 1e-5)
        self.assertIn('/V3/', plan['run_dir'])
        self.assertIn('/V2/', plan['parent_run_root'])
        self.assertEqual(plan['variant'], 'lora_frozen_vit')
        self.assertEqual(plan['parameter_dtype'], 'float32')

    def test_pinned_actual_runtime_sources(self):
        reference = HERE.parents[2] / 'reference'
        if not reference.exists():
            self.skipTest('Reference source is not bundled in deployed scripts')
        for name, expected in RUNTIME_SHA256.items():
            self.assertEqual(hashlib.sha256((reference / name).read_bytes()).hexdigest(), expected, name)

    def test_cache_reuse_preserves_read_only_v2_source(self):
        spec = importlib.util.spec_from_file_location('entry_cache_test', HERE / 'train_lora_v3.py')
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, new = root / 'V2_cache', root / 'V3_cache'
            (old / 'val_cache').mkdir(parents=True)
            payload = old / 'val_cache/batch_000000.npz'
            payload.write_bytes(b'immutable test cache')
            old_transform = {'v2_config.py': '468f1383c8518ed67677b810650e0af2a28710d8493002a7ab4485816d3d33c8',
                             'dataset_v2.py': '2adc6c25c616d2271e08ec0821ef3e8e390a5a8b9971fbf0dfc7477065d07984', 'tokenizer.py': 'unchanged'}
            new_transform = {'v3_config.py': entry.sha256(HERE / 'v3_config.py'),
                             'dataset_v3.py': old_transform['dataset_v2.py'], 'tokenizer.py': 'unchanged'}
            manifest = {'complete': True, 'cache_contract': {'seed': 42, 'transform_sha256': old_transform},
                        'splits': {'val': {'files': {'val_cache/batch_000000.npz': entry.sha256(payload)}}}}
            entry.atomic_json(old / 'manifest.json', manifest)
            before = {p.relative_to(old): p.read_bytes() for p in old.rglob('*') if p.is_file()}
            entry.reuse_eval_cache(SimpleNamespace(reuse_eval_cache=old),
                                   {'cache_dir': str(new), 'cache_contract': {'seed': 42, 'transform_sha256': new_transform}})
            after = {p.relative_to(old): p.read_bytes() for p in old.rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            self.assertFalse((old / '.prepare.lock').exists())
            self.assertEqual((new / 'val_cache/batch_000000.npz').read_bytes(), payload.read_bytes())

    def test_cpu_ready_reexec_resets_backend_only_after_release(self):
        spec = importlib.util.spec_from_file_location('entry_handoff_test', HERE / 'train_lora_v3.py')
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, {}, clear=True):
            release = Path(temporary) / 'release.json'
            with mock.patch.object(sys, 'argv', ['train_lora_v3.py', '--execute', '--gpu', 'GPU-test', '--name', 'v3_test',
                                                '--preallocate-for-handoff', '--handoff-release-file', str(release)]):
                args = entry.arguments()
                entry.configure_environment(args)
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '')
                self.assertEqual(os.environ['JAX_PLATFORMS'], 'cpu')
                self.assertEqual(os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'], 'false')
                entry.atomic_json(release, {'pid': os.getpid(), 'gpu': args.gpu})
                with mock.patch.object(entry.os, 'execve', side_effect=RuntimeError('captured exec')) as execute:
                    with self.assertRaisesRegex(RuntimeError, 'captured exec'):
                        entry.wait_for_handoff(args, {'peak_lr': 1e-5, 'batch_size': 32, 'steps': 20000})
                child_argv = execute.call_args.args[1]
                self.assertIn('--_handoff-audit-file', child_argv)
                ready = entry.read_json(Path(str(release) + '.ready.json'))
                self.assertEqual(ready['pid'], os.getpid())
                self.assertFalse(ready['gpu_initialized'])
                args._handoff_audit_file = Path(str(release) + '.ready.json')
                entry.configure_environment(args)
                self.assertEqual(os.environ['JAX_PLATFORMS'], 'cuda')
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-test')
                self.assertEqual(os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'], 'true')


if __name__ == '__main__':
    unittest.main()
