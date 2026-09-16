import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from qrank.config import LLMConfig
from qrank.llms import BackendError, available_backends, load_backend
from qrank.baselines.selector import cli as selector_cli
from qrank.baselines.selector.config import load_selector_config
from qrank.verifier import cli as verifier_cli


OPTIONS = dict(model='fixture', device='cuda', dtype='auto', temperature=0,
               max_tokens=8, revision='commit')


class VLLMTests(unittest.TestCase):
    def setUp(self):
        self.engine = MagicMock()
        self.engine.llm_engine.model_config = SimpleNamespace(
            max_model_len=32, dtype='torch.bfloat16',
            hf_config=SimpleNamespace(_commit_hash='resolved'),
        )
        self.engine.get_tokenizer.return_value.apply_chat_template.return_value = [1, 2, 3]
        self.engine.generate.return_value = [
            SimpleNamespace(outputs=[SimpleNamespace(text=' first ', finish_reason='stop', token_ids=[1])]),
            SimpleNamespace(outputs=[SimpleNamespace(text=' second ', finish_reason='stop', token_ids=[2])]),
        ]
        self.module = SimpleNamespace(
            LLM=MagicMock(return_value=self.engine),
            SamplingParams=MagicMock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs)),
            __version__='test',
        )
        self.structured = SimpleNamespace(StructuredOutputsParams=MagicMock(side_effect=lambda **kw: SimpleNamespace(**kw)))
        self.modules = patch.dict(sys.modules, {'vllm': self.module, 'vllm.sampling_params': self.structured})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_lazy_registration(self):
        self.assertIn('vllm', available_backends())
        subprocess.run([sys.executable, '-c', '''
import sys
import qrank.llms as llms
assert 'vllm' in llms.available_backends()
assert 'qrank.llms.vllm' not in sys.modules
assert 'torch' not in sys.modules
assert 'vllm' not in sys.modules
try:
    llms.load_backend('vllm', model='unused', device='cuda', dtype='auto',
                      temperature=0, max_tokens=-1)
except ValueError:
    pass
else:
    raise AssertionError('expected configuration error')
assert 'qrank.llms.vllm' in sys.modules
assert 'vllm' not in sys.modules
'''], check=True)

    def test_batch_preserves_order_and_prompt_contract(self):
        backend = load_backend('vllm', **OPTIONS, max_model_len=32)
        requests = [dict(question_id=str(i), cluster_id='c', system='system', user=str(i))
                    for i in range(2)]
        self.assertEqual(backend.generate_batch(requests), ['first', 'second'])
        template = self.engine.get_tokenizer.return_value.apply_chat_template
        self.assertEqual(template.call_count, 2)
        template.assert_any_call(
            [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': '0'}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
        )
        self.engine.generate.assert_called_once_with(
            [{'prompt_token_ids': [1, 2, 3]}] * 2, backend.sampling_params, use_tqdm=False,
        )
        self.module.LLM.assert_called_once_with(
            model='fixture', revision='commit', tokenizer_revision='commit', dtype='auto',
            generation_config='vllm', tensor_parallel_size=1, gpu_memory_utilization=0.9,
            enable_prefix_caching=True, quantization=None, seed=0, max_model_len=32,
            structured_outputs_config={'backend': 'xgrammar'},
        )
        self.assertEqual(backend.identity()['resolved_revision'], 'resolved')
        self.assertEqual(backend.identity()['backend'], 'vllm')
        json.dumps(backend.identity())
        self.assertEqual(backend.generate_batch([]), [])
        self.assertEqual(self.engine.generate.call_count, 1)
        self.engine.generate.return_value = self.engine.generate.return_value[:1]
        self.assertEqual(backend.generate(**requests[0]), 'first')

    def test_structured_schema_is_per_request_and_does_not_change_selector_sampling(self):
        backend = load_backend('vllm', **OPTIONS)
        schemas = [{'type': 'object'}, {'type': 'array'}]
        requests = [dict(system='s', user='u', response_schema=schema) for schema in schemas]
        backend.generate_batch(requests)
        params = self.engine.generate.call_args.args[1]
        self.assertEqual([p.structured_outputs.json for p in params], schemas)
        self.assertEqual(backend.identity()['structured_outputs_config'], {'backend': 'xgrammar'})
        self.assertFalse(hasattr(backend.sampling_params, 'structured_outputs'))
        self.engine.generate.return_value = self.engine.generate.return_value[:1]
        self.engine.generate.return_value[0].outputs[0].finish_reason = 'length'
        with self.assertRaisesRegex(BackendError, 'did not finish.*raw_text'):
            backend.generate_batch(requests[:1])

    def test_structured_choices_are_enforced_per_request(self):
        backend = load_backend('vllm', **OPTIONS)
        requests = [
            dict(system='s', user='u', response_choices=['A', 'B', 'NONE']),
            dict(system='s', user='u', response_choices=['A', 'NONE']),
        ]
        backend.generate_batch(requests)
        params = self.engine.generate.call_args.args[1]
        self.assertEqual(
            [item.structured_outputs.choice for item in params],
            [['A', 'B', 'NONE'], ['A', 'NONE']],
        )
        for choices in ([], ['A', 'A'], ['A', '']):
            with self.subTest(choices=choices), self.assertRaisesRegex(
                    BackendError, 'response_choices'):
                backend.generate_batch([dict(system='s', user='u', response_choices=choices)])

    def test_sampling_and_identity(self):
        greedy = load_backend('vllm', **OPTIONS)
        self.assertEqual((greedy.sampling_params.top_p, greedy.sampling_params.top_k), (1, -1))
        sampled = load_backend('vllm', **{**OPTIONS, 'temperature': 0.7}, seed=42,
                               tensor_parallel_size=2, gpu_memory_utilization=0.75)
        self.module.SamplingParams.assert_called_with(
            temperature=0.7, max_tokens=8, top_p=0.8, top_k=20,
        )
        self.assertNotEqual(greedy.identity(), sampled.identity())
        self.assertEqual(sampled.identity()['seed'], 42)

    def test_context_limit_fails_before_generation(self):
        backend = load_backend('vllm', **OPTIONS)
        template = self.engine.get_tokenizer.return_value.apply_chat_template
        template.return_value = [1] * 25
        with self.assertRaisesRegex(BackendError, 'context window'):
            backend.generate_batch([dict(system='s', user='u')])
        self.engine.generate.assert_not_called()

    def test_engine_failures_and_missing_completions(self):
        backend = load_backend('vllm', **OPTIONS)
        for outputs in ([], [SimpleNamespace(outputs=[])]):
            self.engine.generate.return_value = outputs
            with self.assertRaisesRegex(BackendError, 'one completion'):
                backend.generate_batch([dict(system='s', user='u')])
        self.engine.generate.side_effect = RuntimeError('engine failed')
        with self.assertRaisesRegex(RuntimeError, 'engine failed'):
            backend.generate_batch([dict(system='s', user='u')])
        self.module.LLM.side_effect = ValueError('bad engine settings')
        with self.assertRaisesRegex(ValueError, 'bad engine settings'):
            load_backend('vllm', **OPTIONS)

    def test_invalid_settings_and_missing_dependency(self):
        invalid = [dict(device='cuda:1'), dict(device='cpu'), dict(max_tokens=0),
                   dict(temperature=float('nan')), dict(tensor_parallel_size=0),
                   dict(gpu_memory_utilization=1.1), dict(gpu_memory_utilization=True),
                   dict(max_model_len=8), dict(enable_prefix_caching='yes'), dict(seed=-1)]
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises((BackendError, ValueError)):
                load_backend('vllm', **{**OPTIONS, **overrides})
        self.module.LLM.assert_not_called()
        with patch.dict(sys.modules, {'vllm': None}):
            with self.assertRaisesRegex(BackendError, 'uv sync --extra vllm'):
                load_backend('vllm', **OPTIONS)


class BackendConfigTests(unittest.TestCase):
    def test_options_reach_both_clis(self):
        data = dict(backend='vllm', **OPTIONS, backend_kwargs={'max_model_len': 32})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            path.write_text(json.dumps(data))
            args = verifier_cli.build_parser().parse_args([
                'transform-dataset', '--config', str(path), '--max-tokens', '12',
            ])
            with patch.object(verifier_cli, 'load_backend') as loader:
                verifier_cli.make_backend(args)
                loader.assert_called_once_with('vllm', **{**OPTIONS, 'max_tokens': 12},
                                               max_model_len=32)
            path.write_text(json.dumps({**data, 'output_dir': tmp + '/output',
                                       'dataset': {'name': 'fixture', 'path': 'input'}}))
            self.assertEqual(load_selector_config(path).backend_options(),
                             {**OPTIONS, 'max_model_len': 32})
            with patch.object(selector_cli, 'load_backend') as loader, \
                    patch.object(selector_cli, 'prepare_output_dir'), \
                    patch.object(selector_cli, 'run_target', side_effect=ValueError('stop')), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(selector_cli.main(['--config', str(path)]), 2)
                loader.assert_called_once_with('vllm', **OPTIONS, max_model_len=32)

    def test_options_cannot_shadow_shared_settings(self):
        for value in (None, [], {'model': 'other'}, {'backend': 'other'}, {1: 2}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'backend_kwargs'):
                LLMConfig.from_dict(dict(backend='vllm', **OPTIONS, backend_kwargs=value))

    def test_empty_options_preserve_existing_constructor_arguments(self):
        config = LLMConfig.from_dict(dict(backend='transformers', **OPTIONS))
        self.assertEqual(config.backend_options(), OPTIONS)
