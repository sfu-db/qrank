import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from qrank.baselines.selector import cli as selector_cli
from qrank.baselines.selector.config import load_selector_config
from qrank.config import VerifierConfig
from qrank.verifier import cli


class ConfigTests(unittest.TestCase):
    def test_supplied_presets_load(self):
        config = VerifierConfig.from_file('training_configs/verifier_qwen3_1p7b.json')
        self.assertEqual(config.model, 'Qwen/Qwen3-1.7B')
        self.assertEqual((config.temperature, config.max_tokens), (0.7, 700))
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(VerifierConfig.from_file('training_configs/verifier_qwen3_8b.json').batch_size, 16)
        for path in Path('training_configs').glob('selector_*.json'):
            with self.subTest(path=path):
                config = load_selector_config(path)
                configured = json.loads(path.read_text())
                self.assertEqual((config.temperature, config.max_tokens),
                                 (configured['temperature'], configured['max_tokens']))

    def test_invalid_config_files_fail_before_backend_loading(self):
        valid = dict(backend='transformers', model='fixture', device='cpu',
                     dtype='auto', temperature=0, max_tokens=128, revision='main')
        invalid = [
            'not JSON', '[]', '{}',
            json.dumps({**valid, 'typo': True}),
            json.dumps({**valid, 'model': None}),
            json.dumps({**valid, 'dtype': 'invalid'}),
            *[json.dumps({**valid, 'temperature': value})
              for value in (-1, float('nan'), float('inf'), True, '0')],
            *[json.dumps({**valid, 'max_tokens': value})
              for value in (0, -1, 1.5, True, '128')],
            *[json.dumps({**valid, 'batch_size': value})
              for value in (0, -1, 1.5, True, '16', None)],
            json.dumps({**valid, 'backend_kwargs': {'batch_size': 16}}),
        ]
        with tempfile.TemporaryDirectory() as tmp, patch.object(cli, 'load_backend') as loader:
            path = Path(tmp) / 'config.json'
            args = cli.build_parser().parse_args(['transform-dataset', '--config', str(path)])
            with self.assertRaises(FileNotFoundError):
                cli.make_backend(args)
            for content in invalid:
                with self.subTest(content=content):
                    path.write_text(content)
                    with self.assertRaises(ValueError):
                        cli.make_backend(args)
            loader.assert_not_called()

    def test_cli_batch_size_config_override_and_fallback(self):
        data = dict(backend='transformers', model='fixture', device='cpu',
                    dtype='auto', temperature=0, max_tokens=128, revision='main')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            for configured, flags, expected in (({}, [], 8), ({'batch_size': 16}, [], 16),
                                                 ({'batch_size': 16}, ['--batch-size', '3'], 3)):
                path.write_text(json.dumps({**data, **configured}))
                args = cli.build_parser().parse_args([
                    'transform-dataset', '--config', str(path), *flags,
                ])
                self.assertEqual(cli.load_config(args).batch_size, expected)
                with patch.object(cli, 'load_backend') as loader:
                    cli.make_backend(args)
                    loader.assert_called_once_with('transformers', **{k: v for k, v in data.items()
                                                                       if k != 'backend'})
            args.batch_size = 0
            with patch.object(cli, 'load_backend') as loader, self.assertRaisesRegex(ValueError, 'batch_size'):
                cli.make_backend(args)
            loader.assert_not_called()

    def test_both_clis_require_explicit_config_paths(self):
        for parser, argv in ((cli.build_parser(), ['transform-dataset']),
                             (selector_cli.build_parser(), [])):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    parser.parse_args(argv)
                self.assertEqual(exc.exception.code, 2)

    def test_selector_does_not_fill_missing_model_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'selector.json'
            path.write_text(json.dumps({
                'output_dir': 'output', 'dataset': {'name': 'fixture', 'path': 'input'},
            }))
            with self.assertRaisesRegex(ValueError, 'Invalid LLM config'):
                load_selector_config(path)
