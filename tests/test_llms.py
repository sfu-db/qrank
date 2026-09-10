import subprocess
import sys
import unittest
from unittest.mock import patch

from qrank.llms import LLMBackend, load_backend, register_backend
from qrank.llms import registry
from qrank.verifier import cli


class BackendTests(unittest.TestCase):
    def test_import_is_lazy_and_builtin_registers_on_load(self):
        subprocess.run([sys.executable, '-c', '''
import sys
from unittest.mock import patch
import qrank.llms as llms
assert 'qrank.llms.transformers' not in sys.modules
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
from qrank.llms import registry
with patch.object(registry.importlib, 'import_module', wraps=registry.importlib.import_module) as imported:
    # Negative token count fails before loading weights.
    try:
        llms.load_backend('transformers', model='unused', device='cpu', dtype='auto',
                          temperature=0, max_tokens=-1)
    except llms.BackendError:
        pass
    else:
        raise AssertionError('expected configuration error')
    imported.assert_any_call('qrank.llms.transformers')
assert registry._BACKENDS['transformers'].__module__ == 'qrank.llms.transformers'
'''], check=True)

    def test_registered_backend_options_batching_and_identity(self):
        with patch.dict(registry._BACKENDS, clear=True):
            @register_backend('fixture')
            class Fixture(LLMBackend):
                def generate(self, **request):
                    return request['user']

            backend = load_backend('fixture', model='example')
            self.assertEqual(backend.generate_batch([{'user': 'a'}, {'user': 'b'}]), ['a', 'b'])
            self.assertEqual(backend.identity(), {'backend': 'fixture', 'model': 'example'})
            with self.assertRaises(ValueError):
                register_backend('fixture')(Fixture)
            with self.assertRaises(KeyError):
                load_backend('missing', model='example')

    def test_cli_uses_registry(self):
        args = cli.build_parser().parse_args(['transform-dataset', '--backend', 'transformers'])
        with patch.object(cli, 'load_backend') as loader:
            self.assertIs(cli.make_backend(args), loader.return_value)
        loader.assert_called_once_with('transformers', model=args.model, device=args.device,
                                       dtype=args.dtype, temperature=args.temperature,
                                       max_tokens=args.max_tokens, revision=args.revision)
