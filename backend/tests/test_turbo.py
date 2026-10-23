"""Run with: python -m unittest discover -s tests -v (no model weights needed)."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

# Keep test data outside the user's model and image folders.
_temp = tempfile.TemporaryDirectory()
os.environ['IMAGEGEN_HOME'] = str(Path(_temp.name) / 'home')
os.environ['IMAGEGEN_OUTPUT_DIR'] = str(Path(_temp.name) / 'output')

from imagegen import catalog, downloader
from imagegen.engine import Engine, GenerateParams


class TurboTests(unittest.TestCase):
    def setUp(self):
        with patch('imagegen.engine.threading.Thread'):
            self.engine = Engine()
        self.engine.status = 'ready'
        self.engine.model_id = 'qwen-image-2.1-turbo'
        self.memory = patch('imagegen.engine.max_resolution', return_value=1536)
        self.memory.start()
        self.addCleanup(self.memory.stop)

    def test_separate_download_and_install_state(self):
        base = catalog.get('qwen-image-2.1')
        turbo = catalog.get(self.engine.model_id)
        self.assertEqual(turbo.repo, 'Qwen/Qwen-Image-2.1-Turbo')
        self.assertNotEqual(base.local_dir, turbo.local_dir)
        turbo.local_dir.mkdir(parents=True, exist_ok=True)
        marker = turbo.local_dir / downloader.COMPLETE_MARKER
        marker.write_text('{}')
        self.addCleanup(marker.unlink)
        self.assertTrue(downloader.is_installed(turbo))
        self.assertFalse(downloader.is_installed(base))

    def test_turbo_defaults_at_all_qualities(self):
        for resolution in (512, 1024, 1536):
            job = self.engine.submit(GenerateParams(prompt='a bird', resolution=resolution))
            self.assertEqual(job.params.steps, 8)
            self.assertEqual(job.total_steps, 8)
            self.assertEqual(job.params.width, resolution)

    def test_turbo_custom_steps_rejected(self):
        for steps in (0, 4, 20, 40, 100):
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, 'fixed 8-step'):
                self.engine.submit(GenerateParams(prompt='a bird', steps=steps))
        self.assertEqual(self.engine.submit(GenerateParams(prompt='a bird', steps=8)).total_steps, 8)

    def test_base_defaults_and_custom_steps_unchanged(self):
        self.engine.model_id = 'qwen-image-2.1'
        for requested, expected in ((None, 40), (20, 20), (101, 100)):
            job = self.engine.submit(GenerateParams(prompt='a bird', steps=requested))
            self.assertEqual(job.total_steps, expected)

    def test_generation_preserves_checkpoint_schedule_and_progress(self):
        from PIL import Image
        ref = Path(_temp.name) / 'reference.png'
        Image.new('RGB', (256, 256)).save(ref)
        for refs in ([], [str(ref)]):
            job = self.engine.submit(GenerateParams(prompt='a bird', transparent=True, reference_images=refs))
            def pipeline(**kwargs):
                self.assertNotIn('sigmas', kwargs)
                self.assertEqual(kwargs['num_inference_steps'], 8)
                self.assertEqual(kwargs['true_cfg_scale'], 1.0)
                self.assertEqual(len(kwargs['image'] or []), len(refs))
                for step in range(8):
                    kwargs['callback_on_step_end'](None, step, None, {})
                return SimpleNamespace(images=[Image.new('RGBA', (256, 256))])
            self.engine._pipe = pipeline
            torch = SimpleNamespace(Generator=Mock())
            with patch.dict('sys.modules', {'torch': torch}), patch('imagegen.engine.memory_pressure_critical', return_value=False):
                self.engine._run(job)
            self.assertEqual(job.status, 'done')
            self.assertEqual(job.step, job.total_steps)
            self.assertEqual(job.image['steps'], 8)
            self.assertEqual(job.image['model_id'], 'qwen-image-2.1-turbo')


if __name__ == '__main__':
    unittest.main()
