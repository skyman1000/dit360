from pathlib import Path
import tempfile
import unittest
from evaluation.prepare_mp3d import stitched_caption


class StitchedTests(unittest.TestCase):
    def test_original_caption_preserved_and_prefix_not_duplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p = root / 'mp3d_skybox/scene/blip3_stitched/view.txt'
            p.parent.mkdir(parents=True)
            for original in ['A hotel room.', 'This is a panorama. A hotel room.']:
                p.write_text(original)
                caption, prompt, source = stitched_caption(root, 'scene', 'view')
                self.assertEqual(caption, original)
                self.assertEqual(prompt, 'This is a panorama. A hotel room.')
                self.assertEqual(source, str(p.resolve()))

    def test_missing_caption_does_not_fall_back_to_direction_caption(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'scene/blip3/view_0.txt'
            p.parent.mkdir(parents=True)
            p.write_text('A room.')
            with self.assertRaisesRegex(ValueError, 'found 0'):
                stitched_caption(Path(directory), 'scene', 'view')

    def test_duplicate_source_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for package in ['one', 'two']:
                p = Path(directory) / package / 'scene/blip3_stitched/view.txt'
                p.parent.mkdir(parents=True)
                p.write_text('A room.')
            with self.assertRaisesRegex(ValueError, 'found 2'):
                stitched_caption(Path(directory), 'scene', 'view')
