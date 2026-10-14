"""The preprocessing stage retains every year and produces one square forward view."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]


class Taipei101Test(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw = self.root / 'raw'
        (self.raw / 'Data').mkdir(parents=True)
        (self.raw / 'Meta').mkdir()
        self.output = self.root / 'output'
        self.records = [('db1', 2017, 25.0), ('db2', 2018, 25.01),
                        ('near1', 2025, 25.00005), ('near2', 2024, 25.01005),
                        ('far', 2025, 25.02)]
        (self.raw / 'panoIDList.txt').write_text(''.join(f'{pid} 121.55 {lat}\n' for pid, _, lat in self.records))
        pixels = np.zeros((64, 128, 3), dtype=np.uint8)
        pixels[:] = [240, 20, 20]
        pixels[:, 48:80] = [20, 240, 20]
        for pid, year, lat in self.records:
            Image.fromarray(pixels).save(self.raw / 'Data' / f'{pid}.jpg')
            (self.raw / 'Meta' / f'{pid}.json').write_text(json.dumps(
                {'id': pid, 'lon': 121.55, 'lat': lat, 'date': {'year': year, 'month': 2}}))

    def run_formatter(self, *extra):
        return subprocess.run([sys.executable, str(PROJECT / 'format_taipei101.py'),
                               '--raw-root', str(self.raw), '--output-root', str(self.output),
                               '--size', '64', '--metadata-workers', '2', '--image-workers', '2', *extra],
                              capture_output=True, text=True)

    def summary(self):
        return json.loads((self.output / 'reports/summary.json').read_text())

    def test_all_years_one_square_forward_view_and_resume(self):
        result = self.run_formatter()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        images = list((self.output / 'images/all').glob('*.jpg'))
        self.assertEqual(len(images), 5)  # The geographically isolated query is retained too.
        self.assertEqual({p.name.split('@')[7] for p in images}, {r[0] for r in self.records})
        self.assertTrue(all(p.name.split('@')[8] == '00' for p in images))
        self.assertEqual(self.summary()['horizontal_fov_deg'], 90)
        self.assertEqual(self.summary()['vertical_fov_deg'], 90)
        with Image.open(images[0]) as image:
            self.assertEqual(image.size, (64, 64))
            red, green, blue = image.getpixel((32, 32))
            self.assertGreater(green, red + 100)
        result = self.run_formatter('--resume')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.summary()['reused_images'], 5)
        self.assertEqual(self.summary()['written_images'], 0)
        self.assertEqual(self.run_formatter('--resume', '--size', '128').returncode, 2)

    def test_dry_run_and_corrupt_source_reporting(self):
        self.assertEqual(self.run_formatter('--dry-run').returncode, 0)
        self.assertFalse(self.output.exists())
        (self.raw / 'Data/far.jpg').write_bytes(b'corrupt source')
        result = self.run_formatter()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.summary()['errors'], 1)
        self.assertEqual(self.summary()['output_images'], 4)
        names = (self.output / 'images/all_images_paths.txt').read_text().splitlines()
        self.assertFalse(any('@far@' in name for name in names))


if __name__ == '__main__':
    unittest.main()
