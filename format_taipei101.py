"""Prepare all Taipei101 panoramas as 90-degree square forward views."""

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import utm
from PIL import Image



DEFAULT_RAW_ROOT = Path('/home/sxl/lzj/Data_raw/TAIBEI101')


def read_index(root):
    records = []
    seen = set()
    with (root / 'panoIDList.txt').open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f'Index line {number}: expected ID longitude latitude')
            pid, lon, lat = fields
            if not re.fullmatch(r'[A-Za-z0-9_-]+', pid) or pid in seen:
                raise ValueError(f'Index line {number}: unsafe or duplicate ID {pid!r}')
            lon, lat = float(lon), float(lat)
            if not (math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180 and -90 <= lat <= 90):
                raise ValueError(f'Index line {number}: invalid coordinates')
            seen.add(pid)
            records.append({'pano_id': pid, 'longitude': lon, 'latitude': lat})
    if not records:
        raise ValueError('Empty panorama index')
    return records

def read_metadata(root, record):
    pid = record['pano_id']
    meta = json.loads((root / 'Meta' / f'{pid}.json').read_text(encoding='utf-8'))
    if not isinstance(meta, dict):
        raise ValueError('JSON metadata must be an object')
    if meta.get('id') != pid:
        raise ValueError('JSON ID does not match filename/index')
    for key, index_key in [('lon', 'longitude'), ('lat', 'latitude')]:
        if abs(float(meta[key]) - record[index_key]) > 1e-9 or not math.isfinite(float(meta[key])):
            raise ValueError(f'JSON {key} does not match index')
    date = meta.get('date') or {}
    if not isinstance(date, dict):
        raise ValueError('Capture date must be an object')
    year, month = date.get('year'), date.get('month')
    if type(year) is not int or type(month) is not int or not (1 <= year <= 9999 and 1 <= month <= 12):
        raise ValueError('Missing/invalid capture year or month')
    return meta, f'{year:04d}-{month:02d}'

def ensure_output_outside_raw(root, output):
    root, output = root.resolve(), output.resolve()
    if output == root or root in output.parents:
        raise ValueError('Output must be outside the raw dataset')


def perspective_rays(height, width, horizontal_fov, yaw, pitch):
    """Unit rays: x right, y up, z forward; positive yaw right/pitch up."""
    focal = width / (2 * math.tan(math.radians(horizontal_fov) / 2))
    xx, yy = np.meshgrid((np.arange(width) + .5 - width / 2) / focal,
                         (height / 2 - np.arange(height) - .5) / focal)
    zz = np.ones_like(xx)
    pitch, yaw = math.radians(pitch), math.radians(yaw)
    yy, zz = math.cos(pitch) * yy + math.sin(pitch) * zz, -math.sin(pitch) * yy + math.cos(pitch) * zz
    xx, zz = math.cos(yaw) * xx + math.sin(yaw) * zz, -math.sin(yaw) * xx + math.cos(yaw) * zz
    norm = np.sqrt(xx * xx + yy * yy + zz * zz)
    return np.arctan2(xx, zz), np.arcsin(np.clip(yy / norm, -1, 1))

def project(image, angles):
    """Bilinear spherical sampling with horizontal wrap and vertical clamp."""
    azimuth, latitude = angles
    height, width = image.shape[:2]
    u = (azimuth / (2 * np.pi) + .5) * width - .5
    v = np.clip((.5 - latitude / np.pi) * height - .5, 0, height - 1)
    x, y = np.floor(u).astype(int), np.floor(v).astype(int)
    dx, dy = (u - x)[..., None], (v - y)[..., None]
    x0, x1 = x % width, (x + 1) % width
    y1 = np.minimum(y + 1, height - 1)
    result = ((1 - dy) * ((1 - dx) * image[y, x0] + dx * image[y, x1])
              + dy * ((1 - dx) * image[y1, x0] + dx * image[y1, x1]))
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)

def output_name(record, date, view_index, yaw, pitch, utm_coordinates):
    east, north, zone, letter = utm_coordinates
    # Heading left blank: yaw is relative to panorama, not calibrated world bearing.
    fields = [f'{east:.3f}', f'{north:.3f}', f'{zone:02d}', letter,
              f"{record['latitude']:.8f}", f"{record['longitude']:.8f}",
              record['pano_id'], f'{view_index:02d}', '', '', '', '',
              date.replace('-', ''), f'yaw_rel_{yaw:g}_pitch_rel_{pitch:g}', '.jpg']
    return '@' + '@'.join(fields)


def load_records(root, workers, metadata_csv=None):
    """Return all valid index records and explicit metadata errors in index order."""
    index = read_index(root)
    if metadata_csv is not None:
        with metadata_csv.open(newline='', encoding='utf-8') as stream:
            cached = list(csv.DictReader(stream))
        by_id = {r['pano_id']: r for r in cached}
        if len(by_id) != len(cached) or set(by_id) != {r['pano_id'] for r in index}:
            raise ValueError('Cached metadata CSV must contain every index ID exactly once')
        records = []
        for record in index:
            saved = by_id[record['pano_id']]
            date = saved['date']
            if not re.fullmatch(r'[0-9]{4}-(?:0[1-9]|1[0-2])', date) or date.startswith('0000'):
                raise ValueError('Cached date must be YYYY-MM with a positive year')
            for key in ['longitude', 'latitude']:
                if not math.isfinite(float(saved[key])) or abs(float(saved[key]) - record[key]) > 1e-9:
                    raise ValueError('Cached coordinates disagree with index')
            records.append(dict(record, date=date))
        return index, records, []

    def read(record):
        try:
            _, date = read_metadata(root, record)
            return dict(record, date=date), None
        except (OSError, ValueError, TypeError, KeyError) as exc:
            return None, {'pano_id': record['pano_id'], 'stage': 'metadata', 'error': str(exc)}

    records, errors = [], []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(index), 1024):
            batch = index[start:start + 1024]
            for record, error in executor.map(read, batch):
                (errors if error else records).append(error if error else record)
            print(f'Metadata: {start + len(batch)}/{len(index)}', flush=True)
    return index, records, errors


def label_records(records):
    zones = set()
    labelled = []
    for record in records:
        coordinates = utm.from_latlon(record['latitude'], record['longitude'])
        east, north, zone, letter = coordinates
        zones.add((zone, letter >= 'N'))
        name = output_name(record, record['date'], 0, 0, 0, coordinates)
        fields = name.split('@')
        labelled.append(dict(record, output_name=name, utm_easting=float(fields[1]),
                             utm_northing=float(fields[2]), utm_zone=zone, utm_letter=letter))
    if len(zones) != 1:
        raise ValueError('Expected one UTM zone and hemisphere')
    return labelled


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def write_csv(path, records, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def convert_one(record, args, maps):
    target = args.output_root / 'images/all' / record['output_name']
    temporary = target.with_suffix('.jpg.tmp')
    try:
        if args.resume and target.exists():
            try:
                with Image.open(target) as existing:
                    if existing.format != 'JPEG' or existing.size != (args.size, args.size):
                        raise ValueError('Wrong existing output dimensions/format')
                    existing.load()
                return dict(record, output_path=str(target.relative_to(args.output_root)), reused=True), None
            except (OSError, ValueError):
                pass  # Regenerate incomplete/corrupt files from the original panorama.
        source = args.raw_root / 'Data' / f"{record['pano_id']}.jpg"
        with Image.open(source) as image:
            if image.width != 2 * image.height:
                raise ValueError('Expected a 2:1 equirectangular image')
            if args.max_input_width and image.width > args.max_input_width:
                image.draft('RGB', (args.max_input_width, args.max_input_width // 2))
            image = image.convert('RGB')
            if args.max_input_width and image.width > args.max_input_width:
                image.thumbnail((args.max_input_width, args.max_input_width // 2), Image.Resampling.LANCZOS)
            view = project(np.asarray(image), maps)
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(view).save(temporary, format='JPEG', quality=args.jpeg_quality)
        temporary.replace(target)
        return dict(record, output_path=str(target.relative_to(args.output_root)), reused=False), None
    except (OSError, ValueError, TypeError) as exc:
        temporary.unlink(missing_ok=True)
        # These are generated files owned by this formatter; avoid publishing stale failures.
        target.unlink(missing_ok=True)
        return None, {'pano_id': record['pano_id'], 'stage': 'image', 'error': str(exc)}


def convert_many(records, args, maps):
    successful, errors = [], []
    with ThreadPoolExecutor(max_workers=args.image_workers) as executor:
        for start in range(0, len(records), 128):
            batch = records[start:start + 128]
            for result, error in executor.map(lambda r: convert_one(r, args, maps), batch):
                (errors if error else successful).append(error if error else result)
            print(f'Images: {start + len(batch)}/{len(records)}', flush=True)
    return successful, errors


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--raw-root', type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument('--output-root', type=Path, default=Path('datasets/taipei101_forward'))
    parser.add_argument('--size', type=int, default=512, help='Square output side; FOV is always 90 x 90 degrees')
    parser.add_argument('--max-input-width', type=int, default=4096)
    parser.add_argument('--jpeg-quality', type=int, default=95)
    parser.add_argument('--metadata-workers', type=int, default=16)
    parser.add_argument('--image-workers', type=int, default=2)
    parser.add_argument('--metadata-csv', type=Path, help='Reuse a trusted complete metadata CSV (pano_id, longitude, latitude, date)')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Validate all metadata without writing files or decoding JPEGs')
    args = parser.parse_args()
    try:
        ensure_output_outside_raw(args.raw_root, args.output_root)
        if min(args.size, args.metadata_workers, args.image_workers) <= 0 or args.size > 4096:
            raise ValueError('Size/workers must be positive; square side is limited to 4096')
        if args.max_input_width < 0 or args.max_input_width == 1 or not 1 <= args.jpeg_quality <= 100:
            raise ValueError('Invalid input width or JPEG quality')
    except ValueError as exc:
        parser.error(str(exc))
    return parser, args


def main():
    parser, args = parse_args()
    try:
        index, records, errors = load_records(args.raw_root, args.metadata_workers, args.metadata_csv)
        labelled = label_records(records)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    summary = {'indexed_panoramas': len(index), 'valid_metadata': len(records),
               'year_counts': dict(sorted(Counter(r['date'][:4] for r in records).items())),
               'horizontal_fov_deg': 90, 'vertical_fov_deg': 90, 'size': args.size,
               'forward_definition': 'panorama image center, relative yaw=0, pitch=0; no world-heading correction',
               'dry_run': args.dry_run}
    if args.dry_run:
        print(json.dumps(dict(summary, metadata_errors=len(errors)), indent=2))
        return 1 if errors else 0
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    config = {key: getattr(args, key) for key in ['size', 'max_input_width', 'jpeg_quality']}
    config.update(raw_root=str(args.raw_root.resolve()), metadata_sha256=fingerprint,
                  projection='all_forward_square_90_v2', horizontal_fov=90, vertical_fov=90)
    config_path = args.output_root / 'preprocess_config.json'
    if args.output_root.exists() and any(args.output_root.iterdir()):
        if not args.resume or not config_path.exists() or json.loads(config_path.read_text()) != config:
            parser.error('Nonempty output requires --resume with identical projection/metadata; choose a new output root')
    (args.output_root / 'images/all').mkdir(parents=True, exist_ok=True)
    atomic_text(config_path, json.dumps(config, indent=2) + '\n')
    atomic_text(args.output_root / 'images/all_images_paths.txt', '')
    atomic_text(args.output_root / 'reports/summary.json', json.dumps(dict(summary, status='in_progress'), indent=2) + '\n')
    Image.MAX_IMAGE_PIXELS = 200_000_000
    maps = perspective_rays(args.size, args.size, 90, 0, 0)
    successful, image_errors = convert_many(labelled, args, maps)
    errors.extend(image_errors)
    fields = ['pano_id', 'date', 'longitude', 'latitude', 'utm_easting', 'utm_northing', 'utm_zone', 'utm_letter',
              'output_name', 'output_path', 'reused']
    reports = args.output_root / 'reports'
    write_csv(reports / 'images.csv', successful, fields)
    write_csv(reports / 'errors.csv', errors, ['pano_id', 'stage', 'error'])
    summary.update(output_images=len(successful), written_images=sum(not r['reused'] for r in successful),
                   reused_images=sum(r['reused'] for r in successful), errors=len(errors),
                   status='complete' if not errors and len(successful) == len(index) else 'incomplete')
    atomic_text(args.output_root / 'images/all_images_paths.txt',
                ''.join(r['output_name'] + '\n' for r in sorted(successful, key=lambda r: r['output_name'])))
    atomic_text(reports / 'summary.json', json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return 0 if summary['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
