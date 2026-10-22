from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROTO_SHA256 = '4715d16f43dff5e8e21c76d5b564cced592722b5734b6946dc3af1da665f53f3'
THRESHOLD = 0.02152947708964348


def is_han(char: str) -> bool:
    return len(char) == 1 and (
        0x4E00 <= ord(char) <= 0x9FFF
        or 0x3400 <= ord(char) <= 0x4DBF
        or 0x20000 <= ord(char) <= 0x2EBEF
        or 0xF900 <= ord(char) <= 0xFAFF
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description='Validate the standalone TextVerifier release.')
    parser.add_argument('--root', type=Path, default=PACKAGE_ROOT)
    parser.add_argument('--check-checkpoint', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    benchmark = root / 'benchmark'
    assets = root / 'assets'

    expected_ids = [f'{index:03d}.png' for index in range(308)]
    image_dir = benchmark / 'images'
    images = sorted(path.name for path in image_dir.iterdir() if path.is_file())
    if images != expected_ids or any(path.is_symlink() for path in image_dir.iterdir()):
        raise ValueError('benchmark images must be exactly anonymous IDs 000.png through 307.png without symlinks')

    rows = [json.loads(line) for line in (benchmark / 'annotations.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    fields = {'sample_id', 'image_id', 'image', 'item_index', 'char', 'box', 'rect', 'render_error'}
    if len(rows) != 7358 or sum(bool(row['render_error']) for row in rows) != 642:
        raise ValueError('benchmark row or positive count mismatch')
    if len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('sample IDs are not unique')
    benchmark_chars = {row['char'] for row in rows}
    for row in rows:
        if set(row) != fields or not is_han(row['char']):
            raise ValueError(f'invalid annotation row: {row.get("sample_id")}')
        image = Path(row['image'])
        if image.is_absolute() or '..' in image.parts or not (benchmark / image).is_file():
            raise ValueError(f'invalid image path: {row["image"]}')
        x0, y0, x1, y1 = map(float, row['rect'])
        if not all(math.isfinite(value) for value in (x0, y0, x1, y1)) or x1 <= x0 or y1 <= y0:
            raise ValueError(f'invalid rectangle: {row["sample_id"]}')

    stats = json.loads((benchmark / 'stats.json').read_text(encoding='utf-8'))
    if stats != {'images': 308, 'character_crops': 7358, 'render_errors': 642, 'render_correct': 6716}:
        raise ValueError('aggregate stats mismatch')

    proto_path = assets / 'textverifier_k10_q90.pt'
    if sha256(proto_path) != PROTO_SHA256:
        raise ValueError('prototype asset hash mismatch')
    pack = torch.load(proto_path, map_location='cpu', weights_only=True)
    prototypes = pack['prototypes']
    if tuple(prototypes.shape) != (8189, 10, 1024) or int(pack['K']) != 10 or float(pack['thr']) != THRESHOLD:
        raise ValueError('prototype shape, K, or threshold mismatch')
    if len(pack['chars']) != 8189 or len(pack['char_to_index']) != 8189:
        raise ValueError('prototype character mapping mismatch')
    if any(pack['char_to_index'][char] != index for index, char in enumerate(pack['chars'])):
        raise ValueError('prototype mapping is not row aligned')
    if not torch.isfinite(prototypes).all() or not bool(torch.all((prototypes.float().norm(dim=-1) > 0.999) & (prototypes.float().norm(dim=-1) < 1.001))):
        raise ValueError('invalid prototype vectors')

    forbidden = ('/mnt/',)
    for path in root.rglob('*'):
        if path == Path(__file__).resolve() or not path.is_file() or path.suffix.lower() not in {'.md', '.json', '.jsonl', '.py', '.txt'}:
            continue
        if any(token in path.read_text(encoding='utf-8') for token in forbidden):
            raise ValueError(f'private/source-specific path found: {path.relative_to(root)}')

    if args.check_checkpoint:
        state = torch.load(assets / 'textverifier_stage3.pth', map_location='cpu', weights_only=True)
        state = state['model_state_dict'] if 'model_state_dict' in state else state
        if not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
            raise ValueError('invalid checkpoint state dict')

    asset_files = [assets / 'textverifier_stage3.pth', assets / 'textverifier_k10_q90.pt', assets / 'prototype_metadata.json']
    expected_manifest = ''.join(f'{sha256(path)}  {path.name}\n' for path in asset_files)
    if (assets / 'manifest.sha256').read_text(encoding='utf-8') != expected_manifest:
        raise ValueError('asset manifest mismatch')

    print(json.dumps({
        'images': len(images),
        'rows': len(rows),
        'positives': sum(bool(row['render_error']) for row in rows),
        'benchmark_unique_characters': len(benchmark_chars),
        'prototype_characters': len(pack['chars']),
        'prototype_shape': list(prototypes.shape),
        'threshold': float(pack['thr']),
        'checkpoint_checked': args.check_checkpoint,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
