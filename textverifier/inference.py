from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .ids_dictionary import IDSDictionary
from .model import build_model

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PACKAGE_ROOT / 'assets' / 'textverifier_stage3.pth'
DEFAULT_PROTOTYPES = PACKAGE_ROOT / 'assets' / 'textverifier_k10_q90.pt'
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / 'data'


class TextVerifier:
    """Loads the frozen Stage-3 verifier and per-character K=10 prototypes."""

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        prototypes: str | Path = DEFAULT_PROTOTYPES,
        device: str = 'cuda',
        dtype: str = 'float32',
    ) -> None:
        if dtype not in {'float32', 'bfloat16'}:
            raise ValueError(f'unsupported dtype: {dtype}')
        self.device = torch.device(device if device.startswith('cuda') and torch.cuda.is_available() else 'cpu')
        self.model_dtype = torch.bfloat16 if dtype == 'bfloat16' else torch.float32
        self.bfloat16_features = dtype == 'bfloat16'
        ids = IDSDictionary.load(DEFAULT_DATA_DIR)
        self.model = build_model(ids, model_size='large', embed_dim=1024, img_size=128, patch_size=16)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state, strict=True)
        self.model = self.model.to(device=self.device, dtype=self.model_dtype).eval()

        pack = torch.load(prototypes, map_location='cpu', weights_only=True)
        required = {'prototypes', 'chars', 'char_to_index', 'thr', 'K'}
        missing = required - set(pack)
        if missing:
            raise ValueError(f'prototype pack missing keys: {sorted(missing)}')
        if tuple(pack['prototypes'].shape[1:]) != (int(pack['K']), 1024):
            raise ValueError(f'unexpected prototype shape: {tuple(pack["prototypes"].shape)}')
        self.prototypes = F.normalize(pack['prototypes'].float(), dim=-1).to(self.device)
        self.char_to_index = dict(pack['char_to_index'])
        self.threshold = float(pack['thr'])
        self.k = int(pack['K'])
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    @torch.no_grad()
    def score_crops(self, crops: Iterable[Image.Image], chars: Iterable[str]) -> list[dict]:
        crops, chars = list(crops), list(chars)
        if len(crops) != len(chars):
            raise ValueError('crops and chars must have equal length')
        results = [None] * len(crops)
        valid = [index for index, char in enumerate(chars) if char in self.char_to_index]
        for index, char in enumerate(chars):
            if char not in self.char_to_index:
                results[index] = {
                    'char': char,
                    'scored': False,
                    'reason': 'char_not_in_prototypes',
                    'error_score': None,
                    'threshold': self.threshold,
                    'pred_render_error': False,
                }
        if valid:
            tensors = torch.stack([self.transform(crops[index].convert('RGB')) for index in valid]).to(device=self.device, dtype=self.model_dtype)
            features = F.normalize(self.model.encode_image(tensors), dim=-1).float()
            if self.bfloat16_features:
                features = F.normalize(features, dim=-1)
            rows = torch.tensor([self.char_to_index[chars[index]] for index in valid], device=self.device)
            selected = self.prototypes[rows]
            errors = 1.0 - torch.einsum('nd,nkd->nk', features, selected).max(dim=1).values
            for index, error in zip(valid, errors.detach().cpu().tolist()):
                results[index] = {
                    'char': chars[index],
                    'scored': True,
                    'reason': None,
                    'error_score': float(error),
                    'threshold': self.threshold,
                    'pred_render_error': bool(error > self.threshold),
                }
        return results

    def score_image(self, image: str | Path, char: str) -> dict:
        with Image.open(image) as crop:
            return self.score_crops([crop.convert('RGB')], [char])[0]


def _resolve_image(path: str, image_root: Path | None) -> Path:
    candidate = Path(path)
    if image_root is not None and not candidate.is_absolute():
        candidate = image_root / candidate
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description='Run frozen TextVerifier inference on crop images.')
    parser.add_argument('--image', help='one cropped character image')
    parser.add_argument('--char', help='target Han character for --image')
    parser.add_argument('--input', type=Path, help='JSONL rows containing image and char fields')
    parser.add_argument('--image-root', type=Path, help='base directory for relative JSONL image paths')
    parser.add_argument('--output', type=Path, help='write JSON or JSONL predictions here')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--prototypes', type=Path, default=DEFAULT_PROTOTYPES)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    args = parser.parse_args()

    if bool(args.image) == bool(args.input):
        parser.error('provide exactly one of --image or --input')
    if args.image and not args.char:
        parser.error('--char is required with --image')

    verifier = TextVerifier(args.checkpoint, args.prototypes, args.device, args.dtype)
    if args.image:
        result = verifier.score_image(args.image, args.char)
        result['image'] = str(args.image)
        output = json.dumps(result, ensure_ascii=False, indent=2)
        print(output)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output + '\n', encoding='utf-8')
        return

    rows = [json.loads(line) for line in args.input.read_text(encoding='utf-8').splitlines() if line.strip()]
    crops, chars = [], []
    for row in rows:
        path = _resolve_image(row['image'], args.image_root)
        with Image.open(path) as image:
            crops.append(image.convert('RGB').copy())
        chars.append(row['char'])
    results = verifier.score_crops(crops, chars)
    for row, result in zip(rows, results):
        result['id'] = row.get('id', row.get('sample_id'))
        result['image'] = row['image']
    text = ''.join(json.dumps(result, ensure_ascii=False) + '\n' for result in results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
