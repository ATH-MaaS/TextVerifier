from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .inference import DEFAULT_CHECKPOINT, DEFAULT_PROTOTYPES, TextVerifier

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = PACKAGE_ROOT / 'benchmark'


def crop_for_row(image: Image.Image, rect: list[float]) -> Image.Image:
    x0, y0, x1, y1 = rect
    crop = image.crop((int(x0), int(y0), int(round(x1)), int(round(y1))))
    if crop.width < 2 or crop.height < 2:
        raise ValueError(f'invalid crop dimensions: {crop.size}')
    return crop.convert('RGB').copy()


def metrics(rows: list[dict]) -> dict:
    tp = fp = fn = tn = 0
    scored = 0
    unsupported = 0
    for row in rows:
        gold = bool(row['render_error'])
        prediction = bool(row['pred_render_error'])
        if row['scored']:
            scored += 1
        else:
            unsupported += 1
        if gold and prediction:
            tp += 1
        elif not gold and prediction:
            fp += 1
        elif gold:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        'N': len(rows),
        'positives': sum(bool(row['render_error']) for row in rows),
        'negatives': sum(not bool(row['render_error']) for row in rows),
        'scored': scored,
        'unsupported_characters_predicted_normal': unsupported,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
        'P': precision,
        'R': recall,
        'F1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        'FPR': fp / (fp + tn) if fp + tn else 0.0,
        'ACC': (tp + tn) / len(rows) if rows else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate frozen TextVerifier on the released benchmark.')
    parser.add_argument('--benchmark', type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--prototypes', type=Path, default=DEFAULT_PROTOTYPES)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--output', type=Path, default=PACKAGE_ROOT / 'results' / 'predictions.jsonl')
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    annotation_path = args.benchmark / 'annotations.jsonl'
    rows = [json.loads(line) for line in annotation_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    verifier = TextVerifier(args.checkpoint, args.prototypes, args.device, args.dtype)

    predictions: list[dict] = []
    batch_crops = []
    batch_rows = []

    def flush() -> None:
        if not batch_rows:
            return
        for row, result in zip(batch_rows, verifier.score_crops(batch_crops, [item['char'] for item in batch_rows])):
            predictions.append({
                'sample_id': row['sample_id'],
                'image_id': row['image_id'],
                'char': row['char'],
                'render_error': bool(row['render_error']),
                **result,
            })
        batch_crops.clear()
        batch_rows.clear()

    current_path = None
    current_image = None
    for index, row in enumerate(rows):
        path = args.benchmark / row['image']
        if path != current_path:
            if current_image is not None:
                current_image.close()
            current_path = path
            current_image = Image.open(path).convert('RGB')
        batch_crops.append(crop_for_row(current_image, row['rect']))
        batch_rows.append(row)
        if len(batch_rows) == args.batch_size:
            flush()
        if (index + 1) % 1000 == 0:
            print(f'prepared {index + 1}/{len(rows)} rows', flush=True)
    flush()
    if current_image is not None:
        current_image.close()

    report = metrics(predictions)
    benchmark_chars = {row['char'] for row in rows}
    covered_chars = benchmark_chars & set(verifier.char_to_index)
    report.update({
        'checkpoint': str(args.checkpoint.name),
        'prototypes': str(args.prototypes.name),
        'dtype': args.dtype,
        'threshold': verifier.threshold,
        'K': verifier.k,
        'prototype_characters': len(verifier.char_to_index),
        'benchmark_unique_characters': len(benchmark_chars),
        'covered_benchmark_characters': len(covered_chars),
        'uncovered_benchmark_characters': len(benchmark_chars - covered_chars),
        'decision_rule': 'pred_render_error = error_score > threshold',
        'crop_rule': 'int(x0), int(y0), int(round(x1)), int(round(y1))',
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    metrics_path = args.output.with_suffix('.metrics.json')
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'wrote {args.output} and {metrics_path}')


if __name__ == '__main__':
    main()
