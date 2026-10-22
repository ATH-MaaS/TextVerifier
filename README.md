# TextVerifier

TextVerifier is a target-conditioned verifier for structural Chinese glyph rendering errors. Given a glyph crop and its intended character, it detects missing, extra, broken, merged, split, distorted, or misplaced strokes and components while remaining insensitive to ordinary font, style, color, and background variation.

The released model uses a ViT-L/16 image encoder and an IDS-aware DeBERTa-v2 text encoder. At inference time, the crop feature is compared with ten learned visual prototypes for the requested character.

## Release components

The release is distributed across three repositories:

| Repository | Contents |
|---|---|
| This GitHub repository | inference/evaluation code, IDS resources, and the benchmark conversion script |
| Hugging Face model repository: `<YOUR_ORG>/TextVerifier-CJK-CLIP` | checkpoint, prototype embeddings, metadata, and checksum manifest |
| Hugging Face Dataset repository: `<YOUR_ORG>/TextRenderBench-ZH` | TextRenderBench-ZH in standard Parquet format |

Large model assets and benchmark data are intentionally not stored in GitHub. For reproducibility, download both Hugging Face repositories at immutable commit revisions.

## Installation

```bash
git clone <REPOSITORY_URL>
cd TextVerifier-public

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Select the PyTorch wheel matching your CUDA environment.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

CPU execution is suitable for functional checks. A CUDA GPU is recommended for full benchmark evaluation.

## 1. Download model assets

Download the following files from `<YOUR_ORG>/TextVerifier-CJK-CLIP` at the same immutable revision:

- `textverifier_stage3.pth`
- `textverifier_k10_q90.pt`
- `prototype_metadata.json`
- `manifest.sha256`

For example, with the Hugging Face CLI:

```bash
huggingface-cli download "<YOUR_ORG>/TextVerifier-CJK-CLIP" \
  textverifier_stage3.pth \
  textverifier_k10_q90.pt \
  prototype_metadata.json \
  manifest.sha256 \
  --revision "<MODEL_COMMIT_SHA>" \
  --local-dir assets
```

The files must be placed as follows:

```text
assets/
├── textverifier_stage3.pth
├── textverifier_k10_q90.pt
├── prototype_metadata.json
└── manifest.sha256
```

## 2. Download TextRenderBench-ZH

Download the complete Dataset repository `<YOUR_ORG>/TextRenderBench-ZH` at an immutable revision:

```bash
huggingface-cli download "<YOUR_ORG>/TextRenderBench-ZH" \
  --repo-type dataset \
  --revision "<DATASET_COMMIT_SHA>" \
  --local-dir downloads/TextRenderBench-ZH
```

Keep its Parquet files in their downloaded structure:

```text
downloads/TextRenderBench-ZH/
└── data/
    ├── images/test-00000-of-00001.parquet
    └── annotations/test-00000-of-00001.parquet
```

The Dataset repository contains two `test` configurations:

- `images`: 308 source images with embedded PNG bytes.
- `annotations`: 7,358 character-level annotations, including 642 rendering errors.

## 3. Convert the benchmark for evaluation

The Hugging Face Dataset uses standard Parquet files, while `textverifier.evaluate` reads local PNG files and JSONL annotations. Convert the downloaded Dataset with:

```bash
python scripts/materialize_benchmark.py \
  --dataset-dir downloads/TextRenderBench-ZH \
  --output benchmark
```

This creates:

```text
benchmark/
├── images/
│   ├── 000.png
│   └── ...
├── annotations.jsonl
└── stats.json
```

The converter preserves the embedded PNG bytes, reconstructs the evaluator annotation schema, validates 308 images, 7,358 annotations, and 642 rendering errors, and refuses to overwrite an existing output directory.

## 4. Validate the release

After downloading the model assets and materializing the benchmark, run:

```bash
python -m textverifier.verify_release --root . --check-checkpoint
```

This validates the model/prototype assets, checksums, benchmark structure, annotation schema, and aggregate statistics.

## 5. Run benchmark evaluation

```bash
python -m textverifier.evaluate \
  --benchmark benchmark \
  --checkpoint assets/textverifier_stage3.pth \
  --prototypes assets/textverifier_k10_q90.pt \
  --device cuda \
  --dtype bfloat16 \
  --batch-size 32 \
  --output outputs/benchmark_predictions.jsonl
```

The command writes:

```text
outputs/benchmark_predictions.jsonl
outputs/benchmark_predictions.metrics.json
```

The metrics include precision, recall, F1, false-positive rate, accuracy, and character coverage.

## Single-image inference

```bash
python -m textverifier.inference \
  --image /path/to/character_crop.png \
  --char 北 \
  --checkpoint assets/textverifier_stage3.pth \
  --prototypes assets/textverifier_k10_q90.pt \
  --device cuda \
  --dtype bfloat16
```

The output includes `error_score`, the active threshold, and `pred_render_error`.

## Batch inference

The input is JSONL. Each row must contain `image` and `char`; `id` is optional:

```json
{"id":"sample-001","image":"crops/001.png","char":"北"}
```

```bash
python -m textverifier.inference \
  --input examples/crops.jsonl \
  --image-root examples \
  --checkpoint assets/textverifier_stage3.pth \
  --prototypes assets/textverifier_k10_q90.pt \
  --output outputs/predictions.jsonl \
  --device cuda \
  --dtype bfloat16
```

## TextRenderBench-ZH

| Statistic | Value |
|---|---:|
| Source images | 308 |
| Character crops | 7,358 |
| Rendering errors | 642 |
| Correct renders | 6,716 |

After conversion, each annotation contains:

```json
{
  "sample_id":"000-0000",
  "image_id":"000",
  "image":"images/000.png",
  "item_index":0,
  "char":"北",
  "box":[[417.8,259.3],[513.3,259.3],[513.3,363.0],[417.8,363.0]],
  "rect":[417.8,259.3,513.3,363.0],
  "render_error":false
}
```

Evaluation crops an annotation rectangle with:

```python
crop = image.crop((int(x0), int(y0), int(round(x1)), int(round(y1))))
```

## Repository layout

The public GitHub repository contains only code, small IDS resources, documentation, and the conversion script:

```text
TextVerifier-public/
├── textverifier/
│   ├── data/
│   ├── evaluate.py
│   ├── ids_dictionary.py
│   ├── inference.py
│   ├── model.py
│   └── verify_release.py
├── scripts/
│   └── materialize_benchmark.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

Downloaded model assets, materialized benchmark data, Hugging Face Dataset exports, and evaluation outputs are excluded by `.gitignore`.

## Citation

A BibTeX entry will be added with the final paper release.

## License

See [LICENSE](LICENSE).