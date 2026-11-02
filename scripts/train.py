#!/usr/bin/env python3
"""Explicit, path-portable entry point for the archived training schedules."""

import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from biometaner.evaluation import TYPES, load_golden, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture', choices=['binary', 'joint'], default='binary')
    parser.add_argument('--model', choices=['bert', 'biobert', 'scibert', 'pubmedbert', 'biomegatron'], required=True)
    parser.add_argument('--experiment', choices=[f'E{i}' for i in range(1, 10)], required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--model-paths', type=Path, required=True, help='JSON mapping backbone keys to local model directories')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.architecture == 'joint' and (args.experiment not in ['E3', 'E6'] or args.model not in ['pubmedbert', 'biobert', 'scibert']):
        parser.error('Archived Joint implementation supports E3/E6 and PubMedBERT/BioBERT/SciBERT only')
    models = json.loads(args.model_paths.read_text())
    if args.model not in models or not Path(models[args.model]).is_dir():
        parser.error('Supply an existing local pretrained-model directory for the selected backbone')
    data = args.data_root.resolve()
    golden = data / 'golden_metasigndb_v2_for_eval.jsonl'
    load_golden(golden)
    inputs = [golden]
    if args.experiment not in ['E1', 'E2']:
        inputs.append(data / 'silver_new.jsonl')
    if args.experiment not in ['E3', 'E7']:
        inputs.extend(data / 'bronze_type_by_source_80_10_10_filtered_v1' / t / f'{split}.jsonl'
                      for t in TYPES for split in ['train', 'dev', 'test'])
    elif args.architecture == 'binary':
        # The original runner also evaluates the Bronze test split after fitting.
        inputs.extend(data / 'bronze_type_by_source_80_10_10_filtered_v1' / t / 'test.jsonl' for t in TYPES)
    hashes = {str(p.relative_to(data)): sha256(p) for p in inputs}
    os.environ['BIOMETANER_DATA_ROOT'] = str(data)
    os.environ['BIOMETANER_OUTPUT_ROOT'] = str(args.output_root.resolve())
    sys.path.insert(0, str(ROOT / 'training'))
    name = 'train_ner_binary' if args.architecture == 'binary' else 'train_ner_joint'
    module = importlib.import_module(name)
    module.MODEL_PATHS.update({k: str(Path(v).resolve()) for k, v in models.items()})
    recipe = module.EXPERIMENTS[args.experiment] if args.architecture == 'binary' else ('silver_full' if args.experiment == 'E3' else 'silver_full_bronze_lastlayer')
    prefix = '' if args.architecture == 'binary' else 'JOINT_'
    directory = args.output_root / f'{prefix}{args.experiment}_{args.model}_{recipe}_seed{args.seed}'
    if (directory / 'DONE').exists() and not args.force:
        print(f'Existing completed run left unchanged: {directory}')
        return
    directory.mkdir(parents=True, exist_ok=True)
    import torch
    import transformers
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    input_sha256=hashes, training_source_sha256=sha256(Path(module.__file__)),
                    python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
                    gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                    note='New-run provenance; does not reconstruct absent historical training-input hashes')
    (directory / 'run_input_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if args.architecture == 'binary':
        module.run_experiment(args.model, args.experiment, seed=args.seed, batch_size=args.batch_size, force=args.force)
    else:
        module.run_experiment_joint(args.model, args.experiment, seed=args.seed, batch_size=args.batch_size, force=args.force)


if __name__ == '__main__':
    main()
