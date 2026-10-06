#!/usr/bin/env python3
"""Real bounded CPU smoke, never a target-GPU validation or quality score."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'aimo-interpretability-2026-source-tgpPvy'
SOLUTION = SOURCE / 'solutions/uncertainty-profiling'
MODEL_ID = 'HuggingFaceTB/SmolLM2-135M-Instruct'
MODEL_REVISION = '12fd25f77366fa6b3b4b768ec3050bf629380bac'
ARTIFACT_SHA = 'bc90bd8eed4d9f6383cbb9ae1626896afea90b0e71c46d9a56bc87d376f85592'
PROMPTS = ['Compute 17 + 25. Give the integer answer.',
           'If three books cost 12 dollars, what does one book cost?']
RECEIPT = ROOT / 'native_cpu_smoke_r00_receipt.json'


def worker() -> dict:
    import numpy as np
    import pandas as pd
    import torch
    from huggingface_hub import snapshot_download

    sys.path.insert(0, str(SOLUTION))
    from uncertainty_profile.artifact import load_artifact
    from uncertainty_profile.config import DEEPSEEK_MODEL_ID, FEATURE_NAMES, GenerationConfidenceConfig
    from uncertainty_profile.extraction import extract_generation_features, load_model_and_tokenizer
    from uncertainty_profile.inference import predict_robustness
    import uncertainty_profile.extraction as extraction_runtime
    from trace_shape_r01 import summarize_trace

    owned_trace_rows = []
    original_collector = extraction_runtime.GenerationConfidenceCollector

    class ReadOnlyTemporalCollector(original_collector):
        def compute_features(self, **kwargs):
            metrics, tokens = super().compute_features(**kwargs)
            valid = extraction_runtime.build_valid_token_mask(
                tokens, pad_token_id=kwargs['pad_token_id'], eos_token_ids=kwargs['eos_token_ids']).tolist()
            entropies = torch.stack(self.entropy, dim=1).float().cpu().tolist()
            margins = torch.stack(self.top2_margins, dim=1).float().cpu().tolist()
            logprobs = torch.stack(self.selected_log_probs, dim=1).float().cpu().tolist()
            for signals in zip(entropies, margins, logprobs, valid):
                owned_trace_rows.append(summarize_trace(*signals))
            return metrics, tokens

    # Read-only trace hook in this smoke only; no source/package edits or altered logits.
    extraction_runtime.GenerationConfidenceCollector = ReadOnlyTemporalCollector

    started = time.monotonic()
    torch.set_num_threads(2)
    artifact_path = SOLUTION / 'uncertainty_artifacts/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B.joblib'
    assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == ARTIFACT_SHA
    artifact = load_artifact(DEEPSEEK_MODEL_ID)
    assert artifact is not None and artifact.model_id == DEEPSEEK_MODEL_ID
    assert artifact.generation_config.to_dict() == GenerationConfidenceConfig().to_dict()
    assert math.isclose(artifact.decision_threshold, 0.576, rel_tol=0, abs_tol=5e-5)
    versions = {name: importlib.metadata.version(name) for name in (
        'torch', 'transformers', 'scikit-learn', 'joblib', 'pandas', 'numpy', 'accelerate')}
    assert versions['scikit-learn'] == '1.8.0' and versions['joblib'] == '1.5.3'
    assert versions['transformers'] == '5.13.0'

    cache = ROOT / 'public-model-cache'
    # Known official model; no credentials or remote-code execution.
    snapshot = snapshot_download(MODEL_ID, revision=MODEL_REVISION, cache_dir=str(cache),
                                 allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', 'LICENSE'],
                                 token=False, max_workers=2)
    model_bytes = sum(p.stat().st_size for p in cache.rglob('*') if p.is_file() and not p.is_symlink())
    assert model_bytes <= 1024**3, 'public smoke model exceeds frozen1GiB budget'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    model, tokenizer = load_model_and_tokenizer(snapshot, local_files_only=True)
    assert next(model.parameters()).device.type == 'cpu'
    config = GenerationConfidenceConfig(max_prompt_length=128, max_new_tokens=24, batch_size=2)
    first = extract_generation_features(model=model, tokenizer=tokenizer,
                                       problem_texts=PROMPTS, config=config)
    second = extract_generation_features(model=model, tokenizer=tokenizer,
                                        problem_texts=PROMPTS, config=config)
    assert len(first) == len(second) == 2
    for row in first:
        assert tuple(key for key in row if key != 'generation_num_tokens') == FEATURE_NAMES
        assert row['generation_num_tokens'] > 0
        assert all(np.isfinite(float(row[key])) for key in FEATURE_NAMES)
    assert first == second, 'CPU greedy repeated features are not exactly deterministic'
    assert len(owned_trace_rows) == 4 and owned_trace_rows[:2] == owned_trace_rows[2:]
    assert all(len(row) == 13 and all(np.isfinite(float(value)) for value in row.values())
               for row in owned_trace_rows)
    scores = np.asarray(artifact.estimator.predict(pd.DataFrame(first, columns=FEATURE_NAMES)), dtype=float)
    assert scores.shape == (2,) and np.isfinite(scores).all()
    booleans = [bool(x < artifact.decision_threshold) for x in scores]
    assert all(type(x) is bool for x in booleans)
    # Real entry-point early returns: no unrelated model loaded or test labels.
    assert predict_robustness(DEEPSEEK_MODEL_ID, []) == []
    for unsupported in ('Qwen/Qwen3.5-4B', 'Skywork/Skywork-OR1-Math-7B', 'allenai/Olmo-3-7B-Think'):
        assert predict_robustness(unsupported, PROMPTS) == [False, False]
    return {
        'scope': 'REAL_CPU_GENERATION_COMPATIBILITY_NOT_TARGET_MODEL_GPU_OR_QUALITY',
        'status': 'PASS', 'versions': versions, 'smoke_model_id': MODEL_ID,
        'smoke_model_revision': MODEL_REVISION, 'public_model_cache_bytes': model_bytes,
        'feature_rows': 2, 'feature_count': 14, 'generation_tokens_cap': 24,
        'repeat_features_exact': True, 'bundled_artifact_schema_loaded': True,
        'bundled_artifact_sha256': ARTIFACT_SHA,
        'artifact_library_versions': artifact.library_versions,
        'artifact_generation_config': artifact.generation_config.to_dict(),
        'artifact_actual_decision_threshold': artifact.decision_threshold,
        'artifact_training_provenance': artifact.training_provenance,
        'unsupported_models_fixed_false_verified': 3,
        'native_bool_outputs': True, 'wall_seconds': time.monotonic() - started,
        'owned_temporal_features': {'module':'trace_shape_r01.py', 'rows':4, 'features':13,
                                    'finite':True, 'repeat_exact':True, 'no_trained_predictor_or_quality':True},
        'owned_temporal_source_sha256': hashlib.sha256((ROOT / 'trace_shape_r01.py').read_bytes()).hexdigest(),
        'target_8B_model_validated': False, 'CUDA_validated': False,
        'official_Docker_validated': False, 'official_score': None,
        'CPU_smoke_labels_or_accuracy': None, 'upstream_baseline_not_original_method': True,
    }


def main() -> None:
    if '--worker' in sys.argv:
        result = worker()
        print(json.dumps(result, sort_keys=True))
        return
    assert shutil.disk_usage(ROOT).free >= 15 * 1024**3, 'stop: free disk under15GiB'
    started = time.monotonic()
    try:
        result = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--worker'],
                                text=True, capture_output=True, timeout=360, check=False)
        if result.returncode == 0:
            receipt = json.loads(result.stdout.strip().splitlines()[-1])
        else:
            receipt = {'status': 'FAIL', 'exit_code': result.returncode,
                       'stderr_tail': result.stderr[-6000:], 'quality_unknown': True}
    except subprocess.TimeoutExpired:
        receipt = {'status': 'FAIL_RESOURCE_TIMEOUT', 'timeout_seconds': 360, 'quality_unknown': True}
    receipt['parent_wall_seconds'] = time.monotonic() - started
    receipt['worker_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': receipt['status'], 'receipt': str(RECEIPT)}, sort_keys=True))
    if receipt['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
