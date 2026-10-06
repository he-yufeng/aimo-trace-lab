#!/usr/bin/env python3
"""Verify literal upstream package bytes without importing pickle or models."""
import ast
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'aimo-interpretability-2026-source-tgpPvy'
COMMIT = 'de794053debe75a711400696e66b60937cebbb1b'
EXPECTED = {
    'solution-uncertainty-profiling-small.zip': '95965eea1d189482e39d3705a2ff6596a7ad5b341ef44dd4826b91070c8445a7',
    'solution-uncertainty-profiling.zip': 'b9f9bfedd930638fc8e4b5ad65eb70d9066b898c9e35b8c796296c64b42ab34f',
}


def main():
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip()
    assert revision == COMMIT
    assert subprocess.check_output(['git', 'diff', '--exit-code', 'HEAD', '--', 'solutions/uncertainty-profiling'], cwd=SOURCE) == b''
    results = []
    for filename, expected in EXPECTED.items():
        path = SOURCE / 'dist' / filename
        archive_bytes = path.read_bytes()
        assert hashlib.sha256(archive_bytes).hexdigest() == expected
        hashes = {}
        with ZipFile(path) as bundle:
            assert bundle.testzip() is None
            names = bundle.namelist()
            assert len(names) == len(set(names)) and 'solution.py' in names
            assert ('small.txt' in names) == filename.endswith('-small.zip')
            for name in names:
                relative = PurePosixPath(name)
                assert not relative.is_absolute() and '..' not in relative.parts
                assert not any(part in ('.venv', '__pycache__', 'data', 'node_modules', '.git') for part in relative.parts)
                assert not name.endswith(('.pyc', '.safetensors', '.csv', '.parquet', '.env'))
                assert not any(x in name.lower() for x in ('password', 'credential', 'api_key', 'token.json'))
                contents = bundle.read(name)
                if name == 'small.txt':
                    assert contents == b''
                else:
                    source_path = SOURCE / 'solutions/uncertainty-profiling' / name
                    assert source_path.read_bytes() == contents
                if name.endswith('.py'):
                    ast.parse(contents.decode('utf-8'), filename=name)
                hashes[name] = hashlib.sha256(contents).hexdigest()
            module = ast.parse(bundle.read('solution.py'))
            entry = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'are_robust')
            assert [arg.arg for arg in entry.args.args] == ['model_id', 'reasoning_effort', 'problems']
        results.append({'filename':filename, 'sha256':expected, 'bytes':len(archive_bytes), 'entries':hashes})
    result = {'scope':'EXACT_UPSTREAM_BYTE_AST_TRACK_AUDIT_NOT_GPU_MODEL_VALIDATION',
              'status':'PASS', 'source_commit':COMMIT, 'archives':results,
              'official_score':None, 'upstream_baseline_not_original_method':True}
    (ROOT / 'package_audit_r00.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'status':'PASS', 'archives':[{k:v for k,v in x.items() if k!='entries'} for x in results]}, sort_keys=True))


if __name__ == '__main__':
    main()
