#!/usr/bin/env python3
"""One bounded native CPU validation; no target-GPU or quality claims."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'aimo-interpretability-2026-source-tgpPvy'
COMMIT = 'de794053debe75a711400696e66b60937cebbb1b'
CPU_WHEEL = ('torch @ https://download-r2.pytorch.org/whl/cpu/'
             'torch-2.12.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl'
             '#sha256=ae4bb28409f5370852bd71af221066236c38d647f780d9b0a7240c330a9c12df')
DEPENDENCIES = [CPU_WHEEL, 'accelerate==1.13.0', 'huggingface-hub==1.22.0',
                'joblib==1.5.3', 'pandas==3.0.3', 'safetensors==0.8.0',
                'scikit-learn==1.8.0', 'tokenizers==0.22.2',
                'transformers==5.13.0', 'numpy==2.5.1']


def main():
    start = time.monotonic()
    stages = []
    sources = subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    hashes = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}

    def run(label, argv, seconds, cwd=ROOT):
        remaining = 1080-(time.monotonic()-start)
        if remaining <= 0:
            raise RuntimeError('parent1080s wall consumed')
        began = time.monotonic()
        try:
            result = subprocess.run(argv,cwd=cwd,text=True,capture_output=True,
                                    timeout=min(seconds,remaining),check=False)
            stages.append({'stage':label,'exit_code':result.returncode,
                           'wall_seconds':time.monotonic()-began,
                           'stdout_tail':result.stdout[-20000:], 'stderr_tail':result.stderr[-12000:]})
        except subprocess.TimeoutExpired:
            stages.append({'stage':label,'exit_code':124,'wall_seconds':time.monotonic()-began})
            raise RuntimeError(label+' resource timeout')
        if result.returncode:
            raise RuntimeError(label+' failed')

    status = 'FAIL'
    failure = None
    try:
        assert not SOURCE.exists(), 'do not reuse a previous execution clone'
        run('official_sparse_clone', ['git','clone','--filter=blob:none','--no-checkout','--depth','1',
                                     'https://github.com/aimo-interp/getting-started.git',str(SOURCE)],90)
        run('official_sparse_scope', ['git','sparse-checkout','set','solutions/uncertainty-profiling',
                                     'solutions/always-true','components','scripts','tests'],30,SOURCE)
        run('official_source_checkout',['git','checkout',COMMIT],60,SOURCE)
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip() == COMMIT
        run('pinned_native_CPU_dependencies',[sys.executable,'-m','pip','install','--no-cache-dir',
                                            '--timeout','120','--retries','1',*DEPENDENCIES],600)
        run('upstream_all_unit_contract_tests',[sys.executable,'-B','-m','unittest','discover','-s','tests','-v'],120,SOURCE)
        run('original_temporal_controls',[sys.executable,'-B','-m','unittest','test_trace_shape_r01.py','-v'],30)
        run('unchanged_Small_build',[sys.executable,'-B','scripts/build.py','solutions','--small'],30,SOURCE)
        run('unchanged_Main_build',[sys.executable,'-B','scripts/build.py','solutions'],30,SOURCE)
        run('exact_package_audit',[sys.executable,'-B','check_upstream_package_r00.py'],30)
        run('real_CPU_model_smoke',[sys.executable,'-B','native_cpu_smoke_r00.py'],370)
        assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value for name,value in hashes.items())
        status = 'PASS'
    except Exception as exc:
        failure = str(exc)
    report = {'scope':'ACTUAL_LINUX_CPU_PREFLIGHT_NOT_8B_CUDA_DOCKER_OR_QUALITY',
              'status':status,'failure':failure,'wall_seconds':time.monotonic()-start,
              'source_hashes':hashes,'official_source_commit':COMMIT,'stages':stages,
              'native_receipt':json.loads((ROOT/'native_cpu_smoke_r00_receipt.json').read_text())
                               if (ROOT/'native_cpu_smoke_r00_receipt.json').exists() else None,
              'package_audit':json.loads((ROOT/'package_audit_r00.json').read_text())
                              if (ROOT/'package_audit_r00.json').exists() else None,
              'official_score':None,'award':None,'new_service_spend_or_commit_cny':0}
    print('AIMO_NATIVE_R00_RECEIPT '+json.dumps(report,sort_keys=True))
    if status != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
