#!/usr/bin/env bash
# Offline checks for the GPU Operator migration files (runs on the laptop, needs only Docker):
#   1. every key in values*.yaml exists in the chart's own values.yaml (the chart has no
#      values.schema.json, so helm would silently ignore a typo)
#   2. helm template renders with both value sets, and the ClusterPolicy carries our settings
#   3. kubeconform validates the plain manifests and the rendered chart
# Usage: bash scheduling/gpu-operator/validate.sh
set -euo pipefail
cd "$(dirname "$0")"
here=$(pwd -W 2>/dev/null || pwd)
export MSYS_NO_PATHCONV=1
CHART_VERSION=v26.7.2
HELM_IMAGE=alpine/helm:3.19.0
KUBECONFORM_IMAGE=ghcr.io/yannh/kubeconform:v0.7.0
out=.validate
mkdir -p "$out"

echo "== helm: pull chart $CHART_VERSION, render"
docker run --rm -v "$here":/w -w /w --entrypoint sh "$HELM_IMAGE" -c "
  set -e
  helm pull gpu-operator --repo https://helm.ngc.nvidia.com/nvidia --version $CHART_VERSION -d $out
  helm show values $out/gpu-operator-$CHART_VERSION.tgz > $out/chart-values.yaml
  helm template gpu-operator $out/gpu-operator-$CHART_VERSION.tgz -n gpu-operator --kube-version 1.36.5 \
    -f values.yaml > $out/render-default.yaml
  helm template gpu-operator $out/gpu-operator-$CHART_VERSION.tgz -n gpu-operator --kube-version 1.36.5 \
    -f values.yaml -f values-toolkit-k3s.yaml > $out/render-toolkit.yaml
  echo rendered: \$(grep -c '^kind:' $out/render-default.yaml) objects"

echo "== keys: every values key exists in the chart"
docker run --rm -v "$here":/w -w /w python:3.13-slim sh -c "
  pip install -q --disable-pip-version-check --root-user-action=ignore pyyaml==6.0.2 && python - <<'PY'
import sys, yaml
chart = yaml.safe_load(open('$out/chart-values.yaml'))
bad = 0
def walk(ours, theirs, path):
    global bad
    for k, v in ours.items():
        p = path + [k]
        if not isinstance(theirs, dict) or k not in theirs:
            print('  UNKNOWN KEY', '.'.join(p)); bad += 1; continue
        if isinstance(v, dict) and v:
            walk(v, theirs[k], p)
for f in ('values.yaml', 'values-toolkit-k3s.yaml'):
    before = bad
    walk(yaml.safe_load(open(f)), chart, [])
    print(f'  {f}: ' + ('ok' if bad == before else 'FAILED'))
cp = next(d for d in yaml.safe_load_all(open('$out/render-default.yaml')) if d and d.get('kind') == 'ClusterPolicy')['spec']
want = {('driver','enabled'): False, ('toolkit','enabled'): False, ('cdi','enabled'): False,
        ('migManager','enabled'): False, ('dcgmExporter','enabled'): False,
        ('devicePlugin','config','name'): 'time-slicing-config', ('devicePlugin','config','default'): 'any',
        ('operator','runtimeClass'): 'nvidia'}
for keys, v in want.items():
    got = cp
    for k in keys: got = got[k]
    ok = got == v; bad += not ok
    print(f\"  ClusterPolicy.{'.'.join(keys)} = {got!r}\", '' if ok else f'(want {v!r})')
sys.exit(1 if bad else 0)
PY"

echo "== kubeconform"
docker run --rm -v "$here/..":/w -w /w "$KUBECONFORM_IMAGE" -summary -strict \
  -kubernetes-version 1.33.0 -ignore-missing-schemas \
  gpu-operator/namespace.yaml gpu-operator/time-slicing-config.yaml gpu-operator/gpu-limit-policy-binding.yaml \
  gpu-operator/$out/render-default.yaml gpu-operator/$out/render-toolkit.yaml manifests/
