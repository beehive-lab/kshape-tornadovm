#!/bin/bash
# Benchmarks the TornadoVM k-Shape against the upstream Python CPU (NumPy) and GPU (PyTorch) versions.
#
# Needs: TORNADOVM_HOME and JAVA_HOME set, PY pointing at a Python with numpy (and torch with CUDA for the GPU
# baseline), KSHAPE_SRC pointing at a checkout of https://github.com/TheDatumOrg/kshape-python.
#
#   scripts/bench.sh OUTDIR [config ...]
#
# A config is name:kind:m:L:D:k:max_iter:repeat (repeat applies to the TornadoVM runs). Without configs, the default set below runs.
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:?usage: bench.sh OUTDIR [config ...]}; shift
mkdir -p "$OUT"
PY=${PY:-python3}
SKIP_CPU=${SKIP_CPU:-}
SKIP_GPU=${SKIP_GPU:-}
# the Python references are slow: first run plus one warm run by default
REF_REPEAT=${REF_REPEAT:-2}
CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
  CONFIGS=(
    "ucr-small:shapes:1000:256:1:8:100:5"
    "ucr-large:shapes:5000:512:1:10:100:5"
    "multi:shapes:2000:1000:3:5:100:5"
    "notebook:uniform:500:5000:6:3:5:3"
  )
fi

(cd "$HERE" && ./mvnw -q -B compile -Dtornadovm.version="${TVM_VERSION:-8.1.0-jdk22plus}") || exit 1

for cfg in "${CONFIGS[@]}"; do
  IFS=: read -r name kind m L D k iters rep <<< "$cfg"
  d=$OUT/$name
  echo "== $name: $kind m=$m L=$L D=$D k=$k max_iter=$iters repeat=$rep"
  [ -f "$d.bin" ] || $PY "$HERE/python/gen_data.py" --kind "$kind" --m "$m" --L "$L" --D "$D" --k "$k" --out "$d"
  java @"$TORNADOVM_HOME/tornado-argfile" -Dtornado.device.memory=40GB -cp "$HERE/target/classes" org.beehive.kshape.KShapeBench \
    --data "$d.bin" --init "$d.init" --k "$k" --max-iter "$iters" --repeat "$rep" --out "$d.tvm" 2>&1 | tee "$d.tvm.log" | grep '\[kshape\]'
  if [ -z "$SKIP_GPU" ]; then
    $PY "$HERE/python/run_reference.py" --kshape-src "$KSHAPE_SRC" --backend gpu --data "$d.bin" --init "$d.init" --k "$k" \
      --max-iter "$iters" --repeat "$REF_REPEAT" --out "$d.gpu" 2>&1 | tee "$d.gpu.log" | grep '\[ref'
    $PY "$HERE/python/compare.py" --a "$d.gpu" --b "$d.tvm" --k "$k" --L "$L" --D "$D" --truth "$d.truth" | tee "$d.cmp-gpu.txt"
  fi
  if [ -z "$SKIP_CPU" ]; then
    $PY "$HERE/python/run_reference.py" --kshape-src "$KSHAPE_SRC" --backend cpu --data "$d.bin" --init "$d.init" --k "$k" \
      --max-iter "$iters" --repeat "$REF_REPEAT" --out "$d.cpu" 2>&1 | tee "$d.cpu.log" | grep '\[ref'
    $PY "$HERE/python/compare.py" --a "$d.cpu" --b "$d.tvm" --k "$k" --L "$L" --D "$D" --truth "$d.truth" | tee "$d.cmp-cpu.txt"
  fi
done
