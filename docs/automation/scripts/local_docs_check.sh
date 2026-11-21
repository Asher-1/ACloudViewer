#!/usr/bin/env bash
#
# Execute and validate the documentation locally, the way CI does.
#
# WHY
# ---
# CI costs ~40-70 minutes per round. Anything that can be answered locally
# should never be pushed to find out. The blocker used to be "the cloudViewer
# module only exists in CI" -- that is only true if nobody has built it. With
# an existing build tree, every notebook can be executed here.
#
# USAGE
#   bash docs/automation/scripts/local_docs_check.sh            # all notebooks
#   bash docs/automation/scripts/local_docs_check.sh <name>     # one notebook
#
# Requires:
#   - a built python package, e.g. build_app/lib/python_package
#   - jupyter kernel:  python3 -m pip install ipykernel && python3 -m ipykernel install --user --name python3
#
# Exit codes: 0 everything executed cleanly, 1 at least one notebook failed.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Resolve the built package without guessing a single location.
PY_PKG=""
for candidate in build_app/lib/python_package build/lib/python_package; do
    if [ -d "$candidate/cloudViewer" ]; then
        PY_PKG="$(cd "$candidate" && pwd)"
        break
    fi
done
if [ -z "$PY_PKG" ]; then
    echo "❌ No built cloudViewer package found." >&2
    echo "   Looked in: build_app/lib/python_package, build/lib/python_package" >&2
    echo "   Build one first:  cmake --build build_app --target python-package" >&2
    exit 1
fi

export PYTHONPATH="$PY_PKG:$REPO_ROOT/docs/jupyter"
export CI=true              # disables interactive windows, as in CI
export MPLBACKEND=Agg

if ! python3 -c 'import cloudViewer' 2>/dev/null; then
    echo "❌ cloudViewer is on PYTHONPATH ($PY_PKG) but does not import." >&2
    echo "   Common cause: a missing optional dependency (plotly, dash)." >&2
    python3 -c 'import cloudViewer' 2>&1 | tail -5 >&2
    exit 1
fi
echo "✅ cloudViewer imports from $PY_PKG"

if ! python3 -m jupyter kernelspec list 2>/dev/null | grep -q python3; then
    echo "❌ No Jupyter kernel registered." >&2
    echo "   python3 -m pip install ipykernel && python3 -m ipykernel install --user --name python3" >&2
    exit 1
fi

# Notebooks CI deliberately does not execute (see make_docs.py nb_direct_copy).
SKIP="draw_plotly.ipynb hashmap.ipynb 3d_gaussian_splatting.ipynb jupyter_visualization.ipynb t_icp_registration.ipynb tensor.ipynb"

targets=()
if [ "$#" -gt 0 ]; then
    while IFS= read -r f; do targets+=("$f"); done < <(find docs/jupyter -name "$1" -o -name "$1.ipynb" 2>/dev/null)
else
    while IFS= read -r f; do
        base="$(basename "$f")"
        case " $SKIP " in *" $base "*) continue ;; esac
        targets+=("$f")
    done < <(find docs/jupyter -name '*.ipynb' | sort)
fi

if [ "${#targets[@]}" -eq 0 ]; then
    echo "❌ No notebooks matched." >&2
    exit 1
fi

echo "Executing ${#targets[@]} notebook(s)..."
failed=()
for nb in "${targets[@]}"; do
    name="$(basename "$nb")"
    out="/tmp/local_nb_${name%.ipynb}.out.ipynb"
    log="/tmp/local_nb_${name%.ipynb}.log"
    printf '  %-42s ' "$name"
    ( cd "$(dirname "$nb")" && \
      timeout 1800 python3 -m nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=1500 --output "$out" "$(basename "$nb")" ) \
      > "$log" 2>&1
    rc=$?

    errs=0
    if [ -f "$out" ]; then
        errs=$(python3 - "$out" <<'PY'
import json, sys
nb = json.load(open(sys.argv[1]))
print(sum(1 for c in nb['cells'] for o in c.get('outputs', [])
          if o.get('output_type') == 'error'))
PY
)
    fi

    if [ "$rc" -ne 0 ] || [ "$errs" -gt 0 ]; then
        echo "FAIL (rc=$rc, errors=$errs) → $log"
        failed+=("$name")
    else
        echo "ok"
    fi
done

echo
if [ "${#failed[@]}" -gt 0 ]; then
    echo "❌ ${#failed[@]} notebook(s) failed: ${failed[*]}"
    echo "   Logs: /tmp/local_nb_*.log"
    exit 1
fi
echo "✅ All ${#targets[@]} notebook(s) executed cleanly"
