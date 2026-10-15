#!/usr/bin/env bash
# Copy a finished run into the repository: examples/my_room (report, ledger, figures, maps), docs/ (GitHub
# Pages: the 3D viewer, the Rerun recording, the report, the annotated videos) and release/example_outputs.zip
# (everything, including the meshes). Local paths in run_info.json and keyframes.json become file names.
# usage: tools/package_repo.sh outputs/my_room [release_dir]
set -euo pipefail
RUN=$(cd "$1" && pwd); RES="$RUN/result"
REPO=$(cd "$(dirname "$0")/.." && pwd)
EX="$REPO/examples/my_room"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
python3 - "$RUN" "$TMP" <<'EOF'
import json, os, sys
run, tmp = sys.argv[1:3]
for name in ('run_info.json', 'keyframes.json'):
    p = os.path.join(run, name)
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    if name == 'run_info.json':
        d['inputs'] = {k: os.path.basename(v) for k, v in d.get('inputs', {}).items()}
    else:
        for ep in ('before', 'after'):
            if isinstance(d.get(ep), dict) and 'video' in d[ep]:
                d[ep]['video'] = os.path.basename(d[ep]['video'])
    json.dump(d, open(os.path.join(tmp, name), 'w'), indent=1)
EOF
rm -rf "$EX"; mkdir -p "$EX/data" "$EX/maps"
cp "$RES"/report.md "$RES"/ledger.json "$RES"/changes.json "$EX"/
cp -r "$RES"/images "$EX"/
cp "$RES"/maps/*.pgm "$RES"/maps/*.yaml "$RES"/maps/routes.json "$EX"/maps/
cp "$RES"/data/next_views.json "$RES"/data/descriptions.json "$EX"/data/
[ -f "$TMP/run_info.json" ] && cp "$TMP/run_info.json" "$EX"/
[ -d "$RUN/experiments" ] && cp "$RUN"/experiments/experiments.json "$RUN"/experiments/experiments.md "$EX"/ 2>/dev/null || true
mkdir -p "$REPO/docs/video"
cp "$RES"/viewer.html "$REPO"/docs/index.html
cp "$RES"/scene.rrd "$REPO"/docs/scene.rrd
cp "$RES"/report.html "$REPO"/docs/report.html
cp "$RES"/video/*.mp4 "$REPO"/docs/video/
touch "$REPO"/docs/.nojekyll
OUT=${2:-$REPO/../release}; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
rm -f "$OUT/example_outputs.zip"
(cd "$RUN" && zip -qr "$OUT/example_outputs.zip" result -x "result/meshes/surfaces.pkl")
(cd "$TMP" && zip -q "$OUT/example_outputs.zip" ./*.json)
du -sh "$EX" "$REPO/docs" "$OUT"/* 2>/dev/null
