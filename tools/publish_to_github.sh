#!/usr/bin/env bash
# Publish this repository to GitHub in one go (run it from the repository folder, on your own computer):
#   1. creates the public repo <you>/deltanav and pushes the code,
#   2. fills your GitHub username into the README links,
#   3. turns on GitHub Pages (the 3D viewer, the report, the Rerun recording),
#   4. makes the v1.0 release with the example videos (used by `deltanav demo`).
# Needs the GitHub CLI, logged in once:  gh auth login
# Usage:  bash tools/publish_to_github.sh /path/to/folder/with/before.mp4,after.mp4,example_outputs.zip
set -euo pipefail
ASSETS=${1:-..}
NAME=deltanav
command -v gh >/dev/null || { echo "Install the GitHub CLI first (see PUBLISH_STEPS.md), then run: gh auth login"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "Log in first: gh auth login"; exit 1; }
USER=$(gh api user -q .login)
echo "GitHub user: $USER"
# 1. your username in the links
grep -rlI ANSPER-GITHUB-USERNAME . --exclude-dir=.git --exclude=publish_to_github.sh | xargs -r sed -i.bak "s/ANSPER-GITHUB-USERNAME/$USER/g"
find . -name '*.bak' -not -path './.git/*' -delete
# 2. git repository and first commit
if [ ! -d .git ]; then git init -b main; fi
git add -A
git -c user.name="${GIT_NAME:-$(gh api user -q .name 2>/dev/null || echo "$USER")}" -c user.email="${GIT_EMAIL:-$USER@users.noreply.github.com}" \
    commit -m "What changed in the room: visibility-aware 3D change detection from two phone videos" || true
# 3. create the repo on GitHub and push
if gh repo view "$USER/$NAME" >/dev/null 2>&1; then
  git remote get-url origin >/dev/null 2>&1 || git remote add origin "https://github.com/$USER/$NAME.git"
  git push -u origin main
else
  gh repo create "$NAME" --public --source=. --remote=origin --push \
     --description "DeltaNav: what changed in the room, verified by line of sight, turned into robot map and route updates. Two phone videos in."
fi
# 4. GitHub Pages from /docs
gh api -X POST "repos/$USER/$NAME/pages" -f "source[branch]=main" -f "source[path]=/docs" >/dev/null 2>&1 \
  || gh api -X PUT "repos/$USER/$NAME/pages" -f "source[branch]=main" -f "source[path]=/docs" >/dev/null 2>&1 || true
# 5. release with the example videos and outputs
FILES=()
for f in before.mp4 after.mp4 example_outputs.zip; do [ -f "$ASSETS/$f" ] && FILES+=("$ASSETS/$f"); done
if [ ${#FILES[@]} -gt 0 ]; then
  gh release view v1.0 >/dev/null 2>&1 && gh release upload v1.0 "${FILES[@]}" --clobber \
    || gh release create v1.0 "${FILES[@]}" --title "Example: my bedroom" --notes "The two example videos (re-encoded to 1280 px) used by \`deltanav demo\`, and every output of the example run."
fi
echo
echo "Repository:  https://github.com/$USER/$NAME"
echo "3D viewer:   https://$USER.github.io/$NAME/            (Pages takes a minute or two the first time)"
echo "Report:      https://$USER.github.io/$NAME/report.html"
