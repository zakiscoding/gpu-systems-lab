#!/usr/bin/env bash
# Copy raw results/ to a private R2 bucket. Run on the laptop, from anywhere in the repo,
# after enabling R2 and `wrangler login`. Existing objects are overwritten.
#   bash scripts/sync_results_r2.sh [--dry-run]
#   R2_BUCKET=other-name bash scripts/sync_results_r2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BUCKET=${R2_BUCKET:-gpu-systems-lab-results}
DRY_RUN=${1:-}

if [[ "$DRY_RUN" == "--dry-run" ]]; then
  find results -type f | sort
  echo "would upload $(find results -type f | wc -l) files to $BUCKET"
  exit 0
fi

cd gateway  # for its pinned wrangler
npx wrangler r2 bucket info "$BUCKET" >/dev/null 2>&1 || npx wrangler r2 bucket create "$BUCKET"
count=0
while IFS= read -r -d '' file; do
  npx wrangler r2 object put "$BUCKET/${file#../}" --file "$file" --remote >/dev/null
  count=$((count + 1))
  echo "$count: ${file#../}"
done < <(find ../results -type f -print0)
echo "uploaded $count files to $BUCKET"
