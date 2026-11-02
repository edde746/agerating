#!/bin/bash
# Weekly job: add new Wikidata films / TV shows to the full database and publish both databases as a GitHub release.
#
# 1. data/agerating.db is the working copy. When it is missing, the latest release's agerating.db.zst is downloaded.
# 2. update.py adds the items that appeared on Wikidata since the last run.
# 3. If the database changed since the last published release, the small database is rebuilt and both databases are
#    published as release vYYYY.MM.DD (zstd, plus a gzip copy of the small database and its packed list, pack_ids.py),
#    with SHA256SUMS. The database's
#    version is info.updated_at (set by update.py when it adds titles), else info.built_at (a fresh build);
#    data/published holds the last published version and is written only after the release exists, so a failed
#    upload is retried on the next run.
# The exit status is update.py's (1 if some chunks failed); whatever was added is published either way.
#
# Env: AGERATING_REPO        GitHub repository for the releases (default edde746/agerating)
#      AGERATING_USER_AGENT  contact for the Wikimedia APIs, e.g. "agerating/1.0 (https://github.com/edde746/agerating)"
#      AGERATING_MINUTES     time budget for update.py (default 240)
#      PYTHON                default .venv/bin/python
# Usage: ./weekly.sh   (studio.local runs it every Monday through launchd; see README.md)
set -euo pipefail
cd "$(dirname "$0")"
REPO=${AGERATING_REPO:-edde746/agerating}
PY=${PYTHON:-.venv/bin/python}
DB=data/agerating.db
OUT=data/release
log() { echo "$(date -u +%FT%TZ) $*"; }
version() { sqlite3 "$DB" "SELECT coalesce((SELECT value FROM info WHERE key = 'updated_at'),
                                           (SELECT value FROM info WHERE key = 'built_at'))"; }

mkdir -p data
if [ ! -f "$DB" ]; then
  log "$DB missing: downloading the latest release of $REPO"
  rm -rf data/download && mkdir data/download
  gh release download --repo "$REPO" --pattern agerating.db.zst --pattern SHA256SUMS --dir data/download
  (cd data/download && grep ' agerating.db.zst$' SHA256SUMS | shasum -a 256 -c -)
  zstd -d -q data/download/agerating.db.zst -o "$DB"
  rm -rf data/download
  version > data/published
fi

log "update started"
status=0
"$PY" -u update.py --db "$DB" --minutes "${AGERATING_MINUTES:-240}" || status=$?

current=$(version)
published=$(cat data/published 2>/dev/null || true)
if [ "$current" = "$published" ]; then
  log "no titles added since the last release"
  exit $status
fi

"$PY" build_ids_db.py "$DB" data/agerating-ids.db
"$PY" pack_ids.py data/agerating-ids.db data/agerating-ids.bin --check
rm -rf "$OUT" && mkdir "$OUT"
zstd -19 -T0 -q "$DB" -o "$OUT/agerating.db.zst"
zstd -19 -T0 -q data/agerating-ids.db -o "$OUT/agerating-ids.db.zst"
gzip -9 -n -c data/agerating-ids.db > "$OUT/agerating-ids.db.gz"
gzip -9 -n -c data/agerating-ids.bin > "$OUT/agerating-ids.bin.gz"
(cd "$OUT" && shasum -a 256 agerating-ids.db.zst agerating-ids.db.gz agerating-ids.bin.gz agerating.db.zst > SHA256SUMS)

ids=data/agerating-ids.db
n() { printf "%'d" "$(sqlite3 "$1" "$2")"; }
since="scored_at > (SELECT value FROM info WHERE key = 'built_at') AND scored_at > '$published'"
notes="Wikidata as of $current.

- $(n "$DB" "SELECT COUNT(*) FROM title") films and TV shows, $(n "$DB" "SELECT COUNT(*) FROM title WHERE $since") added since the previous release; $(n "$DB" "SELECT COUNT(*) FROM title WHERE fsk_official IS NOT NULL") with an official FSK rating.
- agerating-ids.db: $(n $ids "SELECT COUNT(*) FROM imdb") IMDb, $(n $ids "SELECT COUNT(*) FROM tmdb_movie") TMDB movie, $(n $ids "SELECT COUNT(*) FROM tmdb_tv") TMDB TV, $(n $ids "SELECT COUNT(*) FROM tvdb_series") TheTVDB series and $(n $ids "SELECT COUNT(*) FROM tvdb_movie") TheTVDB movie IDs.
- Estimates by $(sqlite3 "$DB" "SELECT value FROM info WHERE key = 'model'") (https://huggingface.co/edde746/$(sqlite3 "$DB" "SELECT value FROM info WHERE key = 'model'")).

Ages are suggestions on the FSK scale (0, 6, 12, 16, 18): the official FSK rating where Wikidata has one, otherwise a model estimate. Estimates are not official ratings."
tag=v$(date -u +%Y.%m.%d)
if gh release view "$tag" --repo "$REPO" >/dev/null 2>&1; then tag=$tag.$(date -u +%H%M); fi
gh release create "$tag" "$OUT"/* --repo "$REPO" --title "$tag" --notes "$notes"
echo "$current" > data/published
log "published $tag"
exit $status
