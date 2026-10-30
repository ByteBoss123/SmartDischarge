#!/usr/bin/env bash
# Fetch the UCI "Diabetes 130-US Hospitals for Years 1999-2008" dataset (Strack et al., 2014)
# and verify it against the SHA-256 of the file this project was built and scored on.
set -euo pipefail
cd "$(dirname "$0")"

EXPECTED_SHA256="da21942c25a10a51a8842907ab01a8530b06232616b17d317441d592a20c4359"
UCI_ZIP="https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
# Byte-identical public mirror, used only if UCI is unreachable. The checksum below is what makes it trustworthy.
MIRROR="https://raw.githubusercontent.com/andrewwlong/diabetes_readmission/master/diabetic_data.csv"

if [ ! -f diabetic_data.csv ]; then
  if curl -fsSL -o uci.zip "$UCI_ZIP"; then
    unzip -o -q uci.zip diabetic_data.csv* -d . || unzip -o -q uci.zip -d .
    find . -name 'diabetic_data.csv' -not -path './diabetic_data.csv' -exec mv {} . \; 2>/dev/null || true
    rm -f uci.zip
  else
    echo "UCI unreachable, using mirror"
    curl -fsSL -o diabetic_data.csv "$MIRROR"
  fi
fi

ACTUAL=$(sha256sum diabetic_data.csv | cut -d' ' -f1)
if [ "$ACTUAL" != "$EXPECTED_SHA256" ]; then
  echo "Checksum mismatch: $ACTUAL" >&2
  exit 1
fi
echo "OK: diabetic_data.csv ($(wc -l < diabetic_data.csv) lines) matches the expected SHA-256"
