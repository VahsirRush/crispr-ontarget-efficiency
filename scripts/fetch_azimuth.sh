#!/usr/bin/env bash
# Fetch the Azimuth source tree (Doench et al. 2016 Rule Set 2 codebase + data).
#
# We take the tarball rather than `git clone` because we only need the data files
# and the reference implementation to read; there is no reason to carry the history.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -d _azimuth_src ]; then
  echo "_azimuth_src already present; nothing to do."
  exit 0
fi

echo "Downloading MicrosoftResearch/Azimuth ..."
curl -sL -o azimuth.tar.gz \
  https://github.com/MicrosoftResearch/Azimuth/archive/refs/heads/master.tar.gz
tar xzf azimuth.tar.gz
mv Azimuth-master _azimuth_src
rm azimuth.tar.gz

echo "Data files:"
ls -la _azimuth_src/azimuth/data/
