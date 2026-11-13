#!/usr/bin/env bash
# Refresh indexes and install as one transaction: a mirror may rotate between
# update and install. Never reuse the failed transaction's partial indexes.
set -euo pipefail

apt_options=(-o Acquire::Retries=3 -o Acquire::http::Timeout=60
             -o Acquire::https::Timeout=60)
for attempt in 1 2 3 4 5; do
    if apt-get "${apt_options[@]}" -o APT::Update::Error-Mode=any update \
        && apt-get "${apt_options[@]}" install -y "$@"; then
        exit 0
    fi
    if [ "$attempt" -eq 5 ]; then
        echo "APT update/install failed after ${attempt} attempts" >&2
        exit 1
    fi
    echo "APT update/install failed (attempt ${attempt}/5); refreshing indexes" >&2
    rm -rf /var/lib/apt/lists/*
    sleep "$((attempt * 15))"
done
