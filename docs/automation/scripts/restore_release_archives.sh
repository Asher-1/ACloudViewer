#!/usr/bin/env bash
# Restore only known, valid documentation trees from published GitHub assets.
# Usage: restore_release_archives.sh <destination> <restore-dev:true|false> [current-release]
set -euo pipefail

usage() { echo "Usage: $0 <destination> <restore-dev:true|false> [current-release]" >&2; exit 2; }
[ "$#" -ge 2 ] && [ "$#" -le 3 ] || usage

destination="$1"
restore_dev="$2"
current_release="${3:-}"
[[ "$restore_dev" == true || "$restore_dev" == false ]] || usage
[[ -z "$current_release" || "$current_release" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || {
    echo "Refusing invalid current release tag: $current_release" >&2
    exit 1
}

mkdir -p "$destination"
destination="$(realpath "$destination")"
work="$(mktemp -d "${destination}/.restore.XXXXXX")"
cleanup() {
    local resolved
    resolved="$(realpath -m -- "$work")"
    [[ "$resolved" == "$destination"/.restore.* && "$resolved" != "$destination" ]] \
        || { echo "Refusing unexpected cleanup path: $resolved" >&2; return 1; }
    rm -rf -- "$resolved"
}
trap cleanup EXIT

die() { echo "❌ $*" >&2; exit 1; }

# One authoritative inventory. Failure is not an empty release list: on a Pages
# replacement deploy that mistake silently drops every historical version.
release_json="$(gh release list --limit 1000 --json tagName,isDraft)" \
    || die "GitHub release inventory failed; refusing to publish an incomplete version set"
jq -e 'type == "array" and all(.[];
    (.tagName | type == "string" and length > 0) and (.isDraft | type == "boolean")) and
    ([.[].tagName] | length == (unique | length))' <<<"$release_json" >/dev/null \
    || die "Could not parse the GitHub release inventory"
tag_output="$(jq -r '.[] | select(.isDraft | not) | .tagName' <<<"$release_json")" \
    || die "Could not parse release tags"
tag_count="$(jq 'length' <<<"$release_json")" \
    || die "Could not count releases"
if [ "$tag_count" -ge 1000 ]; then
    die "Release inventory hit the 1000-release limit; refusing to omit older versions"
fi
tags=()
if [ -n "$tag_output" ]; then
    mapfile -t tags <<<"$tag_output"
fi

read_assets() {
    local release="$1" listing
    listing="$(gh release view "$release" --json assets)" \
        || die "Could not inspect assets for release $release"
    jq -e 'type == "object" and (.assets | type == "array") and
        all(.assets[]; (.name | type == "string" and length > 0)) and
        ([.assets[].name] | length == (unique | length))' <<<"$listing" >/dev/null \
        || die "Invalid asset inventory for release $release"
    printf '%s\n' "$listing"
}

install_archive() {
    local release="$1" name="$2" listing="$3" expected_asset="$4"
    local match_count
    match_count="$(jq --arg name "$expected_asset" '[.assets[] | select(.name == $name)] | length' <<<"$listing")" \
        || die "Could not parse asset inventory for $release"
    if [ "$match_count" -eq 0 ]; then
        echo "  ⏭️  $release has no $expected_asset"
        return 0
    fi
    [ "$match_count" -eq 1 ] || die "Release $release has duplicate asset name $expected_asset"

    local slot extract target
    slot="$(mktemp -d "${work}/${name}.XXXXXX")"
    if ! gh release download "$release" --pattern "$expected_asset" --dir "$slot"; then
        die "Known asset download failed: $release/$expected_asset"
    fi
    [ -f "$slot/$expected_asset" ] || die "Download reported success but omitted $expected_asset"

    extract="$slot/tree"
    mkdir -p "$extract"
    tar -xzf "$slot/$expected_asset" -C "$extract" \
        || die "Archive cannot be extracted: $release/$expected_asset"
    [ -f "$extract/index.html" ] && [ ! -L "$extract/index.html" ] \
        || die "Archive has no regular root index.html: $release/$expected_asset"

    target="$destination/$name"
    if [ -e "$target" ] || [ -L "$target" ]; then
        die "Refusing to overwrite existing restore target: $target"
    fi
    [ "$(realpath -- "$extract")" = "$slot/tree" ] &&
        [ "$(realpath -m -- "$target")" = "$destination/$name" ] \
        || die "Refusing unexpected archive publication path: $target"
    mv -- "$extract" "$target" \
        || die "Could not publish validated archive tree: $target"
    echo "  ✅ restored $name ($(find "$target" -type f | wc -l) files)"
}

for tag in "${tags[@]}"; do
    [[ "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || continue
    [ "$tag" = "$current_release" ] && continue
    assets="$(read_assets "$tag")"
    install_archive "$tag" "$tag" "$assets" "acloudviewer-${tag}-docs.tar.gz"
done

if [ "$restore_dev" = true ]; then
    has_main_devel=false
    for tag in "${tags[@]}"; do
        [ "$tag" = main-devel ] && has_main_devel=true
    done
    if [ "$has_main_devel" = false ]; then
        echo "ℹ️  No main-devel release exists; no dev archive is available yet"
    else
        assets="$(read_assets main-devel)"
        # Upload precedes pruning, so several SHA/run/attempt-stamped assets
        # can coexist. Git ordering is not publication time.
        dev_asset="$(jq -er '
            [.assets[] | select(.name | test("^acloudviewer-[0-9a-f]+(-[0-9]+-[0-9]+)?-docs\\.tar\\.gz$"))] as $docs |
            if ($docs | length) == 0 then ""
            elif all($docs[]; (.createdAt | type == "string") and
                     (try (.createdAt | fromdateiso8601 | type == "number") catch false)) then
                ($docs | sort_by(.createdAt | fromdateiso8601)) as $sorted |
                ($sorted[-1].createdAt | fromdateiso8601) as $latest |
                if ([$sorted[] | select((.createdAt | fromdateiso8601) == $latest)] | length) == 1
                then $sorted[-1].name
                else error("newest dev archive createdAt is ambiguous") end
            else error("dev archive has missing or invalid createdAt") end
        ' <<<"$assets")" || die "Cannot select newest main-devel documentation archive"
        if [ -n "$dev_asset" ]; then
            echo "  Selected newest dev archive: $dev_asset"
            install_archive main-devel dev "$assets" "$dev_asset"
        else
            echo "ℹ️  main-devel has no commit-stamped dev docs archive"
        fi
    fi
fi
