#!/usr/bin/env python3
"""Fetch the pose-accuracy fixture clips for CI.

Two sources; the first one configured wins:
  1. POSE_FIXTURE_URLS — whitespace-separated download URLs, one per clip,
     in the order the clips appear in fixture-manifest.json.
  2. Cloudflare R2 — POSE_FIXTURE_R2_BUCKET (plus optional
     POSE_FIXTURE_R2_PREFIX); credentials come from AWS_ACCESS_KEY_ID /
     AWS_SECRET_ACCESS_KEY and the account endpoint from R2_ENDPOINT.
     Each clip is fetched as s3://<bucket>/<prefix><file> with the
     runner's preinstalled `aws` CLI (ubuntu-latest ships it).

Verifies sha256 of every download against the manifest. Exits non-zero with
a clear message when no source is configured or a hash mismatches —
the gate must FAIL, never silently pass.
"""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_url(url: str, dest: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "turnip-pose-harness/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        for chunk in iter(lambda: r.read(1 << 20), b""):
            f.write(chunk)


def fetch_r2(bucket: str, prefix: str, key: str, dest: str, endpoint: str) -> bool:
    aws = shutil.which("aws")
    if not aws:
        print("::error::R2 fixture source is configured but the `aws` CLI is not "
              "installed on this runner.")
        return False
    src = f"s3://{bucket}/{prefix}{key}"
    print(f"  {src}")
    r = subprocess.run(
        [aws, "s3", "cp", src, dest, "--endpoint-url", endpoint],
        capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"::error::failed to download {key} from R2:\n{r.stderr.strip()}")
        return False
    return True


def main() -> int:
    manifest_path, out_dir = sys.argv[1], sys.argv[2]
    with open(manifest_path) as f:
        manifest = json.load(f)
    clips = manifest["clips"]

    urls = os.environ.get("POSE_FIXTURE_URLS", "").split()
    bucket = os.environ.get("POSE_FIXTURE_R2_BUCKET", "").strip()

    if urls and len(urls) != len(clips):
        print(f"::error::POSE_FIXTURE_URLS has {len(urls)} URLs but the manifest lists "
              f"{len(clips)} clips.")
        return 1

    r2 = None
    if not urls:
        if not bucket:
            print("::error::No fixture source configured. The pose-accuracy fixture "
                  "videos are Hoie's personal footage and are not committed to the "
                  "repo; configure either POSE_FIXTURE_URLS (one download URL per "
                  "clip, in manifest order) or POSE_FIXTURE_R2_BUCKET (plus "
                  "POSE_FIXTURE_R2_PREFIX, R2_ENDPOINT, AWS_ACCESS_KEY_ID and "
                  "AWS_SECRET_ACCESS_KEY for Cloudflare R2). See "
                  "PoseAccuracy/README.md.")
            return 1
        prefix = os.environ.get("POSE_FIXTURE_R2_PREFIX", "").strip().lstrip("/")
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        endpoint = os.environ.get("R2_ENDPOINT", "").strip().rstrip("/")
        if not endpoint:
            print("::error::POSE_FIXTURE_R2_BUCKET is set but R2_ENDPOINT is not. "
                  "Add it as a repo secret: https://<account-id>.r2.cloudflarestorage.com")
            return 1
        if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get("AWS_SECRET_ACCESS_KEY"):
            print("::error::POSE_FIXTURE_R2_BUCKET is set but AWS_ACCESS_KEY_ID / "
                  "AWS_SECRET_ACCESS_KEY are missing. Add them as repo secrets from "
                  "the R2 API token.")
            return 1
        r2 = (bucket, prefix, endpoint)

    os.makedirs(out_dir, exist_ok=True)
    for i, clip in enumerate(clips):
        dest = os.path.join(out_dir, clip["file"])
        print(f"downloading {clip['id']} ({clip['label']}) ...")
        if r2:
            bkt, prefix, endpoint = r2
            if not fetch_r2(bkt, prefix, clip["file"], dest, endpoint):
                return 1
        else:
            fetch_url(urls[i], dest)
        got = sha256_file(dest)
        if got != clip["sha256"]:
            print(f"::error::sha256 mismatch for {clip['file']}: expected "
                  f"{clip['sha256'][:16]}..., got {got[:16]}...")
            return 1
        print(f"  ok sha256={got[:16]}...")
    print(f"fixture ready: {len(clips)} clips in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
