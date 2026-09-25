"""Acquire the OULAD dataset into ``data/raw/oulad/``.

The Open University's own download link (``anonymisedData.zip`` on the
schools.stem CDN) returned HTTP 404 as of 2026-09-25, and every other
OU-hosted endpoint now redirects to a generic marketing page. The dataset is
still published under CC-BY 4.0, so this script tries a list of mirrors in
order rather than depending on a single host.

If every mirror fails, the script exits non-zero with manual instructions. It
never silently produces partial data: an incomplete or non-zip download is
deleted before exiting.

Usage::

    python scripts/download_data.py            # download, extract, verify
    python scripts/download_data.py --verify   # verify an existing extraction
    python scripts/download_data.py --force    # re-download even if present
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from dropout_ews.config.settings import DATA_DIR

OULAD_DIR = DATA_DIR / "raw" / "oulad"
ARCHIVE_PATH = DATA_DIR / "raw" / "oulad.zip"

USER_AGENT = "dropout-ews/0.1 (research; +https://github.com)"
CHUNK_BYTES = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class Mirror:
    name: str
    url: str
    note: str = ""


# Ordered by preference. The OU CDN link is kept first and expected to fail so
# that its recovery is detected automatically rather than staying dead in our
# tooling forever.
MIRRORS: tuple[Mirror, ...] = (
    Mirror(
        name="ou-cdn",
        url="http://schools.stem.open.ac.uk/cdn/files/anonymisedData.zip",
        note="Canonical OU link. Returned 404 on 2026-09-25.",
    ),
    Mirror(
        name="uci",
        url="https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip",
        note="UCI ML Repository mirror, dataset 349.",
    ),
)

# The seven CSVs OULAD ships. Verification requires all of them: a partial
# extraction that still contains studentInfo.csv would otherwise look healthy
# and fail much later during feature construction.
EXPECTED_FILES: tuple[str, ...] = (
    "courses.csv",
    "assessments.csv",
    "vle.csv",
    "studentInfo.csv",
    "studentRegistration.csv",
    "studentAssessment.csv",
    "studentVle.csv",
)


class AcquisitionError(RuntimeError):
    """Raised when the dataset cannot be obtained or is not intact."""


def _download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest``, verifying it is actually a zip.

    Servers that have removed a file often answer with an HTML error page and
    HTTP 200, so the magic bytes are checked rather than trusting the status
    code or Content-Type.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    with urllib.request.urlopen(request, timeout=120) as response:
        declared = response.headers.get("Content-Length")
        total = int(declared) if declared and declared.isdigit() else None

        first = response.read(4)
        if not first.startswith(b"PK"):
            raise AcquisitionError(
                f"{url} did not return a zip archive (first bytes: {first!r}). "
                "The host is most likely serving an error page with HTTP 200."
            )

        written = len(first)
        with partial.open("wb") as fh:
            fh.write(first)
            while chunk := response.read(CHUNK_BYTES):
                fh.write(chunk)
                written += len(chunk)
                if total:
                    pct = 100 * written / total
                    print(f"\r  {written / 1e6:,.1f} / {total / 1e6:,.1f} MB ({pct:.0f}%)", end="")
                else:
                    print(f"\r  {written / 1e6:,.1f} MB", end="")
        print()

        if total is not None and written != total:
            partial.unlink(missing_ok=True)
            raise AcquisitionError(
                f"truncated download: expected {total} bytes, received {written}"
            )

    partial.replace(dest)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def acquire_archive(force: bool = False) -> Mirror:
    """Download the archive from the first mirror that works."""
    if ARCHIVE_PATH.exists() and not force:
        print(f"Archive already present: {ARCHIVE_PATH}")
        return Mirror(name="cached", url=str(ARCHIVE_PATH))

    failures: list[str] = []
    for mirror in MIRRORS:
        print(f"Trying mirror '{mirror.name}': {mirror.url}")
        try:
            _download(mirror.url, ARCHIVE_PATH)
        except (urllib.error.URLError, AcquisitionError, OSError, TimeoutError) as exc:
            print(f"  failed: {exc}")
            failures.append(f"{mirror.name}: {exc}")
            continue
        print(f"  ok ({ARCHIVE_PATH.stat().st_size / 1e6:,.1f} MB)")
        return mirror

    raise AcquisitionError(
        "every mirror failed.\n  " + "\n  ".join(failures) + "\n\nManual fallback:\n"
        "  1. Obtain OULAD (CC-BY 4.0) from a source you trust, e.g. the UCI ML\n"
        "     Repository (dataset 349) or Kaggle.\n"
        f"  2. Place the seven CSVs directly in {OULAD_DIR}\n"
        "  3. Re-run: python scripts/download_data.py --verify"
    )


def extract(force: bool = False) -> None:
    """Extract the archive, flattening any nested top-level directory."""
    if OULAD_DIR.exists() and not force:
        present = {p.name for p in OULAD_DIR.glob("*.csv")}
        if set(EXPECTED_FILES) <= present:
            print(f"Already extracted: {OULAD_DIR}")
            return

    OULAD_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        members = [m for m in archive.namelist() if m.lower().endswith(".csv")]
        if not members:
            raise AcquisitionError(f"{ARCHIVE_PATH} contains no CSV files")
        for member in members:
            # Flatten: some mirrors nest the CSVs inside a folder. Use only the
            # basename so path traversal in a crafted archive cannot escape.
            target = OULAD_DIR / Path(member).name
            with archive.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"  extracted {target.name} ({target.stat().st_size / 1e6:,.1f} MB)")


def verify() -> dict[str, int]:
    """Check every expected CSV is present and non-trivial.

    Returns a mapping of filename to line count (including the header).
    """
    if not OULAD_DIR.exists():
        raise AcquisitionError(f"{OULAD_DIR} does not exist; run without --verify first")

    missing = [name for name in EXPECTED_FILES if not (OULAD_DIR / name).is_file()]
    if missing:
        raise AcquisitionError(
            f"missing expected OULAD files: {missing}\n"
            f"Found instead: {sorted(p.name for p in OULAD_DIR.glob('*.csv'))}"
        )

    counts: dict[str, int] = {}
    for name in EXPECTED_FILES:
        path = OULAD_DIR / name
        with path.open("rb") as fh:
            lines = sum(1 for _ in fh)
        if lines < 2:
            raise AcquisitionError(f"{name} has no data rows")
        counts[name] = lines
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="only verify an existing extraction")
    parser.add_argument("--force", action="store_true", help="re-download and re-extract")
    args = parser.parse_args()

    try:
        if not args.verify:
            acquire_archive(force=args.force)
            extract(force=args.force)
        counts = verify()
    except AcquisitionError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print("\nVerified OULAD extraction:")
    width = max(len(n) for n in counts)
    for name, lines in counts.items():
        print(f"  {name:<{width}}  {lines - 1:>12,} data rows")

    if ARCHIVE_PATH.exists():
        print(f"\nArchive sha256: {_sha256(ARCHIVE_PATH)}")
        print("Record this in docs/DATA_CARD.md so the exact snapshot is identifiable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
