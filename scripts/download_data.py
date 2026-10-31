"""Download the official CUAD release and extract its original annotation JSON."""

from __future__ import annotations

import argparse
import os
import tempfile
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DATA_URL = "https://raw.githubusercontent.com/The-Atticus-Project/cuad/main/data.zip"
MAX_FILE_BYTES = 50 * 1024 * 1024


def download(force: bool = False) -> Path:
    """Download only the source archive temporarily and retain CUADv1.json."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / "CUADv1.json"
    if target.is_file() and not force:
        print(f"Using existing {target} ({target.stat().st_size:,} bytes).")
        _write_raw_readme()
        return target

    temporary_json = target.with_suffix(".json.tmp")
    with tempfile.NamedTemporaryFile(
        prefix="cuad-", suffix=".zip", dir=RAW_DIR, delete=False
    ) as temporary:
        archive_path = Path(temporary.name)
    try:
        request = urllib.request.Request(
            DATA_URL, headers={"User-Agent": "cuad-clause-tagging/0.1"}
        )
        total_bytes = 0
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            archive_path.open("wb") as destination,
        ):
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                total_bytes += len(block)
                if total_bytes > MAX_FILE_BYTES:
                    raise ValueError(
                        "CUAD archive exceeded the 50 MiB download limit."
                    )
                destination.write(block)

        with zipfile.ZipFile(archive_path) as archive:
            try:
                entry = archive.getinfo("CUADv1.json")
            except KeyError as error:
                raise ValueError("The official archive lacks CUADv1.json.") from error
            if entry.file_size > MAX_FILE_BYTES:
                raise ValueError("CUADv1.json exceeds the 50 MiB file limit.")
            with archive.open(entry) as source, temporary_json.open("wb") as output:
                while block := source.read(1024 * 1024):
                    output.write(block)
            os.replace(temporary_json, target)
    finally:
        archive_path.unlink(missing_ok=True)
        temporary_json.unlink(missing_ok=True)

    _write_raw_readme()
    print(f"Saved {target} ({target.stat().st_size:,} bytes).")
    return target


def _write_raw_readme() -> None:
    downloaded_at = datetime.now(UTC).date().isoformat()
    readme = RAW_DIR / "README.md"
    readme.write_text(
        "# Raw CUAD data\n\n"
        f"- Source archive: [{DATA_URL}]({DATA_URL})\n"
        "- Extracted source file: `CUADv1.json` (SQuAD-style contract text and "
        "character-offset annotations)\n"
        f"- Downloaded: {downloaded_at} UTC\n"
        "- The archive is temporary and is not retained; `CUADv1.json` is the "
        "original file within it.\n"
        "- CUAD is distributed under CC BY 4.0. Attribute The Atticus Project "
        "and Hendrycks et al.; see the root README.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Download again.")
    args = parser.parse_args()
    download(force=args.force)


if __name__ == "__main__":
    main()
