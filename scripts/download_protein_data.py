#!/usr/bin/env python3
"""
Download the UniProt reference data the `proteins` stage needs.

`protein_postprocessor` resolves a gene symbol and species to a sequence by
reading `data/protein_data/uniprot_sprot.fasta` (see `UNIPROT_FASTA_PATH` in
protein_postprocessor/config_postprocess.py). That file is not in the
repository -- it is ~90 MB compressed -- so fetch it once before running the
pipeline with the `proteins` stage.

Usage:
    python scripts/download_protein_data.py
    python scripts/download_protein_data.py --output-dir /data/protein_data --force
"""

import argparse
import gzip
import sys
from pathlib import Path

import requests

UNIPROT_SPROT_URL = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
    "knowledgebase/complete/uniprot_sprot.fasta.gz"
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "protein_data"

# Swiss-Prot is ~575k records; anything far below this means a truncated file.
MIN_EXPECTED_RECORDS = 100_000


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def download_and_decompress(url: str, target: Path) -> None:
    """Stream a .gz to disk and decompress it, leaving `target` complete or absent."""
    # Write to .part first so an interrupted run cannot leave a truncated FASTA
    # that later looks complete.
    tmp = target.with_suffix(target.suffix + ".part")

    with requests.get(url, stream=True, timeout=(15, 300)) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        print(f"Downloading {url}")
        if total:
            print(f"  compressed size: {human(total)}")

        done = 0
        last_pct = -1
        with gzip.open(response.raw, "rb") as gz, open(tmp, "wb") as out:
            while chunk := gz.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                # response.raw position tracks compressed bytes consumed.
                if total:
                    pct = min(100, int(100 * response.raw.tell() / total))
                    if pct != last_pct and pct % 5 == 0:
                        print(f"  {pct:3d}%  ({human(done)} decompressed)", flush=True)
                        last_pct = pct

    tmp.replace(target)


def count_records(path: Path) -> int:
    """Count FASTA headers, so a truncated download is obvious."""
    count = 0
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(">"):
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download uniprot_sprot.fasta for the proteins stage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to place uniprot_sprot.fasta in")
    parser.add_argument("--url", default=UNIPROT_SPROT_URL,
                        help="Source archive (gzipped FASTA)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if the FASTA is already present")
    args = parser.parse_args()

    output_dir = args.output_dir
    target = output_dir / "uniprot_sprot.fasta"

    if target.exists() and not args.force:
        print(f"Already present: {target} ({human(target.stat().st_size)})")
        print("Use --force to download it again.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        download_and_decompress(args.url, target)
    except requests.RequestException as exc:
        print(f"ERROR: download failed: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"ERROR: could not write {target}: {exc}", file=sys.stderr)
        return 1

    records = count_records(target)
    print(f"\nWrote {target} ({human(target.stat().st_size)}, {records:,} sequences)")

    if records < MIN_EXPECTED_RECORDS:
        print(
            f"WARNING: only {records:,} sequences, expected at least "
            f"{MIN_EXPECTED_RECORDS:,}. The file looks truncated -- re-run with --force.",
            file=sys.stderr,
        )
        return 1

    print("The proteins stage can now run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
