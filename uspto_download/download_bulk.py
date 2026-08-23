"""Download USPTO bulk patent archives.

Queries the USPTO Open Data Portal for a dataset product and downloads every
archive it publishes in a date range. The API returns results per request
window, so the range is walked in three-month steps.

Downloads resume: an archive already present with the expected size is
skipped, as is anything recorded as DONE in ``--logfile``.

An API key is required. Request one at
https://data.uspto.gov/apis/getting-started and put it in ``USPTO_TOKEN``,
either in the environment or in ``.env`` (see ``.env.example``).

Usage:
    export USPTO_TOKEN=your_key_here      # or set it in .env
    python -m uspto_download.download_bulk \
        --product APPDT --start 2021-01-01 --end 2022-01-01 --out data/uspto_bulk
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, List, Set, Tuple
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

USPTO_PRODUCTS_URL = "https://api.uspto.gov/api/v1/datasets/products"

ARCHIVE_EXTS = ('.zip', '.tar', '.tgz', '.tar.gz', '.tar.bz2', '.tbz2',
                '.tar.xz', '.txz')


def ymd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def add_months(d: date, n: int) -> date:
    y, m = d.year, d.month + n
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    last = [31, 29 if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0) else 28,
            31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]
    return date(y, m, min(d.day, last))


def windows_3m(start: date, end: date) -> Iterable[Tuple[date, date]]:
    """Walk ``start``..``end`` in three-month windows."""
    cur = start
    while cur <= end:
        nxt = add_months(cur, 3)
        # fileDataToDate is inclusive, so clamp the final window to `end`.
        yield cur, min(end, nxt)
        cur = nxt


def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-+" else "_" for c in name)


def find_urls(obj: Any) -> List[Tuple[str, str]]:
    """Pull (url, filename) pairs for downloadable archives out of an API response."""
    urls: List[Tuple[str, str]] = []

    def is_archive(name_or_url: str, kind: Any = None) -> bool:
        s = (name_or_url or "").lower()
        if s.endswith(ARCHIVE_EXTS):
            return True
        if kind and str(kind).strip().lower() in (
                'zip', 'tar', 'tgz', 'tar.gz', 'tar.bz2', 'tar.xz'):
            return True
        return False

    def rec(x: Any):
        if isinstance(x, dict):
            pfb = x.get('productFileBag')
            if isinstance(pfb, dict):
                fbag = pfb.get('fileDataBag')
                if isinstance(fbag, list):
                    for fd in fbag:
                        if not isinstance(fd, dict):
                            continue
                        url = (fd.get('fileDownloadURI') or fd.get('fileURL')
                               or fd.get('url'))
                        name = fd.get('fileName') or ""
                        if (isinstance(url, str)
                                and url.lower().startswith('http')
                                and is_archive(name or url, fd.get('fileTypeText'))):
                            urls.append((url, name))
            for v in x.values():
                rec(v)
        elif isinstance(x, list):
            for v in x:
                rec(v)

    rec(obj)
    return urls


class AuthError(RuntimeError):
    """The API rejected the key, so retrying cannot help."""


def req_with_retries(method: str, url: str, session: requests.Session, **kw):
    """Request with exponential backoff on rate limiting and server errors."""
    backoff = 1.0
    for attempt in range(7):
        try:
            r = session.request(method, url, timeout=(15, 300), **kw)
            # A rejected key is permanent; backing off just delays the error.
            if r.status_code in (401, 403):
                raise AuthError(
                    f"USPTO returned {r.status_code} {r.reason}. Check that "
                    f"USPTO_TOKEN holds a valid Open Data Portal API key.")
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} {r.reason}", response=r)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == 6:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)


def list_files_for_range(product: str, dfrom: date, dto: date, api_key: str,
                         session: requests.Session) -> List[Tuple[str, str]]:
    params = {"fileDataFromDate": dfrom.isoformat(),
              "fileDataToDate": dto.isoformat()}
    headers = {"accept": "application/json", "X-API-KEY": api_key}
    url = f"{USPTO_PRODUCTS_URL}/{product}"
    r = req_with_retries("GET", url, session, params=params, headers=headers)
    try:
        data = r.json()
    except json.JSONDecodeError:
        # The endpoint occasionally answers text/html on a transient error.
        time.sleep(1.0)
        r = req_with_retries("GET", url, session, params=params, headers=headers)
        data = r.json()
    return find_urls(data)


def pick_filename(url: str, resp: requests.Response, hinted: str) -> str:
    cd = resp.headers.get("Content-Disposition", "")
    if "filename=" in cd:
        fname = cd.split("filename=")[-1].strip('"; ')
        if fname:
            return safe_name(fname)
    if hinted:
        return safe_name(hinted)
    return safe_name(os.path.basename(urlparse(url).path) or "download.bin")


def download_one(url: str, outdir: str, api_key: str, session: requests.Session,
                 hinted_name: str = "") -> Tuple[str, int]:
    """Download one archive. Returns (path, 1 if fetched else 0)."""
    headers = {"X-API-KEY": api_key}
    with req_with_retries("GET", url, session, headers=headers, stream=True) as r:
        fname = pick_filename(url, r, hinted_name)
        os.makedirs(outdir, exist_ok=True)
        fpath = os.path.join(outdir, fname)

        content_length = r.headers.get("Content-Length")
        if (content_length and os.path.exists(fpath)
                and os.path.getsize(fpath) == int(content_length)):
            return fpath, 0

        # Write to .part first so an interrupted run cannot leave a truncated
        # archive that later looks complete.
        tmp = fpath + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, fpath)
        return fpath, 1


def get_completed_files(logfile=None) -> Set[str]:
    """Names recorded as DONE in the log, so they can be skipped."""
    completed: Set[str] = set()
    if logfile is None or not os.path.exists(logfile):
        return completed
    with open(logfile) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2 and parts[1] == "DONE":
                completed.add(parts[0])
    return completed


def main():
    ap = argparse.ArgumentParser(
        description="Download USPTO bulk patent archives for a date range.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--product", default="APPDT",
                    help="USPTO dataset product identifier, e.g. APPDT")
    ap.add_argument("--start", default='2021-01-01', help="Start date YYYY-MM-DD")
    ap.add_argument("--end", default='2022-01-01',
                    help="End date YYYY-MM-DD (inclusive)")
    ap.add_argument("--out", required=True, help="Output directory for archives")
    ap.add_argument("--logfile",
                    help="Processing log; entries marked DONE are skipped")
    ap.add_argument("--name-filter",
                    help="Only download archives whose filename matches this "
                         "regex, e.g. '^I\\d{8}\\.(tar|ZIP)$' for the main daily "
                         "archives. The product listing also carries supplements "
                         "and viewer software.")
    ap.add_argument("--limit", type=int,
                    help="Stop after this many archives (useful for a trial run)")
    ap.add_argument("--workers", type=int, default=2,
                    help="Concurrent downloads; USPTO rate-limits per key, so "
                         "raise this with care")
    ap.add_argument("--verbose", action=argparse.BooleanOptionalAction,
                    default=True, help="Log every skipped file")
    args = ap.parse_args()

    load_dotenv()
    api_key = os.environ.get("USPTO_TOKEN")
    if not api_key:
        print("ERROR: set USPTO_TOKEN to a USPTO Open Data Portal API key,\n"
              "in the environment or in .env (see .env.example).\n"
              "Request one at https://data.uspto.gov/apis/getting-started",
              file=sys.stderr)
        sys.exit(2)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")

    completed_files = get_completed_files(args.logfile)
    if args.logfile:
        logging.info("%d file(s) marked DONE in %s will be skipped",
                     len(completed_files), args.logfile)

    name_re = re.compile(args.name_filter) if args.name_filter else None

    listing_session = requests.Session()
    # One session per worker: requests.Session is not documented as thread-safe.
    download_sessions = [requests.Session() for _ in range(max(1, args.workers))]

    # ── 1. Collect archive URLs, one three-month window at a time ──
    all_items: List[Tuple[str, str]] = []
    seen: Set[str] = set()
    for dfrom, dto in windows_3m(ymd(args.start), ymd(args.end)):
        logging.info("Listing %s %s..%s", args.product,
                     dfrom.isoformat(), dto.isoformat())
        try:
            items = list_files_for_range(args.product, dfrom, dto, api_key,
                                         listing_session)
        except AuthError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(2)
        except Exception as e:
            logging.error("List failed for %s..%s: %s", dfrom, dto, e)
            continue

        added = 0
        for url, name in items:
            if url in seen:
                continue
            if name_re and not name_re.search(name or ""):
                continue
            if name in completed_files:
                if args.verbose:
                    logging.info("Skipping (already processed per log): %s", name)
                continue
            if (Path(args.out) / name).exists():
                if args.verbose:
                    logging.info("Skipping (file already exists): %s", name)
                continue
            seen.add(url)
            all_items.append((url, name))
            added += 1
        logging.info("Found %d file(s), %d new in this window", len(items), added)
        if args.limit and len(all_items) >= args.limit:
            break

    if args.limit:
        all_items = all_items[:args.limit]

    if not all_items:
        logging.warning("No files found. Nothing to download.")
        return
    logging.info("Downloading %d archive(s): %s", len(all_items),
                 ", ".join(n for _, n in all_items[:5]))

    # ── 2. Download ──
    os.makedirs(args.out, exist_ok=True)
    downloaded = 0
    n_workers = len(download_sessions)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [
            ex.submit(download_one, url, args.out, api_key,
                      download_sessions[i % n_workers], name)
            for i, (url, name) in enumerate(all_items)
        ]
        for fut in as_completed(futures):
            try:
                fpath, fetched = fut.result()
                downloaded += fetched
                logging.info("%s: %s", "Downloaded" if fetched else "Skipped (exists)",
                             fpath)
            except Exception as e:
                logging.error("Download failed: %s", e)

    logging.info("Completed. Total: %d, downloaded: %d, skipped: %d",
                 len(all_items), downloaded, len(all_items) - downloaded)


if __name__ == "__main__":
    main()
