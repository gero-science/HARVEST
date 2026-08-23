"""Extract per-patent ZIPs from USPTO bulk archives.

A bulk archive downloaded by ``download_bulk`` is a .tar or .zip holding one
nested ZIP per patent. This walks those archives and writes out only the
patents worth running the pipeline on, filtered by:

    --with-mols    the patent ships embedded structure files (MOL/CDX)
    --regex-xml    the patent text mentions a binding-affinity metric

The result is a directory of per-patent ZIPs, which is exactly what
``pipeline.py --input-path`` consumes.

Progress is logged per archive, so an interrupted run resumes where it left
off rather than repeating expensive scans.

Usage:
    python -m uspto_download.extract_patents \
        -i data/uspto_bulk -o data/patents \
        --start 2018 --end 2020 --with-mols --regex-xml
"""

import argparse
import io
import itertools
import os
import sys
import tarfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

from tqdm.auto import tqdm

from .bioactivity_filter import decode_bytes, find_regexp

# Chemical structure files worth keeping: MOL feeds patent_processor, CDX feeds
# the CDX parser.
STRUCTURE_EXTS = ('.mol', '.cdx')

log_lock = threading.Lock()
nested_log_lock = threading.Lock()


def get_safe_output_path(path_name: str, output_folder: Path) -> Path:
    """Name the output ZIP after the patent, dropping the date suffix.

    ``path_name`` is "<archive> -> <member>", and the member may or may not
    carry a directory prefix, so strip both before taking the patent id:
    "I2019.tar -> 2019/US20190000001A1-20190103.zip" -> "US20190000001A1.zip".
    """
    member = path_name.split(' -> ')[-1]
    safe_name = member.split('/')[-1].split('-')[0]
    if not safe_name.lower().endswith('.zip'):
        safe_name += '.zip'
    return output_folder / safe_name


def scan_zip_recursive(zip_file_obj, with_mols: bool,
                       regex_xml: bool) -> Tuple[int, int, list]:
    """Count structure and matching XML members, returning the files to keep.

    Structure files are MOL and CDX. USPTO ships them as a pair per depicted
    compound (``...-C00001.MOL`` / ``...-C00001.CDX``); ``patent_processor``
    reads the MOL and the CDX via ``cdx_reader``, so both are kept.
    TIF images are always dropped -- they are the bulk of the archive and
    nothing downstream reads them.
    """
    structure_count = 0
    xml_count = 0
    save_file_list = []
    try:
        file_list = zip_file_obj.infolist()
    except zipfile.BadZipFile:
        return 0, 0, []

    # Checking for structures first avoids parsing XML in patents that cannot
    # qualify anyway.
    has_structure = any(not zi.is_dir()
                        and zi.filename.lower().endswith(STRUCTURE_EXTS)
                        for zi in file_list)

    for zip_info in file_list:
        if zip_info.is_dir():
            continue
        filename = zip_info.filename
        lower = filename.lower()

        if lower.endswith(STRUCTURE_EXTS):
            structure_count += 1
            save_file_list.append(filename)

        elif lower.endswith('.xml'):
            if not regex_xml:
                save_file_list.append(filename)  # keep all XML when not filtering
            elif has_structure or not with_mols:
                try:
                    if find_regexp(decode_bytes(zip_file_obj.read(zip_info))):
                        xml_count += 1
                        save_file_list.append(filename)
                except Exception:
                    continue

        elif lower.endswith('.zip'):
            try:
                with zip_file_obj.open(zip_info) as nested_file:
                    nested_zip_bytes = nested_file.read()
                with zipfile.ZipFile(io.BytesIO(nested_zip_bytes)) as nested_zip_obj:
                    nested_struct, nested_xml, _ = scan_zip_recursive(
                        nested_zip_obj, with_mols, regex_xml)
                    structure_count += nested_struct
                    xml_count += nested_xml
            except Exception:
                continue

    return structure_count, xml_count, save_file_list


def worker_process_zip(zip_bytes, zip_path_name, output_folder, nested_log_path,
                       with_mols, regex_xml):
    """Scan one nested patent ZIP and save it when it matches."""
    structure_count = 0
    xml_count = 0
    status = "ERROR"
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            structure_count, xml_count, files = scan_zip_recursive(
                zf, with_mols, regex_xml)
            if (structure_count > 0) == with_mols and (xml_count > 0 or not regex_xml):
                output_path = get_safe_output_path(zip_path_name, output_folder)
                with zipfile.ZipFile(output_path, "w",
                                     compression=zipfile.ZIP_DEFLATED) as zout:
                    for mn in files:
                        zout.writestr(mn, zf.read(mn))
                status = "SAVED"
            else:
                status = "NO_MATCH"
    except Exception as e:
        print(f"  -> ERROR processing worker job {zip_path_name}: {e}")
        status = "ERROR"
    finally:
        if nested_log_path:
            log_nested_zip_result(nested_log_path, zip_path_name,
                                  structure_count, xml_count)

    return zip_path_name, status, structure_count, xml_count


def update_log(log_path, file_name, status, structure_count=0, xml_count=0):
    """Record an archive's status, replacing any earlier line for it."""
    with log_lock:
        file_key = os.path.basename(file_name)
        new_line = f"{file_key}\t{status}\t{structure_count}\t{xml_count}\n"
        lines = []
        if os.path.exists(log_path):
            with open(log_path) as f:
                lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith(file_key + '\t'):
                lines[i] = new_line
                break
        else:
            lines.append(new_line)
        with open(log_path, 'w') as f:
            f.writelines(lines)


def log_nested_zip_result(log_path, zip_name, mol_flag, xml_flag):
    with nested_log_lock:
        line = zip_name.replace(' -> ', '\t') + f"\t{mol_flag}\t{xml_flag}\n"
        with open(log_path, 'a') as f:
            f.write(line)


def get_completed_files(log_path):
    completed = set()
    if not os.path.exists(log_path):
        return completed
    with open(log_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2 and parts[1] == "DONE":
                completed.add(parts[0])
    return completed


def collect_nested_zips(archive_path: Path):
    """Read every nested ZIP out of one bulk .zip or .tar archive."""
    archive_name = archive_path.name
    jobs = []
    if archive_path.suffix.lower() == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zf:
            for member_name in zf.namelist():
                if member_name.lower().endswith('.zip'):
                    jobs.append((zf.read(member_name),
                                 f"{archive_name} -> {member_name}"))
    else:
        # ignore_zeros lets a concatenation of per-day tars be read as one file.
        with tarfile.open(archive_path, 'r:*', ignore_zeros=True) as tf:
            for member in tf.getmembers():
                if member.isfile() and member.name.lower().endswith('.zip'):
                    with tf.extractfile(member) as stream:
                        if stream:
                            jobs.append((stream.read(),
                                         f"{archive_name} -> {member.name}"))
    return jobs


def main():
    ap = argparse.ArgumentParser(
        description="Extract per-patent ZIPs from USPTO bulk archives.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--start", type=int, default=2018, help="Start year")
    ap.add_argument("--end", type=int, default=2020, help="End year (inclusive)")
    ap.add_argument("-i", "--input-dir", type=Path, required=True,
                    help="Directory holding the bulk archives")
    ap.add_argument("-o", "--output-dir", type=Path, required=True,
                    help="Directory for the extracted per-patent ZIPs")
    ap.add_argument("-c", "--cleanup", action="store_true",
                    help="Delete each bulk archive once processed")
    ap.add_argument('--with-mols', action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Keep only patents that ship embedded structure files (MOL/CDX)")
    ap.add_argument('--regex-xml', action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Keep only patents whose text mentions a binding metric")
    ap.add_argument("--max-thread", type=int, default=8,
                    help="Worker threads per archive")
    ap.add_argument('--nested-log', type=Path,
                    help="Per-patent log path (default: nested_processing_log.txt "
                         "in the output directory)")
    args = ap.parse_args()

    if not args.with_mols and not args.regex_xml:
        sys.exit("Error: enable at least one of --with-mols or --regex-xml.")

    output_folder = args.output_dir
    output_folder.mkdir(parents=True, exist_ok=True)
    log_file = output_folder / "processing_log.txt"
    nested_log_file = args.nested_log or (output_folder / "nested_processing_log.txt")

    print("--- USPTO nested archive extractor ---")
    print(f"Output directory: {output_folder.resolve()}")
    print(f"Main log:   {log_file}")
    print(f"Nested log: {nested_log_file}")
    print(f"Using {args.max_thread} parallel workers.")

    completed_files = get_completed_files(log_file)
    print(f"{len(completed_files)} archive(s) already marked DONE.")

    all_archives = sorted(itertools.chain(
        args.input_dir.glob("*.tar"), args.input_dir.glob("*.zip"),
        args.input_dir.glob("*.TAR"), args.input_dir.glob("*.ZIP")))
    if not all_archives:
        sys.exit(f"No .tar/.zip archives found in {args.input_dir}")

    years = range(args.start, args.end + 1)
    for archive_path in tqdm(all_archives, desc='Bulk archives'):
        archive_name = archive_path.name
        if archive_name in completed_files:
            continue
        if not any(str(year) in archive_name.lower() for year in years):
            continue

        update_log(log_file, archive_name, "IN_PROGRESS")
        try:
            nested_zip_jobs = collect_nested_zips(archive_path)

            saved_count = total_struct = total_xml = 0
            with ThreadPoolExecutor(max_workers=args.max_thread) as executor:
                futures = {
                    executor.submit(worker_process_zip, j_bytes, j_name,
                                    output_folder, nested_log_file,
                                    args.with_mols, args.regex_xml)
                    for j_bytes, j_name in nested_zip_jobs}
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc=f'Unpacking {archive_name}', leave=False):
                    _, status, structure_count, xml_count = future.result()
                    if status == "SAVED":
                        saved_count += 1
                    total_struct += structure_count
                    total_xml += max(xml_count, 0)

            print(f"{archive_name}: {total_struct} structure files "
                  f"(MOL/CDX), {total_xml} matching XMLs, "
                  f"{saved_count} patent ZIPs saved.")
            update_log(log_file, archive_name, "DONE", total_struct, total_xml)

            if args.cleanup:
                os.remove(archive_path)
                print(f"Removed processed archive: {archive_path}")

        except Exception as e:
            print(f"FATAL ERROR processing {archive_name}: {e}")
            update_log(log_file, archive_name, "ERROR")

    print("--- Done. ---")


if __name__ == "__main__":
    main()
