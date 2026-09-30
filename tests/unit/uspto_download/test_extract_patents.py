import io
import zipfile

from uspto_download.extract_patents import (
    get_safe_output_path,
    scan_zip_recursive,
    worker_process_zip,
)

BIOACTIVE_XML = "<p>The IC<sub>50</sub> was 5 nM.</p>"
PLAIN_XML = "<p>A pharmaceutical composition.</p>"


def _patent_zip(patent_id, xml=BIOACTIVE_XML, mol=True, cdx=True, tif=True):
    """Build a patent ZIP shaped like the ones inside a USPTO bulk archive."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        # USPTO writes these names uppercase, which the scan must tolerate.
        if mol:
            z.writestr(f"{patent_id}/{patent_id}-C00001.MOL", "mol block\n")
        if cdx:
            z.writestr(f"{patent_id}/{patent_id}-C00001.CDX", "cdx bytes\n")
        if tif:
            z.writestr(f"{patent_id}/{patent_id}-D00001.TIF", "tif bytes\n")
        z.writestr(f"{patent_id}/{patent_id}.XML", xml)
    return buf.getvalue()


def _scan(payload, with_mols=True, regex_xml=True):
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        return scan_zip_recursive(z, with_mols=with_mols, regex_xml=regex_xml)


def test_scan_keeps_both_mol_and_cdx_and_drops_tif():
    _, _, kept = _scan(_patent_zip("US20060005324A1"))

    lowered = [k.lower() for k in kept]
    assert any(k.endswith(".mol") for k in lowered)
    assert any(k.endswith(".cdx") for k in lowered), "CDX files must be kept"
    assert any(k.endswith(".xml") for k in lowered)
    assert not any(k.endswith(".tif") for k in lowered)


def test_structure_count_covers_mol_and_cdx():
    structures, _, _ = _scan(_patent_zip("US1"))
    assert structures == 2  # one MOL + one CDX

    structures, _, _ = _scan(_patent_zip("US1", cdx=False))
    assert structures == 1


def test_cdx_only_patent_still_counts_as_having_structures():
    structures, xml_hits, kept = _scan(_patent_zip("US1", mol=False))

    assert structures == 1
    assert xml_hits == 1
    assert any(k.lower().endswith(".cdx") for k in kept)


def test_xml_match_survives_structure_file_listed_after_it():
    """A .MOL after the XML used to reset the match count and drop the patent."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("US1/US1.XML", BIOACTIVE_XML)      # XML first
        z.writestr("US1/US1-C00001.MOL", "mol\n")     # structure after it
    _, xml_hits, _ = _scan(buf.getvalue())

    assert xml_hits == 1


def test_patent_without_bioactivity_is_not_saved(tmp_path):
    _, status, _, _ = worker_process_zip(
        _patent_zip("US2", xml=PLAIN_XML), "I2006.ZIP -> US2-20060112.ZIP",
        tmp_path, None, True, True)

    assert status == "NO_MATCH"
    assert not list(tmp_path.glob("*.zip"))


def test_matching_patent_is_saved_with_structures(tmp_path):
    _, status, _, _ = worker_process_zip(
        _patent_zip("US20060005324A1"),
        "I2006.ZIP -> US20060005324A1-20060112.ZIP", tmp_path, None, True, True)

    assert status == "SAVED"
    saved = list(tmp_path.glob("*.zip"))
    assert [p.name for p in saved] == ["US20060005324A1.zip"]

    members = [n.lower() for n in zipfile.ZipFile(saved[0]).namelist()]
    assert any(n.endswith(".mol") for n in members)
    assert any(n.endswith(".cdx") for n in members)


def test_output_name_strips_archive_prefix_and_date(tmp_path):
    """Both flat-zip and directory-prefixed tar members yield the patent id."""
    for job in ("I2019_bulk.zip -> US20190000001A1-20190103.zip",
                "I2019.tar -> 2019/US20190000001A1-20190103.zip"):
        assert get_safe_output_path(job, tmp_path).name == "US20190000001A1.zip"
