"""Tests for CDX parsing done while the patent ZIP is read."""

import zipfile

import pytest

from patent_processor.cdx_extractor import extract_chem_num_from_cdx_filename
from patent_processor.zip_source import ZipPatentSource


PATENT_ID = "US20240001234A1"
PREFIX = f"{PATENT_ID}-20240101/{PATENT_ID}-20240101"


def build_zip(tmp_path, cdx_names, xml=True):
    zip_path = tmp_path / f"{PATENT_ID}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        if xml:
            zf.writestr(f"{PREFIX}.XML", "<patent><invention-title>Test</invention-title></patent>")
        for name in cdx_names:
            zf.writestr(name, f"cdx-bytes-{name}".encode("utf-8"))
    return zip_path


def fake_parser(mapping, failures=()):
    """Build a parse_cdx_bytes stub keyed by the fake CDX payload."""

    def parse(cdx_bytes):
        key = cdx_bytes.decode("utf-8")
        if key in failures:
            raise RuntimeError("boom")
        return mapping.get(key, (None, None))

    return parse


def test_extract_chem_num_from_cdx_filename():
    assert extract_chem_num_from_cdx_filename(f"{PREFIX}-C00013.CDX") == "00013"
    assert extract_chem_num_from_cdx_filename(f"{PREFIX}-C13.cdx") == "13"
    assert extract_chem_num_from_cdx_filename(f"{PREFIX}.MOL") is None
    assert extract_chem_num_from_cdx_filename("") is None
    assert extract_chem_num_from_cdx_filename(None) is None


def test_process_cdx_files_caches_structures_under_padded_and_bare_keys(tmp_path, monkeypatch):
    zip_path = build_zip(tmp_path, [f"{PREFIX}-C00013.CDX", f"{PREFIX}-C00014.CDX"])
    monkeypatch.setattr(
        "patent_processor.zip_source.parse_cdx_bytes",
        fake_parser({f"cdx-bytes-{PREFIX}-C00013.CDX": ("CCO", "LFQSCWFLJHTTHZ-UHFFFAOYSA-N")}),
    )

    source = ZipPatentSource(str(zip_path))
    with zipfile.ZipFile(zip_path) as zf:
        cdx_data = source._process_cdx_files(zf, PATENT_ID)

    # Downstream looks up chem_num exactly as written in chemical_id, so both
    # CHEM-US-00013 and CHEM-US-13 must resolve to the same compound.
    assert cdx_data["00013"] == {"smiles": "CCO", "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"}
    assert cdx_data["13"] == cdx_data["00013"]

    # Compounds without a structure are still recorded.
    assert cdx_data["00014"] == {"smiles": None, "inchikey": None}
    assert cdx_data["14"] == cdx_data["00014"]


def test_process_cdx_files_survives_failures_and_empty_files(tmp_path, monkeypatch):
    zip_path = build_zip(tmp_path, [f"{PREFIX}-C00001.CDX", f"{PREFIX}-C00002.CDX"])
    with zipfile.ZipFile(zip_path, "a") as zf:
        zf.writestr(f"{PREFIX}-C00003.CDX", b"")
        zf.writestr(f"{PREFIX}-noise.CDX", b"cdx-bytes-noise")

    monkeypatch.setattr(
        "patent_processor.zip_source.parse_cdx_bytes",
        fake_parser(
            {f"cdx-bytes-{PREFIX}-C00002.CDX": ("CCO", "KEY")},
            failures={f"cdx-bytes-{PREFIX}-C00001.CDX"},
        ),
    )

    source = ZipPatentSource(str(zip_path))
    with zipfile.ZipFile(zip_path) as zf:
        cdx_data = source._process_cdx_files(zf, PATENT_ID)

    assert cdx_data["2"]["smiles"] == "CCO"
    assert "00001" not in cdx_data
    assert "00003" not in cdx_data


def test_iter_patents_fills_cdx_cache(tmp_path, monkeypatch):
    zip_path = build_zip(tmp_path, [f"{PREFIX}-C00001.CDX"])
    monkeypatch.setattr(
        "patent_processor.zip_source.parse_cdx_bytes",
        fake_parser({f"cdx-bytes-{PREFIX}-C00001.CDX": ("CCO", "KEY")}),
    )

    source = ZipPatentSource(str(zip_path))
    documents = list(source.iter_patents())

    assert len(documents) == 1
    assert source.get_cdx_for_patent(PATENT_ID)["00001"] == {"smiles": "CCO", "inchikey": "KEY"}


def test_parse_cdx_false_skips_cdx_processing(tmp_path, monkeypatch):
    zip_path = build_zip(tmp_path, [f"{PREFIX}-C00001.CDX"])

    def unexpected_call(cdx_bytes):
        raise AssertionError("CDX parsing must be skipped when parse_cdx=False")

    monkeypatch.setattr("patent_processor.zip_source.parse_cdx_bytes", unexpected_call)

    source = ZipPatentSource(str(zip_path), parse_cdx=False)
    list(source.iter_patents())

    assert source.get_cdx_for_patent(PATENT_ID) is None
