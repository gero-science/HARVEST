import asyncio
from types import SimpleNamespace

from pipeline.debug_artifacts import save_debug_data


class FakeExporter:
    calls = []

    async def async_export_chemistry_as_single_file(self, **kwargs):
        self.calls.append(("chemistry", kwargs))


def fake_exporter_factory():
    return FakeExporter()


def test_save_debug_data_noops_when_document_has_no_nodes(tmp_path):
    FakeExporter.calls = []
    document = SimpleNamespace(chemistry_nodes=[])

    asyncio.run(save_debug_data("USUNITTESTA1", document, tmp_path, exporter_factory=fake_exporter_factory))

    assert FakeExporter.calls == []


def test_save_debug_data_exports_chemistry(tmp_path):
    FakeExporter.calls = []
    document = SimpleNamespace(chemistry_nodes=[{"chemistry": 1}])

    asyncio.run(save_debug_data("USUNITTESTA1", document, tmp_path, exporter_factory=fake_exporter_factory))

    assert [call[0] for call in FakeExporter.calls] == ["chemistry"]
    assert FakeExporter.calls[0][1]["filename"] == "debug_chemistry.json"
