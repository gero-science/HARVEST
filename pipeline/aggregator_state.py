"""State container for Agent 2 aggregation."""

from collections import defaultdict
from dataclasses import dataclass, field

from .accounting import add_agent1_usage_for_patent, create_agent1_usage_store


@dataclass
class Agent1Payload:
    patent_id: str
    extracted_data: list
    total_chunks_in_patent: int
    patent_text: str
    agent1_usage: dict
    document: object
    worker_errors: dict
    error_detail: dict | None
    zip_file_path: str


@dataclass
class CompletedPatent:
    patent_id: str
    measures_list: list
    total_chunks_in_patent: int
    received_chunks: int
    patent_text: str
    document: object
    zip_file_path: str
    agent1_usage: dict
    # How many of this patent's chunks came back as an Agent 1 error rather
    # than as data. A failed chunk still counts towards ``received_chunks``, so
    # without this an all-errors patent is indistinguishable from a patent that
    # genuinely holds no bioactivity data.
    failed_chunks: int = 0


def parse_agent1_payload(patent_data) -> Agent1Payload:
    (
        patent_id,
        extracted_data,
        total_chunks_in_patent,
        patent_text,
        agent1_usage,
        document,
        worker_errors,
        error_detail,
        zip_file_path,
    ) = patent_data

    return Agent1Payload(
        patent_id=patent_id,
        extracted_data=extracted_data,
        total_chunks_in_patent=total_chunks_in_patent,
        patent_text=patent_text,
        agent1_usage=agent1_usage,
        document=document,
        worker_errors=worker_errors,
        error_detail=error_detail,
        zip_file_path=zip_file_path,
    )


@dataclass
class AggregatorState:
    pending_data: dict = field(default_factory=lambda: defaultdict(list))
    pending_chunks_count: dict = field(default_factory=lambda: defaultdict(int))
    failed_chunks_count: dict = field(default_factory=lambda: defaultdict(int))
    total_chunks_info: dict = field(default_factory=dict)
    patent_texts: dict = field(default_factory=dict)
    patent_documents: dict = field(default_factory=dict)
    patent_zip_files: dict = field(default_factory=dict)
    agent1_usage_per_patent: dict = field(default_factory=create_agent1_usage_store)

    def ingest_payload(self, patent_data) -> str:
        payload = patent_data if isinstance(patent_data, Agent1Payload) else parse_agent1_payload(patent_data)
        patent_id = payload.patent_id

        self.pending_data[patent_id].extend(payload.extracted_data)
        self.pending_chunks_count[patent_id] += 1
        if payload.error_detail:
            self.failed_chunks_count[patent_id] += 1
        self.total_chunks_info[patent_id] = payload.total_chunks_in_patent
        self.patent_texts[patent_id] = payload.patent_text
        self.patent_documents[patent_id] = payload.document
        if payload.zip_file_path:
            self.patent_zip_files[patent_id] = payload.zip_file_path

        add_agent1_usage_for_patent(self.agent1_usage_per_patent, patent_id, payload.agent1_usage)
        return patent_id

    def is_complete(self, patent_id) -> bool:
        return self.pending_chunks_count[patent_id] >= self.total_chunks_info[patent_id]

    def pop_completed_patent(self, patent_id) -> CompletedPatent:
        completed = CompletedPatent(
            patent_id=patent_id,
            measures_list=self.pending_data[patent_id],
            total_chunks_in_patent=self.total_chunks_info[patent_id],
            received_chunks=self.pending_chunks_count[patent_id],
            patent_text=self.patent_texts[patent_id],
            document=self.patent_documents[patent_id],
            zip_file_path=self.patent_zip_files.get(patent_id, ""),
            agent1_usage=self.agent1_usage_per_patent[patent_id],
            failed_chunks=self.failed_chunks_count.get(patent_id, 0),
        )

        del self.pending_data[patent_id]
        del self.pending_chunks_count[patent_id]
        self.failed_chunks_count.pop(patent_id, None)
        del self.total_chunks_info[patent_id]
        del self.patent_texts[patent_id]
        del self.patent_documents[patent_id]
        if patent_id in self.patent_zip_files:
            del self.patent_zip_files[patent_id]
        del self.agent1_usage_per_patent[patent_id]

        return completed

    def build_incomplete_patents(self) -> list[dict]:
        incomplete_patents = []
        for patent_id in list(self.pending_data.keys()):
            received_chunks = self.pending_chunks_count.get(patent_id, 0)
            expected_chunks = self.total_chunks_info.get(patent_id, 0)
            zip_file_path = self.patent_zip_files.get(patent_id, "")
            incomplete_patents.append({
                "patent_id": patent_id,
                "zip_file_path": zip_file_path,
                "reason": f"Not all chunks received: {received_chunks}/{expected_chunks}",
                "stage": "agent2_aggregator_pending_chunks",
            })
        return incomplete_patents
