"""Shared fixtures for BioactivityExtractor tests."""
import logging

import pytest

from patent_processor.data_structures import ChemistryNode, PatentDocument
from llm.extractor import BioactivityExtractor


def _make_agent(name: str = "tests.extractor") -> BioactivityExtractor:
    agent = BioactivityExtractor.__new__(BioactivityExtractor)
    agent.logger = logging.getLogger(name)
    return agent


def _make_document(*chemistry_nodes: ChemistryNode) -> PatentDocument:
    return PatentDocument(
        file_path="memory.zip",
        patent_id="USUNITTESTA1",
        chemistry_nodes=list(chemistry_nodes),
    )


@pytest.fixture
def make_agent():
    return _make_agent


@pytest.fixture
def make_document():
    return _make_document
