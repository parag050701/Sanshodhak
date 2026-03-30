"""OATD discovery client using OAI-PMH endpoint."""
from typing import Optional

from .oai_pmh_client import OAIPMHClient


class OATDClient(OAIPMHClient):
    """Open Access Theses and Dissertations client."""

    def __init__(self):
        super().__init__(
            base_url="https://oatd.org/oai",
            source_name="oatd",
        )
