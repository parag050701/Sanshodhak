"""
Source health aggregator — collects per-client SourceHealth and renders a
diagnostic report.  Import and call from your search engine or CLI:

    from .health_report import HealthReporter
    reporter = HealthReporter(client_registry)
    print(reporter.report())
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .base_client import BaseAPIClient

logger = logging.getLogger(__name__)


@dataclass
class HealthReporter:
    """Aggregates SourceHealth across all registered clients."""

    clients: Dict[str, "BaseAPIClient"] = field(default_factory=dict)

    def register(self, name: str, client: "BaseAPIClient"):
        self.clients[name] = client
        return self     # fluent

    def summaries(self) -> List[dict]:
        return [c.get_health() for c in self.clients.values()]

    def report(self, verbose: bool = False) -> str:
        lines = ["=" * 60, "  Sanshodhak — Source Health Report", "=" * 60]

        ok, degraded, broken, disabled = [], [], [], []

        for name, client in self.clients.items():
            h = client.health

            if h.last_failure_reason and h.last_failure_reason.value == "not_implemented":
                disabled.append(name)
            elif h.circuit_open:
                broken.append(name)
            elif h.success_rate < 0.5:
                degraded.append(name)
            else:
                ok.append(name)

        def _fmt(names: list, label: str) -> str:
            if not names:
                return ""
            return f"\n  [{label}] {', '.join(sorted(names))}"

        lines.append(_fmt(ok, "OK      "))
        lines.append(_fmt(degraded, "DEGRADED"))
        lines.append(_fmt(broken, "CIRCUIT "))
        lines.append(_fmt(disabled, "DISABLED"))
        lines = [l for l in lines if l]

        if verbose:
            lines.append("\n" + "-" * 60 + "  Details")
            for name, client in sorted(self.clients.items()):
                h = client.get_health()
                lines.append(
                    f"  {name:<25} req={h['total_requests']:>5}  "
                    f"fail={h['total_failures']:>4}  "
                    f"rate={h['success_rate']:>6}  "
                    f"last_err={h['last_failure_reason'] or '-':<25}  "
                    f"circuit={'OPEN' if h['circuit_open'] else 'closed'}"
                )

        lines.append("=" * 60)
        return "\n".join(lines)

    def healthy_clients(self) -> Dict[str, "BaseAPIClient"]:
        """Return only clients whose circuit is closed and are enabled."""
        return {
            name: client
            for name, client in self.clients.items()
            if not client.health.is_open()
            and client.health.last_failure_reason is not None
            and client.health.last_failure_reason.value != "not_implemented"
            or client.health.total_requests == 0  # never tried = tentatively healthy
        }
