# cloud/utils/diagnostics.py
# Per-stage extraction counters for the EXTRACTION QUALITY REPORT (A8/A9/B1).
#
# Pure logging aid — it never changes pipeline output, function signatures, or
# the knowledge-package contract. Each stage creates one `ExtractionStats` per
# run, threads it through its parse path, and logs `report(stage)` at the end
# so discarded information is visible instead of a bare "STATUS: SUCCESS".


class ExtractionStats:
    """Tracks parse/accept/reject counters for one extraction stage run."""

    def __init__(self) -> None:
        # LLM responses received.
        self.requests = 0
        # Parse outcomes.
        self.valid = 0        # parsed on first strict json.loads()
        self.repaired = 0     # parsed after the safe escape repair
        self.failed = 0       # could not be parsed even after repair
        # Item-level outcomes.
        self.accepted = 0     # items that passed validation
        self.normalized = 0   # items accepted via type normalization
        self.rejected = 0     # items dropped by validation
        self.duplicates = 0   # duplicate items dropped
        # Reserved for a future retry policy (none exists in the pipeline today).
        self.retries = 0

    def record_parse(self, status: str) -> None:
        """Record one LLM-response parse outcome by status string.

        "partial" (json_repair salvaged a truncated response) is counted as a
        repair: the response did not parse as emitted, but content survived.
        """
        if status == "ok":
            self.valid += 1
        elif status in ("repaired", "partial"):
            self.repaired += 1
        else:
            self.failed += 1

    def report(self, stage: str) -> str:
        """Render the EXTRACTION QUALITY REPORT line for this stage."""
        total_parse = self.valid + self.repaired + self.failed
        ok_rate = (self.valid / total_parse) if total_parse else 0.0
        repair_rate = (self.repaired / total_parse) if total_parse else 0.0
        return (
            f"{stage} EXTRACTION QUALITY REPORT — "
            f"requests={self.requests} valid_json={self.valid} "
            f"repaired_json={self.repaired} failed_json={self.failed} "
            f"(parse_ok={ok_rate:.1%} repaired={repair_rate:.1%}) "
            f"accepted={self.accepted} normalized={self.normalized} "
            f"rejected={self.rejected} duplicates={self.duplicates} "
            f"retries={self.retries}"
        )
