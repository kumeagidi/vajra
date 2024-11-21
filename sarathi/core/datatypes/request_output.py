from dataclasses import dataclass
from typing import List, Optional

from sarathi.core.datatypes.sequence import Sequence
from sarathi.core.datatypes.sequence_status import SequenceStatus


@dataclass
class RequestOutput:
    """The output data of a request to the LLM.

    Args:
        seq: The output sequences of the request.
        outputs: The output sequences of the request.
        finished: Whether the whole request is finished.
    """

    seq: Sequence
    finished: bool
    finish_reason: Optional[str] = None
    enabled_append_request_execution_stats: Optional[bool] = False

    @classmethod
    def from_seq(
        cls, seq: Sequence, enabled_append_request_execution_stats: bool = False
    ) -> "RequestOutput":
        return cls(
            seq,
            seq.is_finished(),
            SequenceStatus.get_finished_reason(seq.get_status()),
            enabled_append_request_execution_stats,
        )

    @property
    def text(self) -> str:
        if not self.finished:
            return self.seq.output_text

        if not self.enabled_append_request_execution_stats:
            return self.seq.output_text

        usage_text = f"""
        **Processed prefill of {len(self.prompt_token_ids)} tokens in {self.seq.state.e2e_prefill_time:.2f} seconds.**
        """

        return self.seq.output_text + usage_text

    @property
    def seq_id(self) -> int:
        return self.seq.seq_id

    @property
    def prompt(self) -> str:
        return self.seq.prompt

    @property
    def prompt_token_ids(self) -> List[int]:
        return self.seq.prompt_token_ids

    @property
    def token_ids(self) -> List[int]:
        return self.seq.output_token_ids
