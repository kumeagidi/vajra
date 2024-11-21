from typing import Optional, Any, List, Dict
from queue import Queue, PriorityQueue
from collections import deque

from sarathi.core.datatypes.sequence import Sequence
from sarathi.config import SystemConfig
from sarathi.engine.base_llm_engine import BaseLLMEngine
from sarathi.engine.pipeline_parallel_llm_engine import PipelineParallelLLMEngine

from sarathi.utils import Counter


class LLMEngine:

    @classmethod
    def from_system_config(
        cls,
        config: SystemConfig,
        sequence_counter: Optional[Counter] = None,
        sequence_waiting_list: Optional[PriorityQueue] = None,
        global_output_queue: Optional[Queue] = None,
        seq_map : Optional[Dict[str, Sequence]] = None,
        new_seq_global: Optional[List[Sequence]] = None,
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        if config.parallel_config.pipeline_parallel_size > 1:
            engine = PipelineParallelLLMEngine(config)
        else:
            engine = BaseLLMEngine(
                config, sequence_counter, sequence_waiting_list, global_output_queue
            )

        return engine