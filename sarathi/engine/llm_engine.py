from sarathi.config import SystemConfig
from sarathi.engine.base_llm_engine import BaseLLMEngine
from sarathi.engine.mnemosyne_base_llm_engine import MnemosyneBaseLLMEngine
from sarathi.engine.mnemosyne_pipeline_parallel_llm_engine import (
    MnemosynePipelineParallelLLMEngine,
)
from sarathi.engine.pipeline_parallel_llm_engine import PipelineParallelLLMEngine
from sarathi.types import SchedulerType


class LLMEngine:

    @classmethod
    def from_system_config(cls, config: SystemConfig) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        if config.scheduler_config.get_type() == SchedulerType.MNEMOSYNE:
            if config.parallel_config.pipeline_parallel_size > 1:
                engine = MnemosynePipelineParallelLLMEngine(config)
            else:
                engine = MnemosyneBaseLLMEngine(config)
        else:
            assert config.parallel_config.cache_parallel_size == 1

            if config.parallel_config.pipeline_parallel_size > 1:
                engine = PipelineParallelLLMEngine(config)
            else:
                engine = BaseLLMEngine(config)

        return engine
