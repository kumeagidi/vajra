from sarathi.core.sequence_manager.mnemosyne_worker_sequence_manager import (
    MnemosyneWorkerSequenceManager,
)
from sarathi.logger import init_logger
from sarathi.worker.pipeline_parallel_worker import PipelineParallelWorker

logger = init_logger(__name__)


class MnemosynePipelineParallelWorker(PipelineParallelWorker):
    def _verify_parallel_config(self) -> None:
        assert self.config.parallel_config.pipeline_parallel_size > 1

    def _get_seq_manager_impl(self):
        return MnemosyneWorkerSequenceManager
