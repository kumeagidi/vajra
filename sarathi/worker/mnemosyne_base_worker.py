from sarathi.core.sequence_manager.mnemosyne_worker_sequence_manager import (
    MnemosyneWorkerSequenceManager,
)
from sarathi.worker.base_worker import BaseWorker


class MnemosyneBaseWorker(BaseWorker):
    def _get_seq_manager_impl(self):
        return MnemosyneWorkerSequenceManager
