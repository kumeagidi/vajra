import random
import time

from typing import List, Optional, Dict
from threading import Thread
from queue import PriorityQueue

from sarathi.core.datatypes.sampling_params import SamplingParams
from sarathi.logger import init_logger
from sarathi.core.sequence_manager.engine_sequence_manager import EngineSequenceManager
from sarathi.core.datatypes.sequence import SamplerOutputs, Sequence, SequenceWithPriority
from sarathi.utils.threading_utils import synchronized

logger = init_logger(__name__)


class GlobalScheduler:
    def __init__(self, config, num_replicas, sequence_counter, ):
        logger.info(
            f"GlobalScheduler initialized with {num_replicas} replicas"
        )

        self.config = config
        self.num_replicas = num_replicas
        self.replica_llm_engine_mapping = {}
        self.seq_counter = sequence_counter
        self.seq_map = None
        self.new_seq_list = None

    def init_queue(self):
        pass

    def get_replica_queue(self, replica_id):
        pass

    def get_replica_queue_mapping(self):
        pass

    def get_seq_map(self):
        return self.seq_map

    def get_new_seq_list(self):
        return self.new_seq_list

    def set_replica_llm_engine(self, replica_id, replica_llm_engine):
        self.replica_llm_engine_mapping[replica_id] = replica_llm_engine

    def _assign_queue(self, seq: Sequence, replica_id : int):
        pass

    @synchronized
    def assign_seq_replica(self, seq : Sequence) -> None:
        pass

    def assign_replica(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        seq_id: Optional[str] = None,
    ):
        pass

    def get_num_unfinished_requests(self):
        pass

    def has_unfinished_requests(self):
        pass

    def get_replica_id(self):
        pass


class PullScheduler(GlobalScheduler):
    """
    PullScheduler is a global scheduler that assigns requests to a global request queue and the replicas pull requests from the queue.
    """

    def __init__(self, config, num_replicas, sequence_counter):
        super().__init__(config, num_replicas, sequence_counter)
        logger.info(f"PullScheduler initialized with {num_replicas} replicas")

    def init_queue(self):
        self.replica_queue_mapping = {"global": PriorityQueue()}
        self.seq_map: Dict[str, Sequence] = None
        self.new_seq_list = None

    def get_replica_queue_mapping(self):
        return self.replica_queue_mapping

    def get_replica_queue(self, replica_id):
        return self.replica_queue_mapping["global"]

    def assign_replica(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        seq_id: Optional[str] = None,
    ):
        pass

    def _assign_queue(self, seq, replica_id):
        wrapped_seq = SequenceWithPriority(seq.arrived_at, seq)
        self.replica_queue_mapping["global"].put(wrapped_seq)

    @synchronized
    def assign_seq_replica(self, seq : Sequence) -> None:
        self._assign_queue(seq, None)

    def get_num_unfinished_requests(self):
        return self.replica_queue_mapping["global"].qsize()

    def has_unfinished_requests(self):
        return not self.replica_queue_mapping["global"].empty()


class RoundRobinScheduler(GlobalScheduler):
    """
    RoundRobinScheduler is a global scheduler that assigns requests to replicas in a round-robin manner.
    """

    def __init__(self, config, num_replicas, sequence_counter):
        super().__init__(config, num_replicas, sequence_counter)
        self.current_replica_id = 0
        logger.info(f"RoundRobinScheduler initialized with {num_replicas} replicas")

    def init_queue(self):
        self.replica_queue_mapping = {
            replica_id: PriorityQueue() for replica_id in range(self.num_replicas)
        }

    def get_replica_queue_mapping(self):
        return self.replica_queue_mapping

    def get_replica_queue(self, replica_id):
        return self.replica_queue_mapping[replica_id]

    def _assign_queue(self, seq, replica_id):
        logger.info(f"[ROUND ROBIN SCHED] Assigning request to replica {replica_id}")
        wrapped_seq = SequenceWithPriority(seq.arrived_at, seq)
        self.replica_queue_mapping[replica_id].put(wrapped_seq)

    @synchronized
    def assign_seq_replica(self, seq: Sequence) -> None:
        replica_id = self.current_replica_id
        self._assign_queue(seq, replica_id)
        self.current_replica_id = (self.current_replica_id + 1) % self.num_replicas

    def assign_replica(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        seq_id: Optional[str] = None,
    ):
        pass

    def get_num_unfinished_requests(self):
        num_unfinished_requests = 0
        for replica_id in range(self.num_replicas):
            num_unfinished_requests += self.replica_queue_mapping[replica_id].qsize
        return num_unfinished_requests

    def has_unfinished_requests(self):
        for replica_id in range(self.num_replicas):
            if not self.replica_queue_mapping[replica_id].empty():
                return True
        return False
