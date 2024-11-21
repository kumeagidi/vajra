import copy
import math
import time
from functools import partial
from typing import Any, Dict, List, Optional, Tuple
from threading import Thread, Event
from queue import Queue, PriorityQueue

import ray
import zmq

import os
from sarathi import LLMEngine
from sarathi.config import ModelConfig, SystemConfig
from sarathi.core.datatypes.comm_info import CommInfo
from sarathi.core.datatypes.request_output import RequestOutput
from sarathi.core.datatypes.sampling_params import SamplingParams
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.scheduler.global_scheduler import (
    PullScheduler,
    RoundRobinScheduler,
)
from sarathi.core.datatypes.sequence import SamplerOutputs, Sequence, SequenceMetadata
from sarathi.core.datatypes.step_inputs import StepInputs
from sarathi.core.scheduler.scheduler_registry import SchedulerRegistry
from sarathi.core.sequence_manager.engine_sequence_manager import EngineSequenceManager
from sarathi.engine.ray_utils import RayWorker, initialize_cluster, ray
from sarathi.logger import init_logger
from sarathi.config import ReplicaConfig
from sarathi.types import ReplicaResourceMapping, ResourceMapping
from sarathi.metrics.constants import CpuOperationMetrics
from sarathi.metrics.cpu_timer import CpuTimer
from sarathi.metrics.metrics_store import MetricsStore
from sarathi.transformers_utils.tokenizer import get_tokenizer
from sarathi.utils import Counter, get_ip, unset_cuda_visible_devices
from sarathi.utils.threading_utils import synchronized, exit_on_error

logger = init_logger(__name__)

SCHEDULER_LOOP_DELAY = 0.01


class MultiReplicaLLMEngine:
    """A Multi-replica LLM engine that receives requests and generates texts across multiple replicas of the same model.

    Args:
        config; System Config: The system configuration for the engine.
        replica_resource_mapping: The resource mapping for the replicas. (Basically the ip and gpu id for each replica)
    """

    def __init__(
        self,
        config: SystemConfig,
        replica_resource_mapping: Optional[ReplicaResourceMapping] = None,
    ) -> None:
        logger.info(
            "Initializing a Multi Replica LLM engine with config: "
            f"model={config.model_config.model!r}, "
            f"dtype={config.model_config.dtype}, "
            f"tensor_parallel_size={config.parallel_config.tensor_parallel_size}, "
            f"pipeline_parallel_size={config.parallel_config.pipeline_parallel_size}, "
            f"seed={config.model_config.seed})"
        )

        logger.info(f"Replica resource mapping: {replica_resource_mapping}")

        self.config = config
        self._verify_args()

        ray.init(ignore_reinit_error=True)

        if replica_resource_mapping is None:
            self._validate_cluster_resources()
            self.replica_resource_mapping = self._get_replica_resource_mapping()
        else:
            self.replica_resource_mapping = replica_resource_mapping

        self.sequence_counter = Counter()

        self.global_scheduler_type = self.config.global_scheduler_config.scheduler_type

        self.aggregate_metric_store = self._create_aggregate_metric_store()

        self.global_output_queue: Queue = Queue()

        self.request_processing_queue = Queue()

        self.global_scheduler = self._get_global_scheduler()
        self.global_scheduler.init_queue()

        self._init_replica_engines()

        # start daemon thread for the assign_thread function
        self._setup_tokenizer_threads(num_tokenizers=self.config.num_replicas*2)

    def _setup_tokenizer_threads(self, num_tokenizers):
        self.tokenizer_pool = []
        for _ in range(num_tokenizers):
            tokenizer = get_tokenizer(
                self.config.model_config.model,
                trust_remote_code=self.config.model_config.trust_remote_code,
                revision=self.config.model_config.revision,
            )
            self.tokenizer_pool.append(tokenizer)

        # Spawn the daemon threads that keep pulling from the self.request_processing_queue
        self.tokenizer_queue_threads = []
        for i in range(num_tokenizers):
            tokenizer_queue_thread = Thread(target=self._tokenizer_queue_worker, args=(self.tokenizer_pool[i],), daemon=True)
            tokenizer_queue_thread.start()

    @exit_on_error
    def _tokenizer_queue_worker(self, tokenizer):
        while True:
            prompt,sampling_params,prompt_token_ids,arrival_time,seq_id = self.request_processing_queue.get()
            self._tokenize_and_generate_sequence(tokenizer, prompt,sampling_params,prompt_token_ids,arrival_time,seq_id)


    def _tokenize_and_generate_sequence(
        self,
        tokenizer,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        seq_id: Optional[str] = None,
    ):
        try:
            if arrival_time is None:
                arrival_time = time.monotonic()

            if prompt_token_ids is None:
                prompt_token_ids = tokenizer.encode(prompt)

            if not seq_id:
                seq_id = str(next(self.sequence_counter))

            block_size = self.config.cache_config.block_size
            eos_token_id = tokenizer.eos_token_id

            seq = Sequence(
                seq_id,
                prompt,
                prompt_token_ids,
                block_size,
                eos_token_id,
                arrival_time,
                sampling_params,
            )

            self.global_scheduler.assign_seq_replica(seq)
        except Exception as e:
            logger.error(f"Error in tokenizing and generating sequence: {e}")


    def _init_replica_engines(self):
        self.replica_llm_engine_mapping = {}
        for replica_id in range(self.config.num_replicas):

            replica_config = ReplicaConfig(
                replica_id=replica_id,
                output_dir=self.config.replica_config.output_dir,
                resource_mapping=self.replica_resource_mapping[replica_id],
            )
            os.makedirs(replica_config.output_dir, exist_ok=True)

            system_config = copy.deepcopy(self.config)
            system_config.replica_config = replica_config
            # Spawn a replica LLM engine for each replica in their own thread
            # self._spawn_replica_engine(replica_id, system_config, self.sequence_counter, self.global_scheduler.get_replica_queue(replica_id), self.global_output_queue, self.global_scheduler.get_seq_map())
            replica_thread = Thread(
                target=self._spawn_replica_engine,
                args=(
                    replica_id,
                    system_config,
                    self.sequence_counter,
                    self.global_scheduler.get_replica_queue(replica_id),
                    self.global_output_queue,
                    self.global_scheduler.get_seq_map(),
                    self.global_scheduler.get_new_seq_list(),
                ),
            )
            replica_thread.start()

            # Wait for the replica thread to finish
            replica_thread.join()

    def _get_global_scheduler(self):
        if self.global_scheduler_type == "pull":
            return PullScheduler(
                self.config,
                self.config.num_replicas,
                self.sequence_counter
            )
        elif self.global_scheduler_type == "round_robin":
            return RoundRobinScheduler(
                self.config,
                self.config.num_replicas,
                self.sequence_counter
            )
        else:
            raise ValueError(
                f"Unknown global scheduler type: {self.global_scheduler_type}"
            )

    def _spawn_replica_engine(
        self,
        replica_id,
        system_config,
        sequence_counter,
        replica_llm_engine_queue,
        global_output_queue,
        seq_map,
        new_seq_global,
    ):
        replica_llm_engine = LLMEngine.from_system_config(
            system_config,
            sequence_counter,
            replica_llm_engine_queue,
            global_output_queue,
        )

        self.global_scheduler.set_replica_llm_engine(replica_id, replica_llm_engine)

        self.replica_llm_engine_mapping[replica_id] = replica_llm_engine

    def _validate_cluster_resources(self):

        num_replicas = self.config.num_replicas
        num_gpus_required = num_replicas * self.config.parallel_config.world_size

        available_resources = ray.available_resources()

        assert (
            available_resources["GPU"] >= num_gpus_required
        ), f"Insufficient GPUs. Required: {num_gpus_required}, Available: {available_resources['GPU']}"

    def _get_replica_resource_mapping(self) -> ReplicaResourceMapping:

        cluster_resources_keys = list(ray.available_resources().keys())
        num_gpus = ray.available_resources()["GPU"]
        ip_addresses = [
            x
            for x in cluster_resources_keys
            if x.startswith("node:") and x != "node:__internal_head__"
        ]

        runner_ip = f"node:{get_ip()}"

        ip_addresses.remove(runner_ip)
        ip_addresses.insert(0, runner_ip)

        num_nodes = len(ip_addresses)
        assert num_nodes > 0, "No nodes found in the cluster"
        assert num_gpus > 0, "No GPUs found in the cluster"
        assert (
            num_gpus % num_nodes == 0
        ), f"Number of GPUs ({num_gpus}) is not a multiple of number of nodes ({num_nodes})"
        num_gpus_per_node = int(num_gpus // num_nodes)
        num_replicas = self.config.num_replicas
        num_gpus_per_replica = self.config.parallel_config.world_size

        assert (
            num_gpus >= num_replicas * num_gpus_per_replica
        ), f"Insufficient GPUs. Required: {num_replicas * num_gpus_per_replica}, Available: {num_gpus}"

        replica_resource_mapping = []

        available_gpus = []
        for ip_address in ip_addresses:
            for gpu_id in reversed(range(num_gpus_per_node)):
                available_gpus.append((ip_address, gpu_id))

        for _ in range(num_replicas):
            resource_mapping = []
            for _ in range(num_gpus_per_replica):
                resource_mapping.append(available_gpus.pop(0))
            replica_resource_mapping.append(resource_mapping)

        logger.info(f"Replica resource mapping: {replica_resource_mapping}")

        return replica_resource_mapping

    def _validate_parallel_config(self) -> None:
        assert self.config.parallel_config.pipeline_parallel_size == 1

    def _verify_args(self) -> None:
        self._validate_parallel_config()
        self.config.model_config.verify_with_parallel_config(
            self.config.parallel_config
        )

    def add_request(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        seq_id: Optional[str] = None,
    ) -> None:
        """
        Add a new request to a replica LLM engine
        The entire Request allocation is abstracted out to the Global Scheduler
        """
        self.request_processing_queue.put(
            (
                prompt,
                sampling_params,
                prompt_token_ids,
                arrival_time,
                seq_id,
            )
        )

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.global_scheduler.get_num_unfinished_requests()

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.global_scheduler.get_num_unfinished_requests()

    def get_metrics_store(self, replica_id: int) -> MetricsStore:
        return self.replica_llm_engine_mapping[replica_id].get_metrics_store()

    def pull_worker_metrics(self, replica_id: int) -> None:
        self.replica_llm_engine_mapping[replica_id].pull_worker_metrics()

    def _reset_replice_metrics(self, replica_id: int) -> None:
        self.replica_llm_engine_mapping[replica_id].reset_metrics()

    def reset_metrics(self) -> None:
        for replica_id in range(self.config.num_replicas):
            self._reset_replice_metrics(replica_id)

    def start_profiling(self):
        for replica_id in range(self.config.num_replicas):
            self.replica_llm_engine_mapping[replica_id].start_profiling()

    def stop_profiling(self):
        for replica_id in range(self.config.num_replicas):
            self.replica_llm_engine_mapping[replica_id].stop_profiling()

    def _pull_worker_metrics(self, replica_id: int) -> None:
        self.replica_llm_engine_mapping[replica_id].pull_worker_metrics()

    def pull_worker_metrics(self) -> None:
        for replica_id in range(self.config.num_replicas):
            self._pull_worker_metrics(replica_id)

    def _get_metric_store(self, replica_id: int) -> MetricsStore:
        return self.replica_llm_engine_mapping[replica_id].get_metric_store()

    def get_metric_store(self) -> MetricsStore:
        for replica_id in range(self.config.num_replicas):
            self.aggregate_metric_store.merge(self._get_metric_store(replica_id))
        return self.aggregate_metric_store

    def _create_aggregate_metric_store(self):
        replica_config = ReplicaConfig(
            replica_id=0,  # dummy replica id
            output_dir=self.config.replica_config.output_dir,
        )
        metrics_store = MetricsStore.get_or_create_instance(
            replica_config,
            self.config.model_config,
            self.config.metrics_config,
        )

        metrics_store.mark_initial_memory_profiling_done()

        return metrics_store

    def start_engine_execution(self):
        for replica_id in range(self.config.num_replicas):
            self.replica_llm_engine_mapping[replica_id].start_execution_loops()

    def step(self) -> List[RequestOutput]:
        """
        The step function drains the global output queue and returns the list of request outputs.
        The replica LLM engines push the request outputs to the global output queue.
        The replica level LLM engines continously keep calling the step function
        """
        output_list = []
        while not self.global_output_queue.empty():
            output_list.extend(self.global_output_queue.get())
        return output_list
