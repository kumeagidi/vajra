import logging
import os
import time

from typing import Optional, List
from queue import Queue

import ray
import wandb
from tqdm import tqdm

from sarathi import LLMEngine, SamplingParams
from sarathi.benchmark.config import BenchmarkConfig
from sarathi.benchmark.entities import Request
from sarathi.benchmark.request_generator import RequestGeneratorRegistry
from sarathi.benchmark.utils.random import set_seeds
from sarathi.config import ReplicaConfig, BaseGlobalSchedulerTypeConfig
from sarathi.metrics.metrics_store import MetricsStore
from sarathi.types import ReplicaResourceMapping, ResourceMapping
from sarathi.utils import get_ip
from sarathi.core.datatypes.sequence import SamplerOutputs, Sequence
from sarathi.utils import Counter
from sarathi.engine.multi_replica_llm_engine import MultiReplicaLLMEngine

logger = logging.getLogger(__name__)


class BenchmarkRunnerLauncher:

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

        replica_config = ReplicaConfig(0, self.config.output_dir)
        os.makedirs(replica_config.output_dir, exist_ok=True)

        set_seeds(self.config.seed)
        request_generator = RequestGeneratorRegistry.get(
            self.config.request_generator_config.get_type(),
            self.config.request_generator_config,
        )
        self.requests = request_generator.generate()

        self.config.metrics_config.wandb_project = None

        self.system_config = self.config.create_system_config(replica_config)
        self.system_config.num_replicas = self.config.num_replicas

        self.llm_engine = MultiReplicaLLMEngine(self.system_config)

        if wandb.run is not None:
            wandb.config.update(self.config.to_dict())

    def _get_input_params(
        self, request: Request, first_request_time: float
    ) -> SamplingParams:
        sampling_params = SamplingParams(
            ignore_eos=True,
            max_tokens=request.num_decode_tokens,
            temperature=0,
            top_p=1.0,
        )
        prompt_token_ids = [1] * request.num_prefill_tokens

        return {
            "prompt": None,
            "prompt_token_ids": prompt_token_ids,
            "sampling_params": sampling_params,
            "arrival_time": first_request_time + request.arrived_at,
        }

    def warmup(self) -> None:
        self.llm_engine.add_request(**self._get_input_params(self.requests[0], 0))

        is_completed = False
        while not is_completed:
            step_outputs = self.llm_engine.step()
            is_completed = step_outputs[0].finished

        self.llm_engine.reset_metrics()

    def _run(self) -> None:
        if self.config.enable_profiling:
            self.llm_engine.start_profiling()

        num_processed_requests = 0
        num_steps = 0

        pbar = tqdm(
            total=len(self.requests),
            desc=f"Total processed requests",
        )
        start_time = time.monotonic()

        # Run the engine.
        while num_processed_requests < len(self.requests):
            elapsed_time = time.monotonic() - start_time
            if elapsed_time > self.config.time_limit:
                break

            step_outputs = self.llm_engine.step()
            num_steps += 1

            for output in step_outputs:
                if output.finished:
                    num_processed_requests += 1
                    pbar.update(1)

        end_time = time.monotonic()
        pbar.close()

        logger.info(
            f"{num_processed_requests} requests processed and exited before completing all requests"
        )

        logger.info(
            f"Exiting after processing {len(self.requests)} ({num_steps} iterations), Total time taken: {end_time - start_time:.2f} seconds"
        )

        if self.config.enable_profiling:
            self.llm_engine.stop_profiling()

    def _add_requests(self) -> None:
        index = 0
        first_request_time = time.monotonic()
        while index < len(self.requests):
            request = self.requests[index]

            self.llm_engine.add_request(
                **self._get_input_params(request, first_request_time)
            )
            index += 1

    def run_benchmark(self) -> None:
        self.llm_engine.reset_metrics()
        self._add_requests()
        self.llm_engine.start_engine_execution()
        self._run()
        self.llm_engine.pull_worker_metrics()
        metric_store = self.llm_engine.get_metric_store()
        return metric_store

    def run(self):
        metric_store = self.run_benchmark()
        metric_store.plot()
        if wandb.run is not None:
            wandb.finish()