from dataclasses import dataclass
from typing import List, Optional, Tuple

from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import SamplerOutputs, Sequence


@dataclass
class StepInputs:
    """Input data for a single step of the model.

    Attributes:
        scheduler_output: The outputs from the scheduler for this step.
        new_seqs: A list of new sequences to add to the engine
        pending_step_outputs: A list of tuples of scheduler outputs and sampler outputs
    """

    scheduler_output: SchedulerOutputs
    new_seqs: Optional[List[Sequence]] = None
    pending_step_outputs: Optional[List[Tuple[SchedulerOutputs, SamplerOutputs]]] = None


@dataclass
class StepMicrobatchOuputs:
    schedule_id: int


@dataclass
class StepOutputs:
    schedule_id: int
    sampler_outputs: SamplerOutputs
