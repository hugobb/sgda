from .executor import Executor
from dataclasses import dataclass, asdict
from omegaconf import MISSING
from typing import Optional, Callable, Any
from abc import ABC, abstractmethod
try:
    import submitit
except ImportError:
    print("Couldn't import submitit. Install it if you plan on running this code on the cluster.")


@dataclass
class SlurmConfig:
    log_folder: str = MISSING
    gpus_per_node: Optional[int] = None
    mem_by_gpu: Optional[int] = None
    partition: Optional[str] = None
    comment: Optional[str] = None
    gpu_type: Optional[str] = None
    time_in_min: Optional[int] = None
    nodes: Optional[int] = None
    cpus_per_task: Optional[int] = None
    slurm_array_parallelism: Optional[int] = None

    def merge(self, config):
        for key, value in asdict(self).items():
            if value is None or value == MISSING:
                new_value = config[key]
                setattr(self, key, new_value)


default_config = SlurmConfig(
    log_folder=MISSING,
    gpus_per_node=1,
    mem_by_gpu=16,
    partition="",
    comment="",
    gpu_type="",
    time_in_min=5,
    nodes=1,
    cpus_per_task=1,
    slurm_array_parallelism=1,
)


def create_slurm_executor(config: SlurmConfig = SlurmConfig()):

    executor = submitit.AutoExecutor(folder=config.log_folder)
    executor.update_parameters(
        slurm_partition=config.partition,
        slurm_comment=config.comment,
        slurm_constraint=config.gpu_type,
        slurm_time=config.time_in_min,
        timeout_min=config.time_in_min,
        nodes=config.nodes,
        cpus_per_task=config.cpus_per_task,
        tasks_per_node=config.gpus_per_node,
        gpus_per_node=config.gpus_per_node,
        mem_gb=config.mem_by_gpu * config.gpus_per_node,
        slurm_array_parallelism=config.slurm_array_parallelism,
    )

    return executor


class SlurmExecutor(Executor):
    def __init__(self, config: SlurmConfig = SlurmConfig()):
        self.executor = create_slurm_executor(config)

    def __call__(self, func: Callable[[Any], None], *args, **kwargs):
        job = self.executor.submit(func, *args, **kwargs)
        print("Launched job: %s" % (str(job.job_id)))


class ExecutorCallable(ABC):
    @abstractmethod
    def __call__(self, args: Any, resume=False):
        pass

    def checkpoint(self, args: Any, resume=False):
        return submitit.helpers.DelayedSubmission(self, args, resume=True)
