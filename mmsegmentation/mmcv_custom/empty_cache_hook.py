# --------------------------------------------------------
# Empty Cache Hook for CUDA Memory Management
# Copyright (c) 2024
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class CustomEmptyCacheHook(Hook):
    """Hook to empty CUDA cache periodically to prevent memory fragmentation.
    
    Args:
        interval (int): Interval of iterations to empty cache. Default: 100.
        before_run (bool): Whether to empty cache before training starts. Default: True.
        log_memory (bool): Whether to log memory usage. Default: True.
    """
    
    def __init__(self, interval=100, before_run=True, log_memory=True):
        self.interval = interval
        self.before_run_flag = before_run
        self.log_memory = log_memory
    
    def before_run(self, runner):
        """Empty cache before training starts."""
        if self.before_run_flag and torch.cuda.is_available():
            torch.cuda.empty_cache()
            if self.log_memory:
                self._log_memory_info(runner, "Training Start")
    
    def after_train_iter(self, runner):
        """Empty cache periodically during training."""
        if (torch.cuda.is_available() and 
            self.every_n_iters(runner, self.interval)):
            
            torch.cuda.empty_cache()
            
            if self.log_memory:
                self._log_memory_info(runner, f"Iter {runner.iter}")
    
    def _log_memory_info(self, runner, stage):
        """Log GPU memory information."""
        if not torch.cuda.is_available():
            return
            
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            cached = torch.cuda.memory_reserved(i) / 1024**3     # GB
            max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3  # GB
            
            runner.logger.info(
                f"🧹 [{stage}] GPU {i} Memory: "
                f"Allocated: {allocated:.2f}GB, "
                f"Cached: {cached:.2f}GB, "
                f"Max: {max_allocated:.2f}GB"
            ) 