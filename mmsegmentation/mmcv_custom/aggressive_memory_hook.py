"""
激进内存清理Hook - 专门用于解决评测时OOM问题
"""
import torch
import gc
import psutil
import os
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class AggressiveMemoryHook(Hook):
    """激进的内存清理Hook，用于评测时解决OOM问题"""
    
    def __init__(self, 
                 cleanup_interval=1, 
                 verbose=True,
                 force_gc=True,
                 reset_stats=True,
                 clear_grad=True):
        """
        Args:
            cleanup_interval (int): 每隔多少个iteration清理一次内存
            verbose (bool): 是否打印内存使用情况
            force_gc (bool): 是否强制垃圾回收
            reset_stats (bool): 是否重置内存统计
            clear_grad (bool): 是否清理梯度
        """
        self.cleanup_interval = cleanup_interval
        self.verbose = verbose
        self.force_gc = force_gc
        self.reset_stats = reset_stats
        self.clear_grad = clear_grad
    
    def _log_memory_info(self, stage=""):
        """记录内存使用情况"""
        if not self.verbose:
            return
            
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            print(f"🧠 [{stage}] GPU Memory: {allocated:.2f}GB allocated, "
                  f"{reserved:.2f}GB reserved, {max_allocated:.2f}GB max")
        
        # 系统内存
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print(f"💾 [{stage}] RAM: {memory_info.rss / 1024**3:.2f}GB")
    
    def _aggressive_cleanup(self):
        """激进的内存清理"""
        if torch.cuda.is_available():
            # 清理梯度
            if self.clear_grad:
                for param in torch.cuda._C._get_allocator_backend()._get_allocator_state():
                    if hasattr(param, 'grad') and param.grad is not None:
                        param.grad.detach_()
                        param.grad = None
            
            # 强制垃圾回收
            if self.force_gc:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # 重置内存统计
            if self.reset_stats:
                if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                    torch.cuda.reset_peak_memory_stats()
                if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
                    torch.cuda.reset_accumulated_memory_stats()
            
            # 尝试释放更多内存
            torch.cuda.empty_cache()
    
    def before_run(self, runner):
        """运行开始前清理内存"""
        self._log_memory_info("Before Run")
        self._aggressive_cleanup()
        self._log_memory_info("After Cleanup")
    
    def before_epoch(self, runner):
        """每个epoch开始前清理内存"""
        self._log_memory_info("Before Epoch")
        self._aggressive_cleanup()
    
    def after_epoch(self, runner):
        """每个epoch结束后清理内存"""
        self._log_memory_info("After Epoch")
        self._aggressive_cleanup()
    
    def before_iter(self, runner):
        """每个iteration开始前清理内存"""
        if runner._iter % self.cleanup_interval == 0:
            self._aggressive_cleanup()
    
    def after_iter(self, runner):
        """每个iteration结束后清理内存"""
        if runner._iter % self.cleanup_interval == 0:
            self._log_memory_info(f"Iter {runner._iter}")
            self._aggressive_cleanup()
            self._log_memory_info(f"After Iter {runner._iter}")
    
    def before_val_epoch(self, runner):
        """验证开始前清理内存"""
        self._log_memory_info("Before Validation")
        self._aggressive_cleanup()
    
    def after_val_epoch(self, runner):
        """验证结束后清理内存"""
        self._log_memory_info("After Validation")
        self._aggressive_cleanup()
    
    def before_test_epoch(self, runner):
        """测试开始前清理内存"""
        self._log_memory_info("Before Test")
        self._aggressive_cleanup()
    
    def after_test_epoch(self, runner):
        """测试结束后清理内存"""
        self._log_memory_info("After Test")
        self._aggressive_cleanup()
