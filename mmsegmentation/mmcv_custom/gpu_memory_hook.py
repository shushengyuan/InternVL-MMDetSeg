"""
GPU内存清理Hook - 用于在训练后清理GPU内存以避免评估时OOM
"""
import torch
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class GPUMemoryCleanupHook(Hook):
    """在训练后清理GPU内存的Hook"""
    
    def __init__(self, cleanup_interval=1, verbose=True):
        """
        Args:
            cleanup_interval (int): 每隔多少个iteration清理一次内存
            verbose (bool): 是否打印内存使用情况
        """
        self.cleanup_interval = cleanup_interval
        self.verbose = verbose
    
    def after_train_iter(self, runner):
        """在每次训练iteration后清理GPU内存"""
        if runner._iter % self.cleanup_interval == 0:
            if torch.cuda.is_available():
                if self.verbose:
                    allocated_before = torch.cuda.memory_allocated() / 1024**3
                    reserved_before = torch.cuda.memory_reserved() / 1024**3
                
                # 清理梯度
                if hasattr(runner, 'optimizer') and runner.optimizer is not None:
                    runner.optimizer.zero_grad()
                
                # 强制垃圾回收
                import gc
                gc.collect()
                
                # 清理CUDA缓存
                torch.cuda.empty_cache()
                
                if self.verbose:
                    allocated_after = torch.cuda.memory_allocated() / 1024**3
                    reserved_after = torch.cuda.memory_reserved() / 1024**3
                    # print(f"🧹 [MemoryCleanup] Iter {runner._iter}: "
                    #       f"{allocated_before:.2f}→{allocated_after:.2f} GB allocated, "
                    #       f"{reserved_before:.2f}→{reserved_after:.2f} GB reserved")
    
    def before_val_epoch(self, runner):
        """在验证开始前强制清理内存"""
        if torch.cuda.is_available():
            if self.verbose:
                allocated_before = torch.cuda.memory_allocated() / 1024**3
                reserved_before = torch.cuda.memory_reserved() / 1024**3
            
            # 强制清理所有缓存
            import gc
            gc.collect()
            
            # 更激进的内存清理
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # 等待所有CUDA操作完成
            
            # 尝试释放更多内存
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
            
            if self.verbose:
                allocated_after = torch.cuda.memory_allocated() / 1024**3
                reserved_after = torch.cuda.memory_reserved() / 1024**3
                print(f"🧹 [MemoryCleanup] Before validation: "
                      f"{allocated_before:.2f}→{allocated_after:.2f} GB allocated, "
                      f"{reserved_before:.2f}→{reserved_after:.2f} GB reserved")
            
            # 强制模型进入eval模式并清理训练状态
            if hasattr(runner, 'model') and runner.model is not None:
                runner.model.eval()
                # 清理可能的梯度缓存
                for param in runner.model.parameters():
                    if param.grad is not None:
                        param.grad.detach_()
                        param.grad = None
