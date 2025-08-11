# --------------------------------------------------------
# Parameter Freeze Log Hook
# Copyright (c) 2024
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class ParameterFreezeLogHook(Hook):
    """Hook to log frozen/trainable parameters at the beginning of training.
    
    Args:
        log_interval (int): Interval of iterations to print log. Default: 1.
            Set to 1 to only print at the first iteration.
    """
    
    def __init__(self, log_interval=1):
        self.log_interval = log_interval
        self.logged = False  # 确保只打印一次
    
    def before_train_iter(self, runner):
        """Print parameter freeze status before the first training iteration."""
        if not self.logged and runner.iter % self.log_interval == 0:
            self._log_parameter_status(runner)
            self.logged = True
    
    def _log_parameter_status(self, runner):
        """Log the freeze status of model parameters."""
        model = runner.model
        if hasattr(model, 'module'):
            # For DistributedDataParallel
            model = model.module
        
        runner.logger.info("=" * 80)
        runner.logger.info("🔍 MODEL PARAMETER FREEZE STATUS")
        runner.logger.info("=" * 80)
        
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        
        # 统计各模块的参数状态
        module_stats = {}
        
        for name, param in model.named_parameters():
            total_params += param.numel()
            
            # 获取模块名称 (取前两级)
            module_name = '.'.join(name.split('.')[:2])
            
            if module_name not in module_stats:
                module_stats[module_name] = {
                    'total': 0,
                    'trainable': 0,
                    'frozen': 0,
                    'param_names': []
                }
            
            module_stats[module_name]['total'] += param.numel()
            
            if param.requires_grad:
                trainable_params += param.numel()
                module_stats[module_name]['trainable'] += param.numel()
                status = "✅ TRAINABLE"
            else:
                frozen_params += param.numel()
                module_stats[module_name]['frozen'] += param.numel()
                status = "❄️  FROZEN   "
            
            # 记录参数名称（只记录前5个，避免日志过长）
            if len(module_stats[module_name]['param_names']) < 5:
                module_stats[module_name]['param_names'].append((name, status, param.numel()))
        
        # 打印总体统计
        runner.logger.info(f"📊 OVERALL STATISTICS:")
        runner.logger.info(f"   Total Parameters:     {total_params:,}")
        runner.logger.info(f"   Trainable Parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        runner.logger.info(f"   Frozen Parameters:    {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        runner.logger.info("")
        
        # 按模块打印详细信息
        runner.logger.info(f"📋 MODULE-WISE BREAKDOWN:")
        for module_name, stats in sorted(module_stats.items()):
            total_module = stats['total']
            trainable_module = stats['trainable']
            frozen_module = stats['frozen']
            
            status_icon = "✅" if trainable_module > 0 else "❄️ "
            trainable_pct = trainable_module / total_module * 100 if total_module > 0 else 0
            
            runner.logger.info(f"   {status_icon} {module_name:<25} | "
                             f"Total: {total_module:>8,} | "
                             f"Trainable: {trainable_module:>8,} ({trainable_pct:>5.1f}%) | "
                             f"Frozen: {frozen_module:>8,}")
            
            # 显示前几个参数的详细状态
            for param_name, param_status, param_count in stats['param_names'][:3]:
                short_name = param_name.split('.')[-1]  # 只显示最后一级名称
                runner.logger.info(f"     └─ {param_status} {short_name:<20} ({param_count:,} params)")
            
            if len(stats['param_names']) > 3:
                runner.logger.info(f"     └─ ... and {len(stats['param_names'])-3} more parameters")
            
            runner.logger.info("")
        
        # 打印关键信息
        runner.logger.info("🎯 KEY INFORMATION:")
        
        # 检查backbone是否冻结
        backbone_frozen = all(not param.requires_grad 
                            for name, param in model.named_parameters() 
                            if name.startswith('backbone.'))
        
        if backbone_frozen:
            runner.logger.info("   🧊 Backbone is COMPLETELY FROZEN")
        else:
            runner.logger.info("   🔥 Backbone contains TRAINABLE parameters")
        
        # 检查decode_head是否可训练
        decode_head_trainable = any(param.requires_grad 
                                  for name, param in model.named_parameters() 
                                  if name.startswith('decode_head.'))
        
        if decode_head_trainable:
            runner.logger.info("   🎯 Decode Head is TRAINABLE")
        else:
            runner.logger.info("   ❄️  Decode Head is FROZEN")
        
        runner.logger.info("=" * 80) 