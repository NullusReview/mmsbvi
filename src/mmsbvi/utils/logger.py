"""
Logging utilities for MMSBVI
MMSBVI日志工具

Provides consistent logging across the project.
提供项目中一致的日志记录。
"""

import logging
import sys
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console

# Console for rich output / Rich输出控制台
console = Console()

# Configure root logger / 配置根日志记录器
def setup_logger(
    name: str = "mmsbvi",
    level: str = "INFO",
    use_rich: bool = True,
) -> logging.Logger:
    """
    Setup logger with consistent formatting.
    设置具有一致格式的日志记录器。
    
    Args:
        name: Logger name / 日志记录器名称
        level: Logging level / 日志级别
        use_rich: Whether to use rich formatting / 是否使用rich格式
        
    Returns:
        logger: Configured logger / 配置好的日志记录器
    """
    logger = logging.getLogger(name)
    # 如果已经配置，无需重复 / Skip if already configured
    if logger.handlers:
        return logger
    
    # Set level / 设置级别
    logger.setLevel(getattr(logging, level.upper()))
    
    if use_rich and RichHandler is not None:
        # Rich handler for beautiful console output / Rich处理器用于美观的控制台输出
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            markup=True,
        )
        handler.setFormatter(
            logging.Formatter("%(message)s", datefmt="[%X]")
        )
    else:
        # Standard handler / 标准处理器
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
    
    logger.addHandler(handler)
    logger.propagate = False  # 避免重复打印
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get logger instance.
    获取日志记录器实例。
    
    Args:
        name: Logger name, defaults to module name / 日志记录器名称，默认为模块名
        
    Returns:
        logger: Logger instance / 日志记录器实例
    """
    if name is None:
        # Get caller's module name / 获取调用者的模块名
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get("__name__", "mmsbvi")
        else:
            name = "mmsbvi"
    
    # Ensure logger is setup / 确保日志记录器已设置
    if not logging.getLogger("mmsbvi").handlers:
        setup_logger()
    
    return logging.getLogger(name)


# Convenience functions / 便利函数
def log_info(message: str, **kwargs):
    """Log info message / 记录信息消息"""
    logger = get_logger()
    logger.info(message, **kwargs)


def log_warning(message: str, **kwargs):
    """Log warning message / 记录警告消息"""
    logger = get_logger()
    logger.warning(message, **kwargs)


def log_error(message: str, **kwargs):
    """Log error message / 记录错误消息"""
    logger = get_logger()
    logger.error(message, **kwargs)


def log_debug(message: str, **kwargs):
    """Log debug message / 记录调试消息"""
    logger = get_logger()
    logger.debug(message, **kwargs)