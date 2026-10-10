"""YAML 配置加载与项目路径解析。

相对路径统一以当前版本分支根目录为基准，而不是以调用命令时的工作目录为
基准。这样从仓库根目录或 scripts 目录启动都能得到相同文件位置。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# config.py 位于 src/utils，因此 parents[2] 固定指向当前版本分支根目录。
V1_ROOT = Path(__file__).resolve().parents[2]


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """读取 YAML 并要求根节点为映射，避免后续出现模糊的类型错误。"""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"configuration root must be a mapping: {config_path}")
    return config


def resolve_v1_path(value: str | Path) -> Path:
    """将配置中的相对路径解析为 v1 下的绝对路径。"""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = V1_ROOT / path
    return path.resolve()
