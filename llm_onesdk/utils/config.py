from typing import Any, Dict, Optional

class Config:
    def __init__(self, initial_config: Optional[Dict[str, Any]] = None):
        self._config = initial_config or {}

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """设置配置值"""
        self._config[key] = value

    def update(self, new_config: Dict[str, Any]) -> None:
        """更新多个配置值"""
        self._config.update(new_config)

    def remove(self, key: str) -> None:
        """移除配置项"""
        self._config.pop(key, None)

    def clear(self) -> None:
        """清除所有配置"""
        self._config.clear()

    def __getitem__(self, key: str) -> Any:
        """允许使用字典样式访问配置"""
        return self._config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """允许使用字典样式设置配置"""
        self._config[key] = value

    def __contains__(self, key: str) -> bool:
        """检查配置是否存在"""
        return key in self._config

    def __repr__(self) -> str:
        """返回配置的字符串表示"""
        return f"Config({self._config})"

    @property
    def as_dict(self) -> Dict[str, Any]:
        """返回配置的字典副本"""
        return self._config.copy()

    def load_from_file(self, file_path: str) -> None:
        """从文件加载配置"""
        import json
        with open(file_path, 'r') as f:
            self._config.update(json.load(f))

    def save_to_file(self, file_path: str) -> None:
        """保存配置到文件"""
        import json
        with open(file_path, 'w') as f:
            json.dump(self._config, f, indent=2)

# 默认配置
DEFAULT_CONFIG = {
    'debug': False,
    'timeout': 30,
    'max_retries': 3,
    'retry_delay': 1,
}

# 全局配置实例
global_config = Config(DEFAULT_CONFIG)