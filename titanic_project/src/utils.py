"""유틸리티 함수"""
import os
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    """config.yaml 로드"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_dir, config_path)
    
    with open(full_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_project_root() -> str:
    """프로젝트 루트 경로 반환"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_dirs(path: str) -> None:
    """디렉토리가 없으면 생성"""
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
