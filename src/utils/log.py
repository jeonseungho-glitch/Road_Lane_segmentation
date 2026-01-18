import logging
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "Road_Lane_segmentation",
    log_dir: Optional[str] = None,
    log_level: str = "INFO"
) -> logging.Logger:

    # 로그 디렉토리 설정
    if log_dir is None:
        log_dir = Path("logs")
    else:
        log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 로그 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"

    # 로거 설정
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # 핸들러 중복 추가 방지
    if logger.hasHandlers():
        logger.handlers.clear()

    # 파일 핸들러
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        '%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)

    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
