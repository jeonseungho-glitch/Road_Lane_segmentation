import os
import sys
from pathlib import Path

def setup_workspace(project_root: str = None) -> Path:
    """Jupyter 환경에서 프로젝트 루트 설정 및 작업 디렉토리 변경"""

    # Colab 환경 확인
    try:
        import google.colab
        is_colab = True
    except ImportError:
        is_colab = False

    # 환경에 따라 프로젝트 루트 설정
    if is_colab:
        if project_root is None:
            raise ValueError("Colab: project_root is required")
        root = Path(project_root)
    else:
        # Local: 항상 jupyter/ 상위 폴더 사용 (인자 무시)
        root = Path.cwd().parent

    # sys.path에 프로젝트 루트 추가
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # 작업 디렉토리 변경
    os.chdir(root)
    env = "Colab" if is_colab else "Local"
    print(f"Env: {env} | Root: {root}")
    return root
