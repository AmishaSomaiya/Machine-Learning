from pathlib import Path
from typing import Optional

def get_homeworks_path() -> Path:
    # Trick to get homeworks
    # First check current directory
    hw_path: Optional[Path] = None
    if (Path(".") / "data").exists():
        hw_path = Path(".")
    else:
        cur_dir_parents = Path(".").absolute().parents
        for parent in cur_dir_parents:
            if (parent / "data").exists():
                hw_path = parent
                break

    if hw_path is None:
        print("Could not find dataset. Please run from within 446 hw folder.")
        exit(0)

    return hw_path
