import os


def ensure_dir(d: str) -> None:
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
