from pathlib import Path


def test_public_semantics_guard_script_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / 'scripts' / 'check_public_semantics.py').exists()
    assert (root / '.github' / 'workflows' / 'abiogenesis-canon.yml').exists()
