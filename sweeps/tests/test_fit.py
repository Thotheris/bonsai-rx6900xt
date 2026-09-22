import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fit import decide

def test_clean_load_fits():
    assert decide(exit_code=0, shared_bytes=0) == "fit"

def test_shared_memory_does_not_fit():
    assert decide(exit_code=0, shared_bytes=1) == "spill"

def test_nonzero_exit_does_not_fit():
    assert decide(exit_code=1, shared_bytes=0) == "exit"

if __name__ == "__main__":
    test_clean_load_fits()
    test_shared_memory_does_not_fit()
    test_nonzero_exit_does_not_fit()
    print("ok")
