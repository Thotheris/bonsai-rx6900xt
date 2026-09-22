def decide(exit_code, shared_bytes):
    if exit_code != 0:
        return "exit"
    if shared_bytes > 0:
        return "spill"
    return "fit"
