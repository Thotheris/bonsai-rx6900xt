import subprocess
import sys
from common import DEFAULT_CONTEXT, MODEL_SPECS, environment, file_provenance, server_argv

context = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONTEXT
model = MODEL_SPECS["ptq1"]
file_provenance(model["path"], model["sha256"])
argv = server_argv(model["path"], context, model["port"], spec=model["default_spec"])
print(" ".join(argv), flush=True)
raise SystemExit(subprocess.call(argv, env=environment()))
