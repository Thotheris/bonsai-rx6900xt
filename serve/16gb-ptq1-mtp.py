import subprocess
import sys
from common import PTQ1_MTP, environment, server_argv

context = int(sys.argv[1]) if len(sys.argv) > 1 else 262144
argv = server_argv(PTQ1_MTP, context, 8083)
argv += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "1"]
print(" ".join(argv), flush=True)
raise SystemExit(subprocess.call(argv, env=environment()))
