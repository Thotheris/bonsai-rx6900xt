import subprocess
import sys
from common import PQ2, environment, server_argv

context = int(sys.argv[1]) if len(sys.argv) > 1 else 262144
argv = server_argv(PQ2, context, 8081)
print(" ".join(argv), flush=True)
raise SystemExit(subprocess.call(argv, env=environment()))
