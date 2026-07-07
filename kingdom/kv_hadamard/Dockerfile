# kingdom/vllm:v56a5c3e-plus5-rev3-pr78-kv4-had
# = pr78-kv4 + QuaRot-style Hadamard KV-latent fold (kv_hadamard.py) wired
#   into the rev-3 hybrid loader's get_all_weights hook.
#
# The hook injection edits the IN-IMAGE hybrid_loader.py (madeby561's rev-3
# runtime overlay; our permission for that file is runtime-only modification,
# so the modified copy exists only in images, never in this repo).
#
# Gating: KV_HADAMARD=1 enables the fold; default OFF = the stream is
# returned untouched -> byte-identical behavior to pr78-kv4.
#
# Build (on the box):
#   docker build -t kingdom/vllm:v56a5c3e-plus5-rev3-pr78-kv4-had \
#       /mnt/cache/bench/hadamard-build/ctx
# CPU import-smoke (no GPU):
#   docker run --rm --entrypoint python3 kingdom/vllm:...-kv4-had \
#       -c "import kv_hadamard; print(kv_hadamard._selftest())"
#   docker run --rm --entrypoint python3 kingdom/vllm:...-kv4-had \
#       /opt/kv_hadamard/test_kv_hadamard.py

FROM kingdom/vllm:v56a5c3e-plus5-rev3-pr78-kv4

COPY kv_hadamard.py /opt/venv/lib/python3.12/site-packages/kv_hadamard.py
COPY test_kv_hadamard.py /opt/kv_hadamard/test_kv_hadamard.py

RUN python3 - <<'PYEOF'
import sys

p = "/opt/venv/lib/python3.12/site-packages/hybrid_loader.py"
src = open(p).read()
if "kv_hadamard" in src:
    print("kv_hadamard hook already present — nothing to do")
    sys.exit(0)
anchor = (
    "            def get_all_weights(self, model_config, model):\n"
    "                return _fp8_ne_transform(_oga(self, model_config, model))\n"
)
hook = (
    "            def get_all_weights(self, model_config, model):\n"
    "                w = _fp8_ne_transform(_oga(self, model_config, model))\n"
    "                try:\n"
    "                    import kv_hadamard\n"
    "                    w = kv_hadamard.maybe_wrap(w)\n"
    "                except ImportError:\n"
    "                    pass\n"
    "                return w\n"
)
assert src.count(anchor) == 1, (
    "hybrid_loader.py get_all_weights anchor not found exactly once — "
    "rev overlay changed, refusing to inject blindly"
)
open(p, "w").write(src.replace(anchor, hook))
print("kv_hadamard hook injected into hybrid_loader.py")
PYEOF
