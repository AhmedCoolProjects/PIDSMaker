# Run Inside DB

```bash
apptainer exec instance://postgres_instance psql -h localhost -U postgres -d postgres -At -c "select datname from pg_database order by datname;"
# cadets_e3
# cadets_e5
# clearscope_e3
# clearscope_e5
# optc_051
# optc_201
# optc_501
# postgres
# template0
# template1
# theia_e3
# theia_e5
```

```bash
PYTHONNOUSERSITE= ARGUS_PREVIEW_ONE_SENTENCE=1 ARGUS_PREVIEW_RANDOM=1 ARGUS_PREVIEW_SEED=42 ARGUS_PREVIEW_TAKE_N=1 PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH python - <<'PY'
from pidsmaker.config import get_runtime_required_args, get_yml_cfg
from pidsmaker.featurization.featurization_methods.featurization_argus import get_node2corpus

args = get_runtime_required_args(args=["argus", "CADETS_E3", "--cpu"])
cfg = get_yml_cfg(args)
node2corpus = get_node2corpus(cfg, splits=["train"])

if not node2corpus:
    print("No node sentences were constructed.")
else:
    node_id, tokens = next(iter(node2corpus.items()))
    print("PREVIEW_NODE=", node_id, sep="")
    print("PREVIEW_SENTENCE=", " ".join(tokens), sep="")
PY
```