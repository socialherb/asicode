"""Extract the ripgrep install scripts AS LANDED in this branch (probe only).

No copy-paste: the scripts executed by the workflow are read back from the
checked-in YAML, so the bytes executed here are the bytes that were committed.
"""
from __future__ import annotations

import pathlib

import yaml

WF = pathlib.Path(".github/workflows")
TARGETS = ("lint.yml", "release.yml")

sites = []
for name in TARGETS:
    cfg = yaml.safe_load((WF / name).read_text(encoding="utf-8"))
    for job, jb in (cfg.get("jobs") or {}).items():
        for st in jb.get("steps") or []:
            if isinstance(st, dict) and st.get("name") == "Install ripgrep":
                sites.append((name, str(job), str(st.get("run") or "")))

print("sites found:", [(n, j) for n, j, _ in sites])
assert len(sites) == 3, f"expected 2 (lint) + 1 (release) sites, got {len(sites)}"
bodies = {b for _, _, b in sites}
assert len(bodies) == 1, f"recipes drifted: {len(bodies)} distinct bodies"
body = bodies.pop()
for i, (n, j, _) in enumerate(sites):
    pathlib.Path(f"/tmp/site_{i}_{n}.sh").write_text(body, encoding="utf-8")
print("EXTRACT_OK sites=3 recipe_bytes=%d recipe_lines=%d" % (len(body.encode()), len(body.splitlines())))
