"""Inspect a Jupyter notebook for data-path dependencies and import surface.

Used to confirm SOTA_Reproduction_Phase36.ipynb can be reproduced from only the
publicly uploaded inputs.
"""
import json, re, sys
from pathlib import Path

nb_path = Path(sys.argv[1])
nb = json.loads(nb_path.read_text(encoding='utf-8'))
deps = set()
imports = set()
exts = set()
shell_cmds = []
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source'])
    for m in re.finditer(r"""(open|read_csv|read_json|np\.load|torch\.load|Path)\(\s*['"]([^'"]+)['"]""", src):
        deps.add(m.group(2))
    for m in re.finditer(r'^\s*(?:from|import)\s+([\w\.]+)', src, re.M):
        imports.add(m.group(1).split('.')[0])
    for m in re.finditer(r"""['"]([^'"\s]+\.(?:csv|json|jsonl|pt|npy|npz|txt|yaml|yml|md|ipynb|pdf|parquet))['"]""", src):
        exts.add(m.group(1))
    for m in re.finditer(r'^\s*!\s*(.+)$', src, re.M):
        shell_cmds.append(m.group(1).strip())

print(f'notebook: {nb_path}')
print(f'cells: {len(nb["cells"])}  code-cells: {sum(1 for c in nb["cells"] if c["cell_type"]=="code")}')
print('--- explicit file/path references ---')
for d in sorted(deps):
    p = Path(d)
    exists = p.exists()
    print(f'  [{"OK" if exists else "MISSING"}] {d}')
print('--- ALL extension file refs (incl. comments / docstrings) ---')
for e in sorted(exts):
    print(f'  {e}')
print('--- top-level imports ---')
for i in sorted(imports):
    print(f'  {i}')
print('--- shell (!) commands ---')
for s in shell_cmds:
    print(f'  ! {s}')
