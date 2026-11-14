"""Clear all cell outputs / execution counts of the given notebooks in place (no dependencies).
Used by .githooks/pre-commit so executed notebooks (which may print row-level data) never get committed.
"""
import json
import sys


def strip(path):
    raw = open(path, encoding='utf-8').read()
    nb = json.loads(raw)
    for c in nb.get('cells', []):
        if c.get('cell_type') == 'code':
            c['outputs'] = []
            c['execution_count'] = None
        c.get('metadata', {}).pop('execution', None)
    out = json.dumps(nb, indent=1, ensure_ascii=False) + '\n'
    if out != raw:
        open(path, 'w', encoding='utf-8', newline='\n').write(out)
        return True
    return False


if __name__ == '__main__':
    for p in sys.argv[1:]:
        if strip(p):
            print('stripped', p)
