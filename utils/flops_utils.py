import re
from pathlib import Path
from typing import List, Dict, Any, Set

# ---- 1) parse calflops log into flat records --------------------------------
def _unit_mult(unit: str) -> float:
    if not unit: return 1.0
    u = unit.upper()
    return 1e3 if u.startswith('K') else 1e6 if u.startswith('M') else 1e9 if u.startswith('G') else 1.0

_HEADER_RE = re.compile(r'^(\s*)\(([^)]+)\):\s*([^(]+)\(')
_METRICS_RE = re.compile(
    r'(?P<params_val>[\d.]+)\s*(?P<params_unit>[KMG])?\s*=\s*[\d.]+%\s*Params,\s*'
    r'(?P<macs_val>[\d.]+)\s*(?P<macs_unit>[KMG])?MACs\s*=\s*[\d.]+%\s*MACs,\s*'
    r'(?P<flops_val>[\d.]+)\s*(?P<flops_unit>[KMG])?FLOPS\s*=\s*[\d.]+%\s*FLOPs'
)

def parse_calflops_log_text(text: str) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts:
    [{'path': 'backbone.stem', 'params': int, 'macs': int, 'flops': int}, ...]
    FLOPs/MACs/Params are returned as *counts* (not strings), in base units (ops).
    """
    lines = text.splitlines()
    stack, last_path, out = [], None, []
    for line in lines:
        m = _HEADER_RE.match(line)
        if m:
            indent, name, _cls = m.groups()
            level = len(indent) // 2  # calflops uses 2 spaces per level
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, name.strip()))
            last_path = '.'.join(n for _, n in stack)

            # leaf nodes sometimes include metrics on the same line
            mm = _METRICS_RE.search(line)
            if mm:
                gd = mm.groupdict()
                out.append({
                    'path': last_path,
                    'params': float(gd['params_val']) * _unit_mult(gd['params_unit']),
                    'macs':  float(gd['macs_val'])  * _unit_mult(gd['macs_unit']),
                    'flops': float(gd['flops_val']) * _unit_mult(gd['flops_unit']),
                })
            continue

        # parent blocks (e.g., StemBlock) put the metrics on the very next line
        mm = _METRICS_RE.search(line)
        if mm and last_path:
            gd = mm.groupdict()
            out.append({
                'path': last_path,
                'params': float(gd['params_val']) * _unit_mult(gd['params_unit']),
                'macs':  float(gd['macs_val'])  * _unit_mult(gd['macs_unit']),
                'flops': float(gd['flops_val']) * _unit_mult(gd['flops_unit']),
            })
            last_path = None
    return out

# ---- 2) group by your custom blocks -----------------------------------------
def _normalize_pattern(p: str):
    """
    Supports wildcards: '~~', '**', '.*', '*'
    'encoder.~~' -> ('encoder', True)
    'backbone.stem' -> ('backbone.stem', False)
    """
    base = p.strip().replace('.~~', '').replace('~~', '').replace('.*', '').replace('**', '').replace('*', '')
    if base.endswith('.'): base = base[:-1]
    return base, (base != p)

_RANGE_RE = re.compile(r'^\d+\-\d+$')
_INT_RE   = re.compile(r'^\d+$')

def _seg_match(pat_seg: str, key_seg: str) -> bool:
    # exact match
    if pat_seg == key_seg:
        return True
    # key has a range like "0-2", pattern is a single index like "0"
    if _RANGE_RE.match(key_seg) and _INT_RE.match(pat_seg):
        lo, hi = map(int, key_seg.split('-'))
        return lo <= int(pat_seg) <= hi
    # (optional) pattern is a range, key is a single index
    if _RANGE_RE.match(pat_seg) and _INT_RE.match(key_seg):
        lo, hi = map(int, pat_seg.split('-'))
        return lo <= int(key_seg) <= hi
    return False

def _is_prefix_path(pattern_base: str, key_path: str) -> bool:
    """
    True if key_path is inside pattern_base subtree, with range-aware segments.
    """
    if not pattern_base:
        return True
    psegs = pattern_base.split('.')
    ksegs = key_path.split('.')
    if len(psegs) > len(ksegs):
        return False
    for ps, ks in zip(psegs, ksegs):
        if not _seg_match(ps, ks):
            return False
    return True

def _is_strict_ancestor(a: str, b: str) -> bool:
    """
    Returns True if 'a' is a strict ancestor of 'b' (not equal)
    """
    if a == b:
        return False
    # Check if b starts with a followed by a dot
    return b.startswith(a + '.')

def flops_by_blocks(records: List[Dict[str, Any]],
                    blocks: List[List[str]],
                    metric: str = 'flops') -> List[float]:
    """
    blocks: e.g., [['backbone.stem'], ['backbone.stages'], ['encoder.~~'], ['decoder.~~']]
    returns: [float, float, ...] (same order), values are in raw ops (not 'G' units)
    """
    path2val = {r['path']: r[metric] for r in records}
    out = []
    
    for group in blocks:
        # Deduplicate patterns in the group first
        unique_patterns = []
        seen_patterns = set()
        for pat in group:
            if pat not in seen_patterns:
                unique_patterns.append(pat)
                seen_patterns.add(pat)
        
        # Collect all matching paths for this group
        candidates = set()
        for pat in unique_patterns:
            base, is_prefix = _normalize_pattern(pat)
            if base in path2val and not is_prefix:
                candidates.add(base)
            else:
                for p in path2val:
                    if _is_prefix_path(base, p):
                        candidates.add(p)
        
        # Remove descendants if their ancestors are already in the set
        pruned = set(candidates)
        for a in candidates:
            for b in candidates:
                if _is_strict_ancestor(a, b) and b in pruned:
                    pruned.remove(b)
        
        # Sum the values for this group
        group_total = sum(path2val[p] for p in pruned)
        out.append(group_total)
    
    return out

# ---- 3) convenience wrapper --------------------------------------------------
def blockwise_from_log_file(log_path: str, blocks: List[List[str]], metric='flops', unit='G') -> List[float]:
    recs = parse_calflops_log_text(Path(log_path).read_text())
    vals = flops_by_blocks(recs, blocks, metric=metric)
    mult = {'': 1.0, 'K': 1e-3, 'M': 1e-6, 'G': 1e-9}[unit]
    return [v * mult for v in vals]


# ---- 4) Module tracing utilities (unchanged) --------------------------------
import time, torch
from contextlib import contextmanager

def is_leaf(m):
    return len(list(m.children())) == 0

@contextmanager
def trace_module_order(model, only_leaf=True, cuda_sync=False):
    # Map module objects → dotted names
    name_of = {m: n for n, m in model.named_modules()}
    events, stack, handles = [], [], []

    def pre_hook(mod, inputs):
        if only_leaf and not is_leaf(mod):
            return
        if cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        # depth BEFORE pushing gives nice indentation
        events.append(("enter", name_of[mod], len(stack), t))
        stack.append(mod)

    def post_hook(mod, inputs, output):
        if only_leaf and not is_leaf(mod):
            return
        if cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        # pop the matching frame
        if stack and stack[-1] is mod:
            stack.pop()
        events.append(("exit", name_of[mod], len(stack), t))

    # Attach hooks to every module (we filter with only_leaf inside)
    for m in name_of.keys():
        handles.append(m.register_forward_pre_hook(pre_hook))
        handles.append(m.register_forward_hook(post_hook))
    try:
        yield events   # you'll run the model inside the with-block
    finally:
        for h in handles: h.remove()

def print_call_sequence(events, save_to=None):
    lines = []
    for kind, name, depth, _ in events:
        arrow = "→" if kind == "enter" else "↩"
        lines.append(f'{"  " * depth}{arrow} {name}')
    txt = "\n".join(lines)
    if save_to:
        with open(save_to, "w", encoding="utf-8") as f:
            f.write(txt)
    return txt