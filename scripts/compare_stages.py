#!/usr/bin/env python3
"""
Compare metrics across different experiment stages (e.g., projection on/off, fusion on/off).
Generates detailed comparative report.

Usage:
  ./scripts/compare_stages.py stageD4_fusion/*.csv stageD4_no_fusion/*.csv
  ./scripts/compare_stages.py results/*/metrics.csv
"""

import sys
import os
import csv
from collections import defaultdict
from pathlib import Path

def parse_csv(filepath):
    """Parse metrics CSV and return list of row dicts."""
    rows = []
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                for key in row:
                    try:
                        if '.' in row[key]:
                            row[key] = float(row[key])
                        else:
                            row[key] = int(row[key])
                    except (ValueError, TypeError):
                        pass
                rows.append(row)
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
    return rows

def analyze_stage(label, csv_files):
    """Analyze a single stage across multiple CSV runs."""
    all_rows = []
    for csv_file in csv_files:
        rows = parse_csv(csv_file)
        all_rows.extend(rows)
    
    if not all_rows:
        return None
    
    # Aggregate statistics
    stats = {
        'files': len(csv_files),
        'total_frames': len(all_rows),
        'nn_mode_0': sum(1 for r in all_rows if r.get('nn_mode') == 0),
        'nn_mode_2': sum(1 for r in all_rows if r.get('nn_mode') == 2),
    }
    
    # Inlier ratio stats
    inlier_ratios = [r.get('nn_inlier_ratio', 0) for r in all_rows 
                    if isinstance(r.get('nn_inlier_ratio'), (int, float))]
    if inlier_ratios:
        stats['inlier_ratio_min'] = min(inlier_ratios)
        stats['inlier_ratio_max'] = max(inlier_ratios)
        stats['inlier_ratio_avg'] = sum(inlier_ratios) / len(inlier_ratios)
    
    # Surfel stats
    active_counts = [r.get('surfel_active', 0) for r in all_rows 
                     if isinstance(r.get('surfel_active'), int)]
    if active_counts:
        stats['surfel_active_min'] = min(active_counts)
        stats['surfel_active_max'] = max(active_counts)
        stats['surfel_active_avg'] = sum(active_counts) / len(active_counts)
    
    # Phi constraint stats
    checked = sum(r.get('phi_checked', 0) for r in all_rows 
                 if isinstance(r.get('phi_checked'), int))
    accepted = sum(r.get('phi_accepted', 0) for r in all_rows 
                  if isinstance(r.get('phi_accepted'), int))
    rejected = sum(r.get('phi_rejected', 0) for r in all_rows 
                  if isinstance(r.get('phi_rejected'), int))
    projected = sum(r.get('phi_projected', 0) for r in all_rows 
                   if isinstance(r.get('phi_projected'), int))
    
    if checked > 0:
        stats['phi_checked'] = checked
        stats['phi_accepted'] = accepted
        stats['phi_rejected'] = rejected
        stats['phi_projected'] = projected
        stats['phi_accept_rate'] = 100.0 * accepted / checked if checked > 0 else 0
        stats['phi_reject_rate'] = 100.0 * rejected / checked if checked > 0 else 0
        stats['phi_project_rate'] = 100.0 * projected / accepted if accepted > 0 else 0
    
    # Phi distribution
    phi_before_maxes = [r.get('phi_before_max', 0) for r in all_rows 
                       if isinstance(r.get('phi_before_max'), (int, float))]
    if phi_before_maxes:
        stats['phi_before_max_avg'] = sum(phi_before_maxes) / len(phi_before_maxes)
    
    return stats

def print_comparison(stages):
    """Print side-by-side comparison of stages."""
    print("\n" + "="*120)
    print("MULTI-STAGE REGRESSION COMPARISON")
    print("="*120 + "\n")
    
    # Header
    stage_names = list(stages.keys())
    print(f"{'Metric':<35}", end='')
    for name in stage_names:
        print(f" | {name:>25}", end='')
    print("\n" + "-"*120)
    
    # Frames
    print(f"{'Frames Processed':<35}", end='')
    for name in stage_names:
        if stages[name]:
            print(f" | {stages[name].get('total_frames', '-'):>25}", end='')
        else:
            print(f" | {'ERROR':>25}", end='')
    print()
    
    # NN Alignment modes
    print(f"{'NN Mode 0 (success)':<35}", end='')
    for name in stage_names:
        if stages[name]:
            print(f" | {stages[name].get('nn_mode_0', 0):>25}", end='')
        else:
            print(f" | {'-':>25}", end='')
    print()
    
    print(f"{'NN Mode 2 (keyframe_insertion)':<35}", end='')
    for name in stage_names:
        if stages[name]:
            print(f" | {stages[name].get('nn_mode_2', 0):>25}", end='')
        else:
            print(f" | {'-':>25}", end='')
    print()
    
    # Inlier ratio
    print(f"{'Inlier Ratio [min, max, avg]':<35}", end='')
    for name in stage_names:
        if stages[name] and 'inlier_ratio_min' in stages[name]:
            s = stages[name]
            val = f"[{s['inlier_ratio_min']:.4f}, {s['inlier_ratio_max']:.4f}, {s['inlier_ratio_avg']:.4f}]"
            print(f" | {val:>25}", end='')
        else:
            print(f" | {'-':>25}", end='')
    print()
    
    # Surfel counts
    print(f"{'Surfel Active [min, max, avg]':<35}", end='')
    for name in stage_names:
        if stages[name] and 'surfel_active_min' in stages[name]:
            s = stages[name]
            val = f"[{int(s['surfel_active_min']):6d}, {int(s['surfel_active_max']):6d}, {int(s['surfel_active_avg']):6.0f}]"
            print(f" | {val:>25}", end='')
        else:
            print(f" | {'-':>25}", end='')
    print()
    
    # Phi constraint stats
    print()
    print(f"{'Phi Checked (total)':<35}", end='')
    for name in stage_names:
        if stages[name] and 'phi_checked' in stages[name]:
            print(f" | {stages[name]['phi_checked']:>25}", end='')
        else:
            print(f" | {'-':>25}", end='')
    print()
    
    print(f"{'Phi Accepted (total, %)':<35}", end='')
    for name in stage_names:
        if stages[name] and 'phi_accepted' in stages[name]:
            val = f"{stages[name]['phi_accepted']} ({stages[name].get('phi_accept_rate', 0):.1f}%)"
            print(f" | {val:>25}", end='')
        else:
            print(f" | {'-':>25}", end='')
    print()
    
    print(f"{'Phi Rejected (total, %)':<35}", end='')
    for name in stage_names:
        if stages[name] and 'phi_rejected' in stages[name]:
            val = f"{stages[name]['phi_rejected']} ({stages[name].get('phi_reject_rate', 0):.1f}%)"
            print(f" | {val:>25}", end='')
        else:
            print(f" | {'-':>25}", end='')
    print()
    
    print(f"{'Phi Projected (total, % of accepted)':<35}", end='')
    for name in stage_names:
        if stages[name] and 'phi_projected' in stages[name]:
            val = f"{stages[name]['phi_projected']} ({stages[name].get('phi_project_rate', 0):.1f}%)"
            print(f" | {val:>25}", end='')
        else:
            print(f" | {'-':>25}", end='')
    print()
    
    print(f"{'Phi Max (avg |phi|_before)':<35}", end='')
    for name in stage_names:
        if stages[name] and 'phi_before_max_avg' in stages[name]:
            val = f"{stages[name]['phi_before_max_avg']:.4f} m"
            print(f" | {val:>25}", end='')
        else:
            print(f" | {'-':>25}", end='')
    print()
    
    print("\n" + "="*120 + "\n")

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    # Group CSV files by directory/stage
    stages = defaultdict(list)
    for pattern in sys.argv[1:]:
        if '*' in pattern:
            # Glob pattern
            from glob import glob
            for filepath in sorted(glob(pattern)):
                if filepath.endswith('.csv'):
                    # Try to infer stage name from path
                    stage_name = Path(filepath).parent.name
                    stages[stage_name].append(filepath)
        else:
            # Direct file path
            if os.path.isfile(pattern):
                stage_name = Path(pattern).parent.name
                stages[stage_name].append(pattern)
    
    if not stages:
        print("ERROR: No CSV files found", file=sys.stderr)
        sys.exit(1)
    
    # Analyze each stage
    results = {}
    for stage_name, csv_files in sorted(stages.items()):
        print(f"Analyzing {stage_name}: {len(csv_files)} file(s)...", file=sys.stderr)
        results[stage_name] = analyze_stage(stage_name, csv_files)
    
    # Print comparison
    print_comparison(results)
    
    # Summary
    print("Summary:")
    for stage_name in sorted(results.keys()):
        stats = results[stage_name]
        if stats:
            print(f"  {stage_name}: {stats['total_frames']} frames, "
                  f"NN success={stats.get('nn_mode_0', 0)}, "
                  f"phi_accept_rate={stats.get('phi_accept_rate', 0):.1f}%")

if __name__ == '__main__':
    main()
