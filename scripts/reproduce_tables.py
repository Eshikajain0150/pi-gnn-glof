#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,statistics,tempfile,zipfile
from collections import defaultdict
from pathlib import Path

def num(x): return float(x)
def summarize(rows,key,metrics):
    groups=defaultdict(list)
    for r in rows: groups[r[key]].append(r)
    out=[]
    for name,rs in groups.items():
        row={key:name,'n_seeds':len(rs)}
        for metric in metrics:
            values=[num(r[metric]) for r in rs]
            row[f'{metric}_mean']=statistics.fmean(values)
            row[f'{metric}_sample_sd']=statistics.stdev(values)
        out.append(row)
    return out

def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,default=Path('evidence/causal30_v1'));p.add_argument('--archive',type=Path);p.add_argument('--output',type=Path,default=Path('reproduced/tables'));a=p.parse_args()
    tmp=None
    if a.archive:
        tmp=tempfile.TemporaryDirectory(); zipfile.ZipFile(a.archive).extractall(tmp.name); evidence=Path(tmp.name)/'causal30_v1'
    else: evidence=a.evidence
    with (evidence/'seed_metrics'/'per_seed_metrics.csv').open(newline='',encoding='utf-8') as f: main_rows=list(csv.DictReader(f))
    primary=summarize(main_rows,'variant',['test_pr_auc','test_roc_auc','test_brier','test_f1','test_recall','test_false_alarms','test_spatial_mean_rank'])
    write(a.output/'primary_five_seed_summary.csv',primary)
    with (evidence/'physics_ablations'/'per_seed_ablation_metrics.csv').open(newline='',encoding='utf-8') as f: ab_rows=list(csv.DictReader(f))
    ab=summarize(ab_rows,'ablation_id',['test_pr_auc','test_brier','test_f1','test_false_alarms','test_spatial_mean_rank'])
    write(a.output/'physics_ablation_five_seed_summary.csv',ab)
    print(f'wrote {len(primary)} primary and {len(ab)} ablation rows to {a.output}')
    if tmp: tmp.cleanup()
if __name__=='__main__':main()
