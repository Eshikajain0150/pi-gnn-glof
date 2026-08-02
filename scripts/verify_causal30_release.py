#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, math, tempfile, zipfile
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

EXPECTED_SEEDS={7,21,42,84,123}
EXPECTED_MODELS={'cnn','temporal_gru','plain_gnn','pi_gnn'}
EXPECTED_ABLATIONS={
'reference_plain_gnn','reference_pi_gnn','physics_graph_no_physics_loss',
'knn_graph_full_physics_loss','physics_graph_continuity_only',
'physics_graph_flow_only','physics_graph_temperature_only'}

def fail(msg:str)->None: raise AssertionError(msg)
def parse_dt(value:str)->datetime: return datetime.fromisoformat(value.replace('Z','+00:00'))

def verify(root:Path)->dict:
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    if protocol['protocol_id']!='CAUSAL_30DAY_GLOF_FORECAST_V1_1': fail('wrong protocol id')
    records=[json.loads(x) for x in (root/'anchor_manifest.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    if len(records)!=43: fail(f'expected 43 anchors, found {len(records)}')
    split=Counter(r['split'] for r in records)
    if split!=Counter({'train':27,'validation':7,'test':9}): fail(f'wrong split counts: {split}')
    labels=Counter(int(float(r['hazard_label'])) for r in records)
    if labels!=Counter({0:29,1:14}): fail(f'wrong label counts: {labels}')
    years=defaultdict(set)
    for r in records:
        years[r['split']].add(int(r['anchor_year']))
        anchor=date.fromisoformat(r['anchor_date'])
        if r['event_node_indices']!=[62]: fail(f"wrong event node at {r['anchor_date']}")
        for d in r['optical_acquisition_dates']:
            if parse_dt(d).date()>anchor: fail(f"future optical input at {r['anchor_date']}: {d}")
        for d in r['thermal_slot_dates']:
            if date.fromisoformat(d)>anchor: fail(f"future thermal input at {r['anchor_date']}: {d}")
        if any(int(y)>=anchor.year for y in r['velocity_source_years']): fail(f"current/future velocity at {r['anchor_date']}")
    if years!={'train':{2019,2020},'validation':{2021},'test':{2022}}: fail(f'wrong chronology: {dict(years)}')

    with (root/'seed_metrics'/'per_seed_metrics.csv').open(newline='',encoding='utf-8') as f: main=list(csv.DictReader(f))
    if len(main)!=20: fail(f'expected 20 primary seed-model rows, found {len(main)}')
    if {int(r['seed']) for r in main}!=EXPECTED_SEEDS: fail('primary seeds mismatch')
    if {r['variant'] for r in main}!=EXPECTED_MODELS: fail('primary models mismatch')
    if len({(r['seed'],r['variant']) for r in main})!=20: fail('duplicate primary seed-model row')
    for r in main:
        for k in ['test_pr_auc','test_roc_auc','test_brier','test_f1','test_recall','test_false_alarms','test_spatial_mean_rank']:
            if not math.isfinite(float(r[k])): fail(f'non-finite {k}')

    with (root/'physics_ablations'/'per_seed_ablation_metrics.csv').open(newline='',encoding='utf-8') as f: ab=list(csv.DictReader(f))
    if len(ab)!=35: fail(f'expected 35 ablation rows, found {len(ab)}')
    if {int(r['seed']) for r in ab}!=EXPECTED_SEEDS: fail('ablation seeds mismatch')
    if {r['ablation_id'] for r in ab}!=EXPECTED_ABLATIONS: fail('ablation configurations mismatch')
    if len({(r['seed'],r['ablation_id']) for r in ab})!=35: fail('duplicate ablation row')

    with (root/'seed_predictions'/'test_event_node_predictions.csv').open(newline='',encoding='utf-8') as f: pred=list(csv.DictReader(f))
    if len(pred)!=180: fail(f'expected 180 event-node predictions, found {len(pred)}')
    dates=sorted({r['anchor_date'] for r in pred})
    expected_dates=['2022-03-01','2022-03-09','2022-03-17','2022-03-25','2022-04-02','2022-04-10','2022-04-18','2022-04-26','2022-05-04']
    if dates!=expected_dates: fail(f'held-out dates mismatch: {dates}')
    if {int(r['seed']) for r in pred}!=EXPECTED_SEEDS or {r['model'] for r in pred}!=EXPECTED_MODELS: fail('prediction seed/model mismatch')
    if any(int(r['event_node_index'])!=62 for r in pred): fail('event-node prediction index mismatch')
    expected_labels={'2022-03-01':0,'2022-03-09':0,'2022-03-17':0,'2022-03-25':0,'2022-04-02':0,'2022-04-10':1,'2022-04-18':1,'2022-04-26':1,'2022-05-04':1}
    for r in pred:
        if int(r['label'])!=expected_labels[r['anchor_date']]: fail('held-out label mismatch')
        p=float(r['probability']); t=float(r['selected_threshold'])
        if not (0<=p<=1 and 0<=t<=1): fail('probability/threshold outside [0,1]')
        if int(r['alert']) != int(p>=t): fail('alert does not match threshold')

    all_node_total=0
    for model in EXPECTED_MODELS:
        p=root/'seed_predictions'/f'test_all_node_predictions_{model}.csv'
        if not p.exists():
            continue
        with p.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
        if len(rows)!=3780: fail(f'{model}: expected 3780 all-node rows, found {len(rows)}')
        if {int(r['node_index']) for r in rows}!=set(range(84)): fail(f'{model}: node index mismatch')
        if sum(int(r['is_event_node']) for r in rows)!=45: fail(f'{model}: event-node marker count mismatch')
        all_node_total += len(rows)

    checksum_path=root/'checksums.sha256'
    expected={}
    for line in checksum_path.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        digest, rel=line.split('  ',1); expected[rel]=digest
    for rel,digest in expected.items():
        actual=hashlib.sha256((root/rel).read_bytes()).hexdigest()
        if actual!=digest: fail(f'checksum mismatch: {rel}')

    prohibited_tokens=('125_scene','125-scene','rerun_125')
    for path in root.rglob('*'):
        if path.is_file() and path.name not in {'claim_ledger.md'}:
            if any(tok in path.name.lower() for tok in prohibited_tokens): fail(f'prohibited rerun file: {path}')

    return {'status':'PASS','protocol_id':protocol['protocol_id'],'anchors':43,'splits':dict(split),'labels':dict(labels),'primary_rows':20,'ablation_rows':35,'event_node_predictions':180,'all_node_predictions':all_node_total,'seeds':sorted(EXPECTED_SEEDS)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--evidence',type=Path,default=Path('evidence/causal30_v1')); ap.add_argument('--archive',type=Path); ap.add_argument('--json-output',type=Path)
    a=ap.parse_args()
    if a.archive:
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(a.archive) as zf: zf.extractall(tmp)
            report=verify(Path(tmp)/'causal30_v1')
    else:
        report=verify(a.evidence)
    text=json.dumps(report,indent=2,sort_keys=True); print(text)
    if a.json_output: a.json_output.parent.mkdir(parents=True,exist_ok=True); a.json_output.write_text(text+'\n',encoding='utf-8')
if __name__=='__main__': main()
