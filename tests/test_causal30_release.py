from pathlib import Path
import importlib.util
spec=importlib.util.spec_from_file_location('verify_release',Path(__file__).parents[1]/'scripts'/'verify_causal30_release.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def test_frozen_release_integrity():
    repo=Path(__file__).parents[1]
    evidence=repo/'evidence'/'causal30_v1'
    assert evidence.is_dir(), 'evidence/causal30_v1 is missing'
    report=mod.verify(evidence)
    assert report['status']=='PASS'
    assert report['anchors']==43
    assert report['primary_rows']==20
    assert report['ablation_rows']==35
    assert report['all_node_predictions'] in {0,15120}
