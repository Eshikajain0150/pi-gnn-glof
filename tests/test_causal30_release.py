from pathlib import Path
import importlib.util, tempfile, zipfile
spec=importlib.util.spec_from_file_location('verify_release',Path(__file__).parents[1]/'scripts'/'verify_causal30_release.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def test_frozen_release_integrity():
    repo=Path(__file__).parents[1]
    archive=repo/'evidence'/'causal30_v1_release.zip'
    if not archive.exists():
        spec2=importlib.util.spec_from_file_location('materialize',repo/'scripts'/'materialize_causal30_archive.py')
        mat=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(mat)
        mat.materialize(repo/'evidence'/'archive_parts', archive)
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(archive) as zf: zf.extractall(tmp)
        report=mod.verify(Path(tmp)/'causal30_v1')
    assert report['status']=='PASS'
    assert report['anchors']==43
    assert report['primary_rows']==20
    assert report['ablation_rows']==35
    assert report['all_node_predictions'] in {0,15120}
