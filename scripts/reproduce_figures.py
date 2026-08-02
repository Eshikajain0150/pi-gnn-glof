#!/usr/bin/env python3
from __future__ import annotations
import argparse,tempfile,zipfile
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,default=Path('evidence/causal30_v1'));p.add_argument('--archive',type=Path);p.add_argument('--output',type=Path,default=Path('reproduced/figures'));a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    tmp=None
    if a.archive:
        tmp=tempfile.TemporaryDirectory(); zipfile.ZipFile(a.archive).extractall(tmp.name); evidence=Path(tmp.name)/'causal30_v1'
    else: evidence=a.evidence
    main_df=pd.read_csv(evidence/'seed_metrics'/'per_seed_metrics.csv')
    metrics=[('test_pr_auc','PR-AUC'),('test_brier','Brier score'),('test_f1','F1'),('test_spatial_mean_rank','Event-node rank')]
    order=['cnn','temporal_gru','plain_gnn','pi_gnn']
    for col,label in metrics:
        fig,ax=plt.subplots(figsize=(7.2,4.5))
        for seed,g in main_df.groupby('seed'):
            g=g.set_index('variant').loc[order]
            ax.plot(order,g[col],marker='o',linewidth=0.8,alpha=0.65,label=f'seed {seed}')
        means=main_df.groupby('variant')[col].mean().reindex(order)
        sds=main_df.groupby('variant')[col].std(ddof=1).reindex(order)
        ax.errorbar(order,means,yerr=sds,fmt='ko',capsize=4,label='mean ± sample SD')
        ax.set_ylabel(label);ax.grid(axis='y',alpha=.3);ax.legend(fontsize=8,ncol=2);fig.tight_layout();fig.savefig(a.output/f'primary_{col}.png',dpi=300);plt.close(fig)
    ab=pd.read_csv(evidence/'physics_ablations'/'per_seed_ablation_metrics.csv')
    order_ab=['reference_plain_gnn','reference_pi_gnn','physics_graph_no_physics_loss','knn_graph_full_physics_loss','physics_graph_continuity_only','physics_graph_flow_only','physics_graph_temperature_only']
    fig,ax=plt.subplots(figsize=(9,5.2))
    means=ab.groupby('ablation_id')['test_brier'].mean().reindex(order_ab);sds=ab.groupby('ablation_id')['test_brier'].std(ddof=1).reindex(order_ab)
    ax.errorbar(range(len(order_ab)),means,yerr=sds,fmt='o',capsize=4);ax.set_xticks(range(len(order_ab)),[x.replace('_','\n') for x in order_ab],fontsize=7);ax.set_ylabel('Brier score (lower is better)');ax.grid(axis='y',alpha=.3);fig.tight_layout();fig.savefig(a.output/'ablation_brier.png',dpi=300);plt.close(fig)
    chronology=pd.read_csv(evidence/'seed_predictions'/'test_event_node_predictions.csv');chronology['anchor_date']=pd.to_datetime(chronology['anchor_date'])
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
    for ax,(model,g) in zip(axes.flat,chronology.groupby('model',sort=False)):
        for seed,s in g.groupby('seed'):
            ax.plot(s['anchor_date'],s['probability'],marker='o',linewidth=.8,label=str(seed));ax.axhline(s['selected_threshold'].iloc[0],linestyle=':',linewidth=.5)
        ax.set_title(model);ax.set_ylim(0,1.05);ax.grid(axis='y',alpha=.25)
    fig.autofmt_xdate();fig.tight_layout();fig.savefig(a.output/'held_out_chronology.png',dpi=300);plt.close(fig)
    print(f'figures written to {a.output}')
    if tmp: tmp.cleanup()
if __name__=='__main__':main()
