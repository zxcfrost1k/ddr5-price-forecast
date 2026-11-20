"""Independently verify prices, seven-day labels, past-only features and splits."""
import argparse
import csv
import hashlib
import json
import sqlite3
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, pstdev


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path,nargs='?',default=Path(__file__).resolve().parents[1]/'outputs/coursework_dataset');a=p.parse_args();folder=a.directory
    obs=read(folder/'observations.csv');forecast=read(folder/'forecast_7d.csv');cfg=json.loads((folder/'feature_config.json').read_text())
    source=read(folder/'input_snapshot/selected_history.csv');states=defaultdict(list)
    for r in source:states[r['model_key'].upper(),r['date']].append(r)
    lookup={};by_model=defaultdict(list)
    for r in obs:
        key=(r['model_id'],r['date']);assert key not in lookup
        lookup[key]=r;by_model[r['model_id']].append(r)
        originals=states[key]
        assert {x['in_stock'] for x in originals}=={'True'}
        assert {x['price_rub'] for x in originals}=={r['price_rub']}
        assert float(r['price_rub'])>0
        assert int(r['module_count'])*int(r['module_capacity_gb'])==int(r['total_capacity_gb'])
        assert abs(float(r['price_per_gb_rub'])-float(r['price_rub'])/int(r['total_capacity_gb']))<1e-5
    seen=set()
    for r in forecast:
        assert r['row_id'] not in seen;seen.add(r['row_id'])
        fd=date.fromisoformat(r['feature_date']);td=date.fromisoformat(r['target_date'])
        assert td-fd==timedelta(days=7)
        assert lookup[r['model_id'],r['target_date']]['price_rub']==r['target_price_rub']
        assert lookup[r['model_id'],r['feature_date']]['price_rub']==r['current_price_rub']
        # Rebuild the prefix as known at feature_date, excluding all later rows.
        prefix=[x for x in by_model[r['model_id']] if x['date']<=r['feature_date']]
        assert len(prefix)==int(r['observed_count_to_date'])
        for lag in [1,2]:
            if len(prefix)>lag:
                prev=prefix[-lag-1]
                assert r[f'lag{lag}_price_rub']==prev['price_rub']
                assert r[f'lag{lag}_date']==prev['date']<r['feature_date']
            else:assert not r[f'lag{lag}_price_rub'] and not r[f'lag{lag}_date']
        win=prefix[-3:];prices=[int(x['price_rub']) for x in win]
        assert abs(float(r['last3_observed_mean_rub'])-mean(prices))<1e-7
        assert abs(float(r['last3_observed_std_rub'])-pstdev(prices))<1e-7
        assert r['window_start_date']==win[0]['date']
        if r['split']=='train':assert r['target_date']<cfg['validation_start']
        elif r['split']=='validation':assert cfg['validation_start']<=r['feature_date'] and r['target_date']<cfg['test_start']
        else:assert r['split']=='test' and r['feature_date']>=cfg['test_start']
    assert not set(cfg['numeric_features']+cfg['categorical_features']) & set(cfg['excluded_from_features'])
    assert 'target_price_rub' not in cfg['numeric_features']+cfg['categorical_features']
    assert 'price_per_gb_rub' not in cfg['numeric_features']+cfg['categorical_features']
    for split in ['train','validation','test']:
        rows=read(folder/(split+'.csv'));assert {r['row_id'] for r in rows}=={r['row_id'] for r in forecast if r['split']==split}
    with sqlite3.connect(folder/'ram_ddr5.sqlite') as db:
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        assert db.execute('SELECT COUNT(*) FROM observations').fetchone()[0]==len(obs)
        assert db.execute('SELECT COUNT(*) FROM forecast_7d').fetchone()[0]==len(forecast)
    manifest=json.loads((folder/'checksums.json').read_text())
    for name,expected in manifest['files'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==expected,name
    report={'verified':True,'observations':len(obs),'forecast_rows':len(forecast),'checks':['unique model/date and row identifiers','prices match original source states','exact seven-day targets','features reconstructed from historical prefixes','chronological splits and purged target boundaries','feature allowlist excludes targets','split files reconcile','SQLite integrity and row counts','SHA-256 input and output integrity']}
    (folder/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
