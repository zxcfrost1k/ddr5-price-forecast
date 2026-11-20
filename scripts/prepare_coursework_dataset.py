"""Build reproducible DDR5 observation and exact seven-day forecast datasets.

Python 3 standard library only. Example:
python3 scripts/prepare_coursework_dataset.py --source data/ram_ddr5 --output outputs/coursework_dataset
"""
import argparse
import csv
import hashlib
import json
import math
import shutil
import sqlite3
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, pstdev

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def write(path,rows,fields=None):
    fields=fields or list(rows[0])
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT/'data/ram_ddr5')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/coursework_dataset')
    args=parser.parse_args();source=args.source;out=args.output
    out.mkdir(parents=True,exist_ok=True)
    selected=read(source/'selected_models.csv');products=read(source/'selected_products.csv');events=read(source/'selected_history.csv')
    passports=defaultdict(list)
    for p in products:passports[p['mpn'].upper()].append(p)
    model_rows=[];quarantine=[];models={}
    for q in selected:
        key=q['mpn'].upper();ps=passports[key];p=ps[0]
        reasons=[];vals={}
        for f in ['total_capacity_gb','module_capacity_gb','module_count','speed_mt_s','cl']:
            v={int(x[f]) for x in ps if x[f]}
            if len(v)>1:reasons.append('conflicting_'+f)
            vals[f]=next(iter(v)) if len(v)==1 else None
        for f in ['total_capacity_gb','module_capacity_gb','module_count','speed_mt_s']:
            if vals[f] is None:reasons.append('missing_'+f)
        if all(vals[f] for f in ['total_capacity_gb','module_capacity_gb','module_count']):
            if vals['module_capacity_gb']*vals['module_count']!=vals['total_capacity_gb']:reasons.append('capacity_mismatch')
        row={'model_id':key,'brand':p['brand'],'name':p['name'],'ddr':'DDR5',**vals,
             'cl_missing':int(vals['cl'] is None),'is_server':int(any(x['server_memory']=='True' for x in ps)),
             'retailer':'DNS','currency':'RUB','source_card_ids':q['card_ids'],
             'source_urls':';'.join(x['data_url'] for x in ps),'out_of_stock_fraction':float(q['out_of_stock_fraction']),
             'first_source_event':q['first_event_date'],'last_source_event':q['last_event_date'],
             'source_distinct_events':int(q['unique_tuples']),'included':int(not reasons),
             'exclusion_reason':';'.join(reasons)}
        model_rows.append(row)
        if reasons:quarantine.append(row)
        else:models[key]=row
    grouped=defaultdict(list)
    for e in events:grouped[e['model_key'].upper(),e['date']].append(e)
    observations=[];audit=[]
    for (key,day),rs in sorted(grouped.items()):
        states={r['in_stock']=='True' for r in rs}
        prices={int(r['price_rub']) for r in rs if r['price_rub']}
        reasons=[]
        if key not in models:reasons.append('model_characteristics_conflict')
        if False in states:reasons.append('mixed_stock' if True in states else 'out_of_stock')
        if any(not r['price_rub'] for r in rs):reasons.append('missing_price')
        if len(prices)>1:reasons.append('multiple_prices_same_day')
        if not prices:reasons.append('no_known_price')
        audit.append({'model_id':key,'date':day,'source_distinct_events':len(rs),'included':int(not reasons),'exclusion_reason':';'.join(reasons)})
        if reasons:continue
        p=models[key];price=next(iter(prices));assert price>0
        observations.append({'model_id':key,'date':day,'price_rub':price,'brand':p['brand'],
            **{f:p[f] for f in ['total_capacity_gb','module_capacity_gb','module_count','speed_mt_s','cl','cl_missing','is_server']},
            'price_per_gb_rub':round(price/p['total_capacity_gb'],6),'retailer':'DNS','currency':'RUB',
            'source_card_ids':';'.join(sorted({r['product_id'] for r in rs})),
            'source_urls':';'.join(sorted({r['data_url'] for r in rs})),
            'source_distinct_events':len(rs)})
    observations.sort(key=lambda r:(r['date'],r['model_id']))
    per_model=defaultdict(list)
    for r in observations:per_model[r['model_id']].append(r)
    features=[]
    numeric=['total_capacity_gb','module_capacity_gb','module_count','speed_mt_s','cl','cl_missing','is_server',
        'current_price_rub','lag1_price_rub','lag2_price_rub','days_since_previous_observation',
        'previous_observation_gap_days','price_change_from_previous_fraction','last3_observed_mean_rub',
        'last3_observed_std_rub','last3_observed_span_days','observed_count_to_date','year','month','weekday']
    categorical=['brand']
    for key,rs in sorted(per_model.items()):
        lookup={r['date']:r for r in rs}
        for i,r in enumerate(rs):
            day=date.fromisoformat(r['date']);prev=rs[i-1] if i else None;prev2=rs[i-2] if i>1 else None
            window=rs[max(0,i-2):i+1];prices=[x['price_rub'] for x in window]
            target_day=(day+timedelta(days=7)).isoformat()
            future=lookup.get(target_day)
            feature={
                'row_id':key+'@'+r['date'],'model_id':key,'feature_date':r['date'],'brand':r['brand'],
                **{f:r[f] for f in ['total_capacity_gb','module_capacity_gb','module_count','speed_mt_s','cl','cl_missing','is_server']},
                'current_price_rub':r['price_rub'],
                'lag1_price_rub':prev['price_rub'] if prev else None,
                'lag2_price_rub':prev2['price_rub'] if prev2 else None,
                'days_since_previous_observation':(day-date.fromisoformat(prev['date'])).days if prev else None,
                'previous_observation_gap_days':(date.fromisoformat(prev['date'])-date.fromisoformat(prev2['date'])).days if prev2 else None,
                'price_change_from_previous_fraction':r['price_rub']/prev['price_rub']-1 if prev else None,
                'last3_observed_mean_rub':mean(prices),'last3_observed_std_rub':pstdev(prices),
                'last3_observed_span_days':(day-date.fromisoformat(window[0]['date'])).days,
                'observed_count_to_date':i+1,'year':day.year,'month':day.month,'weekday':day.weekday(),
                # Metadata for audit only, not allowed in X.
                'lag1_date':prev['date'] if prev else None,'lag2_date':prev2['date'] if prev2 else None,
                'window_start_date':window[0]['date'],
                'target_date':target_day if future else None,
                'target_price_rub':future['price_rub'] if future else None,
            }
            features.append(feature)
    labeled=sorted([r for r in features if r['target_price_rub'] is not None],key=lambda r:(r['feature_date'],r['model_id']))
    val_start=labeled[int(len(labeled)*.70)]['feature_date']
    test_start=labeled[int(len(labeled)*.85)]['feature_date']
    supervised=[];purged=[]
    for r in labeled:
        if r['feature_date']<val_start:
            split='train' if r['target_date']<val_start else 'purged_train_boundary'
        elif r['feature_date']<test_start:
            split='validation' if r['target_date']<test_start else 'purged_validation_boundary'
        else:split='test'
        row={**r,'split':split}
        (purged if split.startswith('purged') else supervised).append(row)
    assert {r['split'] for r in supervised}=={'train','validation','test'}
    # The latest feature rows have no observed target exactly seven days later.
    last_rows=[next(r for r in reversed(features) if r['model_id']==key) for key in sorted(per_model)]
    latest=[{k:v for k,v in r.items() if k not in ['target_date','target_price_rub']} for r in last_rows]
    write(out/'models.csv',model_rows)
    write(out/'observations.csv',observations)
    write(out/'forecast_7d.csv',supervised)
    write(out/'latest_features.csv',latest)
    for split in ['train','validation','test']:write(out/(split+'.csv'),[r for r in supervised if r['split']==split])
    write(out/'quarantined_models.csv',quarantine,fields=list(model_rows[0]))
    write(out/'cleaning_audit.csv',audit)
    write(out/'purged_boundaries.csv',purged,fields=list(supervised[0]))
    config={'task':'predict an unambiguous observed in-stock price exactly 7 calendar days later',
        'target':'target_price_rub','numeric_features':numeric,'categorical_features':categorical,
        'feature_availability':'end of feature_date','horizon_days':7,
        'validation_start':val_start,'test_start':test_start,
        'excluded_from_features':[k for k in supervised[0] if k not in numeric+categorical],
        'split_method':'shared chronological boundaries, target must stay inside its split',
        'imputation':'fit numeric medians on training only; categorical unknown handling on training only',
        'feature_lags':'previous observed dates, not calendar-day lags',
        'static_metadata':'current product characteristics from 2026-11-17 snapshot; historical constancy not independently verified'}
    (out/'feature_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    stats={'snapshot_date':'2026-11-17','input_selected_models':len(selected),'input_distinct_events':len(events),
        'quarantined_models':len(quarantine),'clean_models':len(per_model),'clean_observations':len(observations),
        'first_observation':observations[0]['date'],'last_observation':observations[-1]['date'],
        'input_model_date_groups':len(grouped),'excluded_model_date_groups':sum(not r['included'] for r in audit),
        'exact_7d_pairs_before_split_purge':len(labeled),'purged_boundary_rows':len(purged),
        'forecast_rows':len(supervised),'forecast_models':len({r['model_id'] for r in supervised}),
        'missing_cl_models':sum(p['cl_missing'] for p in models.values()),
        'server_models':sum(p['is_server'] for p in models.values()),
        'validation_start':val_start,'test_start':test_start,'splits':{},
        'exclusion_reason_counts_overlapping':dict(Counter(reason for r in audit for reason in r['exclusion_reason'].split(';') if reason))}
    for split in ['train','validation','test']:
        rows=[r for r in supervised if r['split']==split]
        stats['splits'][split]={'rows':len(rows),'models':len({r['model_id'] for r in rows}),
            'feature_start':rows[0]['feature_date'],'feature_end':rows[-1]['feature_date'],
            'target_start':min(r['target_date'] for r in rows),'target_end':max(r['target_date'] for r in rows),
            'persistence_mae_rub':round(mean(abs(r['current_price_rub']-r['target_price_rub']) for r in rows),2)}
    train_ids={r['model_id'] for r in supervised if r['split']=='train'}
    stats['test_models_unseen_in_train']=len({r['model_id'] for r in supervised if r['split']=='test'}-train_ids)
    (out/'dataset_summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2))
    # SQLite stores types explicitly; CSV preserves the same cells and ISO dates.
    db_path=out/'ram_ddr5.sqlite'
    if db_path.exists():db_path.unlink()
    with sqlite3.connect(db_path) as db:
        for table,rows in [('models',model_rows),('observations',observations),('forecast_7d',supervised),('latest_features',latest)]:
            fields=list(rows[0]);types={}
            for f in fields:
                v=[r[f] for r in rows if r[f] is not None]
                types[f]='INTEGER' if v and all(isinstance(x,int) for x in v) else 'REAL' if v and all(isinstance(x,(int,float)) for x in v) else 'TEXT'
            db.execute('CREATE TABLE '+table+' ('+','.join('"'+f+'" '+types[f] for f in fields)+')')
            db.executemany('INSERT INTO '+table+' VALUES ('+','.join('?' for _ in fields)+')',[[r[f] for f in fields] for r in rows])
        db.execute('CREATE UNIQUE INDEX idx_obs ON observations(model_id,date)')
        db.execute('CREATE UNIQUE INDEX idx_forecast ON forecast_7d(row_id)')
    # Capture exact inputs used for a repeatable build, without re-downloading.
    inputs=out/'input_snapshot';inputs.mkdir(exist_ok=True)
    for name in ['selected_models.csv','selected_products.csv','selected_history.csv']:
        shutil.copyfile(source/name,inputs/name)
    manifest={'files':{str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.rglob('*')) if f.is_file() and f.suffix in ['.csv','.json','.sqlite'] and f.name not in ['checksums.json','verification.json','workbook_payload.json']},'source':'PCstonks public catalogue / DNS retail offers'}
    (out/'checksums.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
