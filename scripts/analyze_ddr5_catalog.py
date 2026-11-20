"""Analyze all downloaded DDR5 cards; retain full histories and exact filter logic."""
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from analyze_ram_pilot import Text, plain, EVENT_RE, assess

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/ram_ddr5'
RAW=DATA/'raw'
OUT=ROOT/'outputs/ddr5_catalog'
AS_OF=date(2026,11,17)


def save(name, rows):
    if not rows:
        (DATA/name).write_text('',encoding='utf-8-sig')
        return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (DATA/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def parse(pid):
    payload=(RAW/f'{pid}.json').read_bytes()
    meta=json.loads((RAW/f'{pid}.json.meta.json').read_text())
    assert hashlib.sha256(payload).hexdigest()==meta['sha256']
    item=json.loads(payload)
    fragment=item['html']
    text=Text(); text.feed(fragment)
    def after(label):
        try: return text.parts[text.parts.index(label)+1]
        except (ValueError,IndexError): return None
    specs=after('Характеристики') or ''
    name=item['name']
    mpn=specs.split(',')[0].strip().strip('[]')
    # A manufacturer part number must contain an ASCII letter and a digit.
    if not (re.search('[A-Za-z]',mpn) and re.search('[0-9]',mpn)) or mpn in ['DDR3','DDR4','DDR5'] or 'ГБ' in mpn:
        mpn=None
    if mpn is None:
        candidates=[t.strip('[](),') for t in name.split() if re.search('[A-Za-z]',t) and re.search('[0-9]',t)]
        candidates=[t for t in candidates if t not in ['DDR3','DDR4','DDR5'] and len(t)>=8]
        if len(candidates)==1: mpn=candidates[0]
    ddr_match=re.search(r'DDR[345]',name+' '+specs)
    ddr=ddr_match[0] if ddr_match else 'DDR5'
    module=re.search(r'(\d+)\s*ГБ\s*[xх×]\s*(\d+)\s*шт',specs)
    cap=re.search(r'(\d+)\s*ГБ\s*$',name)
    speed=re.search(r'(\d+)\s*МГц',specs)
    cl=re.search(r'CL\s*(\d+)|(\d+)\s*\(CL\)',specs)
    brand_name=re.sub(r'^(?:Серверная оперативная память|Оперативная память)\s*(?:DDR5\s*)?(?:(?:UDIMM|DIMM|SO-DIMM|SODIMM|CUDIMM|RDIMM)\s*)?', '',name)
    brand=brand_name.split()[0] if brand_name else None
    if brand_name.startswith('Silicon Power'): brand='Silicon Power'
    if brand_name.startswith('Team Group'): brand='Team Group'
    product={'product_id':pid,'mpn':mpn,'name':name,'brand':brand,'ddr':ddr,
        'ddr_explicit':bool(ddr_match),'total_capacity_gb':int(cap[1]) if cap else None,
        'module_capacity_gb':int(module[1]) if module else None,
        'module_count':int(module[2]) if module else None,
        'speed_mt_s':int(speed[1]) if speed else None,
        'cl':int(next(g for g in cl.groups() if g)) if cl else None,
        'server_memory':'Серверная' in name,
        'specifications_raw':specs,'retailer':'DNS','region':None,
        'data_url':meta['requested_url'],'page_url':f'https://pcstonks.com/catalog/{pid}',
        'retrieved_at_utc':meta['retrieved_at_utc'],'raw_sha256':meta['sha256']}
    events=[]; seen=set(); unmatched=[]
    for paragraph in re.findall(r'<p\b[^>]*>(.*?)</p>',fragment,re.S):
        t=plain(paragraph)
        if not re.match(r'\d{2}\.\d{2}\.\d{4}',t): continue
        m=EVENT_RE.fullmatch(t)
        if not m: unmatched.append(t); continue
        day,price,stock=m.groups()
        day=datetime.strptime(day,'%d.%m.%Y').date()
        assert day<=AS_OF,(pid,day)
        price=None if price=='Цена не указана' else int(re.sub(r'\D','',price))
        assert price is None or price>0
        key=(day.isoformat(),price,stock=='В наличии')
        events.append({'product_id':pid,'mpn':mpn,'source_order_desc':len(events)+1,
            'date':key[0],'price_rub':price,'in_stock':key[2],
            'same_tuple_repeat':key in seen,'original_text':t,'data_url':product['data_url']})
        seen.add(key)
    if unmatched: raise ValueError('Unparsed history: '+str(unmatched[:2]))
    return product,events


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((DATA/'manifest.json').read_text())
    products=[]; histories=[]; quality=[]; daily=[]; errors=[]
    for pid in manifest['product_ids']:
        try:
            p,e=parse(pid)
            products.append(p)
            if not e:
                errors.append({'product_id':pid,'error':'No historical events; filter cannot be calculated'})
                continue
            q,d,u=assess(p,e)
            q['passes_filter']=p['ddr']=='DDR5' and 5*q['out_of_stock_unique_records']<q['unique_tuples']
            q['mpn_known']=p['mpn'] is not None
            histories.extend(e);quality.append(q);daily.extend(d)
        except Exception as ex:
            errors.append({'product_id':pid,'error':str(ex)})
    # Treat duplicate manufacturer articles as one model. Union date/price/stock
    # states and recalculate the user's threshold, retaining per-card results too.
    groups=defaultdict(list); ep=defaultdict(list)
    for e in histories: ep[e['product_id']].append(e)
    for p in products:
        if ep[p['product_id']]:
            groups[p['mpn'].upper() if p['mpn'] else 'UNKNOWN:'+p['product_id']].append(p)
    model_quality=[]; model_events=[]
    for key,ps in groups.items():
        p=dict(ps[0]); p['product_id']=key
        rows=sorted([dict(e) for x in ps for e in ep[x['product_id']]],key=lambda e:(e['date'],-e['source_order_desc']),reverse=True)
        seen=set()
        for e in rows:
            state=(e['date'],e['price_rub'],e['in_stock'])
            e['same_tuple_repeat']=state in seen;seen.add(state)
        q,d,u=assess(p,rows)
        q['card_ids']=';'.join(x['product_id'] for x in ps)
        q['card_count']=len(ps)
        conflicts=[]
        for field in ['ddr','total_capacity_gb','module_count','speed_mt_s','cl']:
            values={x[field] for x in ps if x[field] is not None}
            if len(values)>1:
                conflicts.append(field+': '+','.join(map(str,sorted(values))))
        q['specification_conflicts']='; '.join(conflicts)
        q['server_memory']=any(x['server_memory'] for x in ps)
        q['mpn_known']=ps[0]['mpn'] is not None
        q['passes_filter']=p['ddr']=='DDR5' and q['mpn_known'] and 5*q['out_of_stock_unique_records']<q['unique_tuples']
        model_quality.append(q)
        for e in u:
            e['model_key']=key
            model_events.append(e)
    selected=[q for q in model_quality if q['passes_filter']]
    selected_keys={q['product_id'] for q in selected}
    selected_history=[e for e in model_events if e['model_key'] in selected_keys]
    selected_ids={i for q in selected for i in q['card_ids'].split(';')}
    summary={'as_of':AS_OF.isoformat(),'category_pages':manifest['page_count'],
        'listed_entries':manifest['listed_entries'],'unique_card_ids':len(manifest['product_ids']),
        'downloaded_parsed_cards':len(products),'successfully_analyzed_cards':len(quality),'errors':len(errors),
        'ddr5_cards':sum(p['ddr']=='DDR5' for p in products),
        'non_ddr5_cards':sum(p['ddr']!='DDR5' for p in products),
        'cards_pass_filter':sum(q['passes_filter'] for q in quality),
        'distinct_known_mpn':sum(q['mpn_known'] for q in model_quality),
        'duplicate_mpn_groups':sum(q['card_count']>1 for q in model_quality),
        'unknown_mpn_groups':sum(not q['mpn_known'] for q in model_quality),
        'selected_models':len(selected),'selected_cards':len(selected_ids),
        'raw_history_rows':len(histories),'selected_distinct_history_rows':len(selected_history),
        'selected_strict_observed_days':sum(q['strict_observed_days'] for q in selected),
        'selected_with_under_30_unique_events':sum(q['unique_tuples']<30 for q in selected),
        'selected_with_under_180_day_span':sum(q['span_days']<180 for q in selected),
        'selected_server_models':sum(q['server_memory'] for q in selected),
        'selected_specification_conflicts':sum(bool(q['specification_conflicts']) for q in selected),
        'first_event_date':min(q['first_event_date'] for q in quality),
        'last_event_date':max(q['last_event_date'] for q in quality)}
    for name,rows in [('products_all.csv',products),('history_all.csv',histories),
        ('quality_by_card.csv',quality),('quality_by_model.csv',model_quality),
        ('selected_models.csv',selected),('selected_history.csv',selected_history),
        ('selected_products.csv',[p for p in products if p['product_id'] in selected_ids]),
        ('daily_observed_all.csv',daily),('analysis_errors.csv',errors)]: save(name,rows)
    (DATA/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (DATA/'analysis_errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2))
    (DATA/'workbook_data.json').write_text(json.dumps({'summary':summary,'quality':model_quality,'selected':selected,'card_quality':quality},ensure_ascii=False))
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if errors: print('ERROR EXAMPLES',errors[:10])


if __name__=='__main__': main()
