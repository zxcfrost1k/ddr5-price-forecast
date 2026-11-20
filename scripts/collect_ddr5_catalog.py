"""Resumable public category crawl. Includes unavailable products; no account needed."""
import concurrent.futures as cf
import html
import json
import re
import threading
import time
import urllib.robotparser
from pathlib import Path
from collect_ram_pilot import fetch, UA

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/ram_ddr5'
RAW = DATA / 'raw'
lock = threading.Lock()
next_request = 0.0
robots = None


def get(url, dest):
    global next_request
    if dest.exists() and dest.with_suffix(dest.suffix + '.meta.json').exists():
        return dest.read_bytes()
    if not robots.can_fetch(UA, url):
        raise RuntimeError('robots.txt disallows ' + url)
    for attempt in range(3):
        with lock:
            delay = max(0, next_request-time.monotonic())
            if delay:
                time.sleep(delay)
            next_request = time.monotonic()+0.6
        try:
            payload, meta = fetch(url)
            temp = dest.with_suffix(dest.suffix + '.tmp')
            temp.write_bytes(payload)
            temp.replace(dest)
            dest.with_suffix(dest.suffix+'.meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
            return payload
        except Exception:
            if attempt == 2:
                raise
            time.sleep(3*(attempt+1))


def listing(page):
    url=f'https://pcstonks.com/catalog/?cat=8&available=none&sort=name&page={page}'
    content=get(url, RAW/f'catalog_{page}.html').decode('utf-8')
    ids=re.findall(r'data-json="/catalog/(\d+)\.json"', content)
    pages=[int(p) for p in re.findall(r'page=(\d+)',html.unescape(content))]
    if not ids:
        raise RuntimeError(f'Empty category page {page}')
    return page, ids, max(pages+[page])


def card(pid):
    payload=get(f'https://pcstonks.com/catalog/{pid}.json', RAW/f'{pid}.json')
    d=json.loads(payload)
    if not d.get('html') or 'История цен и наличия' not in d['html']:
        raise RuntimeError('Missing history ' + pid)
    return pid


def main():
    global robots
    RAW.mkdir(parents=True,exist_ok=True)
    b,m=fetch('https://pcstonks.com/robots.txt')
    (RAW/'robots.txt').write_bytes(b)
    (RAW/'robots.meta.json').write_text(json.dumps(m,indent=2))
    robots=urllib.robotparser.RobotFileParser()
    robots.parse(b.decode().splitlines())
    first=listing(1)
    pages={1:first[1]}
    total=first[2]
    print('CATEGORY_PAGES',total,flush=True)
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        for page, ids, last in pool.map(listing, range(2,total+1)):
            pages[page]=ids
            if page%10==0 or page==total:
                print('LISTINGS',page,'/',total,flush=True)
    ids=list(dict.fromkeys(pid for page in sorted(pages) for pid in pages[page]))
    manifest={'category_url':'https://pcstonks.com/catalog/?cat=8&available=none&sort=name',
              'page_count':total,'pages':pages,'product_ids':ids,
              'listed_entries':sum(map(len,pages.values())), 'unique_product_ids':len(ids)}
    (DATA/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print('CARDS',len(ids),flush=True)
    errors=[]
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        pending={pool.submit(card,pid):pid for pid in ids}
        for i,future in enumerate(cf.as_completed(pending),1):
            try:
                future.result()
            except Exception as e:
                errors.append({'id':pending[future],'error':str(e)})
            if i%50==0 or i==len(ids):
                print('HISTORIES',i,'/',len(ids),'ERRORS',len(errors),flush=True)
    (DATA/'fetch_errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2))
    print('COMPLETE',len(ids)-len(errors),'/',len(ids),flush=True)


if __name__=='__main__':
    main()
