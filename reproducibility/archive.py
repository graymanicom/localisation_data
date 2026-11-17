#!/usr/bin/env python3
"""Create or verify local reproducibility archives. Standard library only; no network.
Run from any directory: python reproducibility/archive.py create RELEASE
Verify: python reproducibility/archive.py verify RELEASE
Archives contain private research material and must not be committed or uploaded.
"""
import argparse, hashlib, json, os, tarfile
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parent

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['create','verify'])
    p.add_argument('release')
    args=p.parse_args()
    if not args.release or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.release):
        p.error('Release must contain only letters, digits, hyphens and underscores.')
    config=json.loads((HERE/'archive_sources.json').read_text())
    dest=HERE/'private'/args.release
    manifest_path=HERE/'archives'/f'{args.release}.json'
    if args.action=='create':
        if dest.exists() or manifest_path.exists():raise SystemExit('Release already exists; refusing overwrite.')
        (HERE/'private').mkdir(mode=0o700,exist_ok=True)
        os.chmod(HERE/'private',0o700)
        dest.mkdir(mode=0o700)
        entries=[]
        for group,sources in config['groups'].items():
            files={}
            for source in sources:
                path=REPO/source
                if not path.exists():raise SystemExit(f'Missing required source: {source}')
                for f in ([path] if path.is_file() else sorted(path.rglob('*'))):
                    if not f.is_file() or f.name=='.DS_Store' or '__pycache__' in f.parts:continue
                    if f.name.startswith('.env') or f.suffix in {'.pem','.key','.token'}:continue
                    rel=f.relative_to(REPO).as_posix()
                    files[rel]={'path':rel,'bytes':f.stat().st_size,'sha256':digest(f)}
            target=dest/f'{group}.tar.gz'
            with tarfile.open(target,'w:gz',compresslevel=6,dereference=True) as tar:
                for rel in sorted(files):tar.add(REPO/rel,arcname=rel,recursive=False)
            os.chmod(target,0o600)
            for rel,item in files.items():
                if digest(REPO/rel)!=item['sha256']:raise RuntimeError(f'Source changed during archival: {rel}')
            entries.append({'archive':str(target.relative_to(HERE)),'sha256':digest(target),'bytes':target.stat().st_size,'members':list(files.values())})
            print(f'Archived {group}: {len(files)} files',flush=True)
        manifest_path.parent.mkdir(exist_ok=True)
        manifest_path.write_text(json.dumps({'schema_version':1,'release':args.release,'repository':REPO.name,'created_at':datetime.now(ZoneInfo('Africa/Johannesburg')).isoformat(),'status':'local private archival snapshot, not historical execution environment','archives':entries},indent=2)+'\n')
    manifest=json.loads(manifest_path.read_text())
    for item in manifest['archives']:
        path=HERE/item['archive']
        if digest(path)!=item['sha256']:raise RuntimeError(f'Archive checksum failed: {path}')
        expected={m['path']:m for m in item['members']}
        with tarfile.open(path,'r:gz') as tar:
            members=tar.getmembers()
            if len(members)!=len(expected) or {m.name for m in members}!=set(expected):raise RuntimeError('Archive membership mismatch')
            for member in members:
                if not member.isfile():raise RuntimeError('Unexpected archive member type')
                h=hashlib.sha256()
                with tar.extractfile(member) as f:
                    for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
                if h.hexdigest()!=expected[member.name]['sha256'] or member.size!=expected[member.name]['bytes']:raise RuntimeError(f'Member checksum failed: {member.name}')
        print(f'Verified {path.name}: {len(expected)} members',flush=True)
    print('All archives verified; source files were not modified.',flush=True)
if __name__=='__main__':main()
