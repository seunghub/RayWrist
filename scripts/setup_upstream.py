#!/usr/bin/env python3
"""Obtain or check an official backend at the revision used by this integration."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,os
ROOT=Path(__file__).resolve().parents[1]


def command(*args,cwd=None):
    return subprocess.check_output(args,cwd=cwd,text=True).strip()


def check(path,spec):
    if Path(command('git','rev-parse','--show-toplevel',cwd=path)).resolve()!=path.resolve():
        raise ValueError('The destination must be an independent upstream checkout')
    head=command('git','rev-parse','HEAD',cwd=path)
    if head!=spec['commit']:
        raise ValueError(f'Expected {spec["commit"]}, found {head}')
    for relative,wanted in spec['files'].items():
        file=path/relative
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=wanted:
            raise ValueError(f'Pinned upstream file differs: {relative}')
    print(f'Verified {path.name} at {head}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backend',choices=['gr00t','openpi'])
    parser.add_argument('--destination',type=Path)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--submodules',action='store_true')
    args=parser.parse_args();spec=json.loads((ROOT/'configs/upstream.json').read_text())[args.backend]
    target=args.destination or ROOT/'third_party'/('Isaac-GR00T'if args.backend=='gr00t'else'openpi')
    if not target.exists():
        if args.check_only:parser.error(f'Checkout is missing: {target}')
        target.parent.mkdir(parents=True,exist_ok=True)
        os.environ.setdefault('GIT_LFS_SKIP_SMUDGE','1')
        subprocess.run(['git','clone','--no-checkout','--filter=blob:none',spec['repository'],str(target)],check=True)
        subprocess.run(['git','fetch','--depth','1','origin',spec['commit']],cwd=target,check=True)
        subprocess.run(['git','checkout','--detach',spec['commit']],cwd=target,check=True)
    check(target,spec)
    if args.submodules:
        subprocess.run(['git','submodule','update','--init','--recursive'],cwd=target,check=True)

if __name__=='__main__':main()
