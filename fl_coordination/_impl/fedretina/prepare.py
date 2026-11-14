"""Download the author's public archive, inspect it, then safely extract once."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
import re
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import urllib.parse
import urllib.request
import zipfile

FILE_ID = '1bW--_qRZnWbkb0XXvGBCSferdqXZ6pe7'
URL = f'https://drive.google.com/uc?export=download&id={FILE_ID}'


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__(); self.action = None; self.fields = {}

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'form' and a.get('id') == 'download-form':
            self.action = a.get('action')
        if tag == 'input' and a.get('name'):
            self.fields[a['name']] = a.get('value', '')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def download(root):
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=True)
    dest = root/'author_archive.bin'
    if dest.exists():
        if not zipfile.is_zipfile(dest) and not tarfile.is_tarfile(dest):
            raise RuntimeError('Existing author archive is incomplete')
        return dest
    url = URL
    for _ in range(3):
        response = urllib.request.urlopen(url, timeout=40)
        if 'text/html' in response.headers.get('Content-Type', ''):
            html = response.read(1024*1024).decode('utf-8', errors='replace'); response.close()
            parser = DownloadForm(); parser.feed(html)
            if not parser.action or urllib.parse.urlparse(parser.action).hostname != 'drive.usercontent.google.com':
                raise RuntimeError('Author archive unavailable; no download confirmation form')
            if parser.fields.get('id') != FILE_ID:
                raise RuntimeError('Unexpected Google Drive file identity')
            url = parser.action+'?'+urllib.parse.urlencode(parser.fields)
            continue
        response.close()
        part = dest.with_suffix('.part')
        # Every continuation must be a verified byte range, never append a full
        # response to a partial archive when a proxy/server ignores Range.
        probe = urllib.request.urlopen(urllib.request.Request(url, headers={'Range':'bytes=0-4095'}), timeout=40)
        with probe:
            match = re.fullmatch(r'bytes 0-4095/(\d+)', probe.headers.get('Content-Range',''))
            if probe.status != 206 or not match: raise RuntimeError('Archive does not support verified range download')
            prefix = probe.read(); total = int(match.group(1))
        if part.exists():
            with part.open('rb') as inp:
                if inp.read(4096) != prefix: raise RuntimeError('Partial archive has a different identity')
        written = part.stat().st_size if part.exists() else 0
        if written > total: raise RuntimeError('Partial archive exceeds advertised size')
        while written < total:
            end = min(total-1, written+256*1024*1024-1)
            response = urllib.request.urlopen(urllib.request.Request(url, headers={'Range':f'bytes={written}-{end}'}), timeout=60)
            expected = f'bytes {written}-{end}/{total}'
            if response.status != 206 or response.headers.get('Content-Range') != expected:
                response.close(); raise RuntimeError('Unexpected download range')
            start = written
            with response, part.open('ab') as out:
                while True:
                    block = response.read(1024*1024)
                    if not block: break
                    out.write(block); written += len(block)
            if written != end+1:
                raise RuntimeError(f'Incomplete transfer after {start} bytes; rerun to resume')
            print(f'Downloaded {written}/{total} bytes', flush=True)
        if not zipfile.is_zipfile(part) and not tarfile.is_tarfile(part):
            raise RuntimeError('Download is not a supported archive')
        part.replace(dest)
        (root/'archive_source.json').write_text(json.dumps(dict(
            url=URL, author_repository='https://github.com/Zeng-Shuang/FedLD',
            bytes=dest.stat().st_size, sha256=sha256(dest)), indent=2))
        return dest
    raise RuntimeError('Repeated download confirmation without archive')


def safe_target(root, member):
    p = PurePosixPath(member.replace('\\', '/'))
    if p.is_absolute() or '..' in p.parts or any(':' in part for part in p.parts):
        raise ValueError('Unsafe archive member: '+member)
    target = root.joinpath(*p.parts).resolve()
    if not target.is_relative_to(root): raise ValueError('Archive path escapes target')
    return target


def extract(archive, root):
    archive, root = Path(archive), Path(root).resolve()
    marker = root/'extracted.json'; digest = sha256(archive)
    if marker.exists():
        if json.loads(marker.read_text())['archive_sha256'] != digest:
            raise RuntimeError('Archive changed; use a new data directory')
        return
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()): raise RuntimeError('Incomplete extraction; inspect before using a fresh destination')
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as src:
            members = src.infolist()
            for m in members:
                safe_target(root, m.filename)
                if stat.S_ISLNK(m.external_attr >> 16): raise ValueError('Archive symlink rejected')
            for m in members:
                p = safe_target(root, m.filename)
                if m.is_dir(): p.mkdir(parents=True, exist_ok=True)
                else:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with src.open(m) as inp, p.open('xb') as out: shutil.copyfileobj(inp, out)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as src:
            members = src.getmembers()
            for m in members:
                safe_target(root, m.name)
                if not (m.isfile() or m.isdir()): raise ValueError('Only regular archive files/directories allowed')
            for m in members:
                p = safe_target(root, m.name)
                if m.isdir(): p.mkdir(parents=True, exist_ok=True)
                else:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with src.extractfile(m) as inp, p.open('xb') as out: shutil.copyfileobj(inp, out)
    else: raise ValueError('Unsupported author archive')
    marker.write_text(json.dumps(dict(archive_sha256=digest), indent=2))
