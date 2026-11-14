"""Check official CIFAR-100 files every time; download only if needed."""
import hashlib
import os
import tarfile
import time
import urllib.request
from pathlib import Path
from fl_coordination._impl.fedteam.common import lock


def md5(path):
    value=hashlib.md5()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def ensure(root):
    from torchvision.datasets import CIFAR100
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    expected={name:checksum for name,checksum in CIFAR100.train_list+CIFAR100.test_list}
    expected[CIFAR100.meta['filename']]=CIFAR100.meta['md5']
    folder=root/CIFAR100.base_folder
    def valid():
        return all((folder/n).is_file() and md5(folder/n)==h for n,h in expected.items())
    with lock(root/'.cifar100-download.lock'):
        if not valid():
            archive=root/CIFAR100.filename
            if not archive.exists() or md5(archive)!=CIFAR100.tgz_md5:
                url=os.environ.get('CIFAR100_ARCHIVE_URL',CIFAR100.url).replace('http://','https://',1)
                if not url.startswith('https://'):raise ValueError('Dataset URL must use HTTPS')
                partial=archive.with_suffix(archive.suffix+'.part')
                for attempt in range(5):
                    try:
                        print(f'Downloading official CIFAR-100 archive, attempt {attempt+1}/5',flush=True)
                        req=urllib.request.Request(url,headers={'User-Agent':'coordination_flatness-CIFAR100/1.0'})
                        with urllib.request.urlopen(req,timeout=60) as src,partial.open('wb') as dst:
                            for block in iter(lambda:src.read(1024*1024),b''):dst.write(block)
                        if md5(partial)!=CIFAR100.tgz_md5:raise ValueError('CIFAR archive MD5 mismatch')
                        os.replace(partial,archive);break
                    except (OSError,ValueError):
                        if attempt==4:raise
                        time.sleep(min(2**attempt,8))
            # Extract only the three known regular files, never arbitrary paths/links.
            folder.mkdir(exist_ok=True)
            with tarfile.open(archive,'r:gz') as tar:
                for name,checksum in expected.items():
                    member=tar.getmember(CIFAR100.base_folder+'/'+name)
                    if not member.isfile():raise ValueError('Unexpected dataset archive member')
                    dest=folder/name;temp=dest.with_name(name+'.tmp')
                    with tar.extractfile(member) as src,temp.open('wb') as dst:
                        for block in iter(lambda:src.read(1024*1024),b''):dst.write(block)
                    if md5(temp)!=checksum:raise ValueError('CIFAR extracted file MD5 mismatch')
                    os.replace(temp,dest)
        if not valid():raise RuntimeError('Official CIFAR-100 file verification failed')
    return dict(root=str(root.resolve()),official_file_md5=expected,verified=True)
