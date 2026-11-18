"""Integrity metadata for trusted team-generated inference bundles."""
import hashlib
import json
from pathlib import Path

INFERENCE_FILES = ('model_final.joblib', 'feature_columns.json', 'model_metadata.json')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_artifact_manifest(directory):
    root = Path(directory)
    entries = [dict(path=name, size_bytes=(root/name).stat().st_size, sha256=file_sha256(root/name))
               for name in INFERENCE_FILES]
    (root/'artifact_manifest.json').write_text(json.dumps(dict(format_version=1, files=entries), indent=2))


def verify_artifact_manifest(directory):
    root = Path(directory)
    manifest = json.loads((root/'artifact_manifest.json').read_text())
    entries = {item['path']: item for item in manifest['files']}
    for name in INFERENCE_FILES:
        if name not in entries:
            raise ValueError(f'Artifact manifest is missing {name}')
        if file_sha256(root/name) != entries[name]['sha256']:
            raise ValueError(f'Artifact integrity check failed: {name}')
