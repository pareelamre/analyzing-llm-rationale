"""Local authenticated encryption for private prepared orders; no paid KMS call.

The owner supplies a random Fernet key to the private runtime. There is no
default key, plaintext fallback, or automatic remote key provisioning.
"""
import json
import os


def _cipher():
    from cryptography.fernet import Fernet

    return Fernet(os.environ['FORESEA_TWIN_ORDER_ENCRYPTION_KEY'].encode('ascii'))


def seal(encoded: str, scope_id: str, intent_hash: str) -> str:
    if not encoded:
        return ''
    envelope = json.dumps({'scope_id': scope_id, 'intent_hash': intent_hash, 'request': encoded},
                          sort_keys=True, separators=(',', ':')).encode()
    return 'fernet-v1:'+_cipher().encrypt(envelope).decode('ascii')


def open_request(encoded: str, scope_id: str, intent_hash: str) -> str:
    if not encoded:
        return ''
    if not encoded.startswith('fernet-v1:') or len(encoded) > 100_000:
        raise ValueError('private prepared request envelope is unsupported')
    value = json.loads(_cipher().decrypt(encoded.removeprefix('fernet-v1:').encode('ascii')))
    if value.get('scope_id') != scope_id or value.get('intent_hash') != intent_hash:
        raise ValueError('private prepared request envelope binding changed')
    request = value['request']
    if not isinstance(request, str) or len(request.encode('utf-8')) > 64_000:
        raise ValueError('private prepared request size is unsupported')
    return request
