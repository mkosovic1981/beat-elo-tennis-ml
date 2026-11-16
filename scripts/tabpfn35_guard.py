"""
Exp D35 guard (one-off, non-commercial research run) - MUST be imported BEFORE `tabpfn`.

Rules enforced here:
  - TabPFN-3.5 weights only from a LOCAL file (model_path). Never the hosted API / Prior Labs client.
  - TABPFN_TOKEN is deleted from the environment, telemetry off, HF offline (no automatic download).
  - ALLOWLIST network policy: every outbound network attempt except loopback (127.0.0.0/8, ::1, localhost,
    AF_UNIX) raises ConnectionError AND is recorded as "host:port". Patched: getaddrinfo, gethostbyname,
    gethostbyname_ex, gethostbyaddr, getnameinfo, create_connection, socket.connect / connect_ex / sendto
    (so urllib / requests / http.client are blocked too, also by raw IP and for telemetry hosts).
    Call assert_no_network_contact() (alias assert_no_priorlabs_contact) after fit/predict - a library that
    swallows the error is still caught.
  - Limit: pure-Python socket patching; C extensions calling OS sockets directly bypass it (torch/CUDA do not
    use network sockets for GPU work).
Weights location: --ckpt argument or env TABPFN35_CKPT. File name must contain 'v3.5' (tabpfn derives
the model version from the file name). Optional integrity check: env TABPFN35_EXPECTED_SHA.
Licence: tabpfn-3-5-license-v1.0 (research/evaluation only). Results must never be used in production.

Self-check without any model:  python tabpfn35_guard.py
"""
import hashlib
import ipaddress
import os
import socket
import sys
from pathlib import Path

BLOCKED_ATTEMPTS = []  # entries "host:port" (every non-loopback outbound attempt)

os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ.pop('TABPFN_TOKEN', None)


class GuardConnectionError(ConnectionError):
    pass


def _is_loopback(host):
    """ALLOWLIST: only loopback is allowed. None (local bind/lookup) is allowed; everything else is blocked."""
    if host is None:
        return True
    if isinstance(host, (bytes, bytearray)):
        host = bytes(host).decode('ascii', 'ignore')
    h = str(host).strip().lower().rstrip('.')
    if h == 'localhost' or h.endswith('.localhost'):
        return True
    h = h.split('%', 1)[0]  # IPv6 zone id
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return False  # any other name: blocked, never resolved
    if getattr(ip, 'ipv4_mapped', None) is not None:
        ip = ip.ipv4_mapped
    return ip.is_loopback


def _check_host(host, port=None):
    if not _is_loopback(host):
        BLOCKED_ATTEMPTS.append(f'{host}:{port}')
        raise GuardConnectionError(f'D35 guard: outbound network access to {host}:{port} is blocked (loopback only)')


def _check_address(address):
    """Address as accepted by connect/sendto. AF_UNIX (str/bytes path) is always allowed."""
    if isinstance(address, tuple) and address:
        _check_host(address[0], address[1] if len(address) > 1 else None)


def _install_network_block():
    if getattr(socket, '_d35_blocked', False):
        return
    o_getaddrinfo = socket.getaddrinfo
    o_gethostbyname = socket.gethostbyname
    o_gethostbyname_ex = socket.gethostbyname_ex
    o_gethostbyaddr = socket.gethostbyaddr
    o_getnameinfo = socket.getnameinfo
    o_create = socket.create_connection
    o_connect = socket.socket.connect
    o_connect_ex = socket.socket.connect_ex
    o_sendto = socket.socket.sendto

    def getaddrinfo(host, port, *a, **k):
        _check_host(host, port)
        return o_getaddrinfo(host, port, *a, **k)

    def gethostbyname(host):
        _check_host(host)
        return o_gethostbyname(host)

    def gethostbyname_ex(host):
        _check_host(host)
        return o_gethostbyname_ex(host)

    def gethostbyaddr(host):
        _check_host(host)
        return o_gethostbyaddr(host)

    def getnameinfo(sockaddr, flags):
        _check_address(tuple(sockaddr))
        return o_getnameinfo(sockaddr, flags)

    def create_connection(address, *a, **k):
        _check_address(tuple(address))
        return o_create(address, *a, **k)

    def connect(self, address):
        _check_address(address)
        return o_connect(self, address)

    def connect_ex(self, address):
        _check_address(address)
        return o_connect_ex(self, address)

    def sendto(self, data, *args):
        if args:
            _check_address(args[-1])
        return o_sendto(self, data, *args)

    socket.getaddrinfo = getaddrinfo
    socket.gethostbyname = gethostbyname
    socket.gethostbyname_ex = gethostbyname_ex
    socket.gethostbyaddr = gethostbyaddr
    socket.getnameinfo = getnameinfo
    socket.create_connection = create_connection
    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.socket.sendto = sendto
    socket._d35_blocked = True


_install_network_block()


def assert_no_network_contact():
    """Stop the run if anything tried to reach a non-loopback address (even if the library swallowed the error)."""
    if BLOCKED_ATTEMPTS:
        sys.exit('\nSTOP: an external network access was attempted: ' + ', '.join(sorted(set(BLOCKED_ATTEMPTS))) +
                 '\nThis experiment must run offline with local weights only. Do not continue; report this.')


assert_no_priorlabs_contact = assert_no_network_contact  # backwards-compatible name used by the scripts


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def ckpt_path(explicit=None, require=True):
    raw = explicit or os.environ.get('TABPFN35_CKPT')
    if not raw:
        if require:
            sys.exit('TabPFN-3.5 weights path not given. Use --ckpt <file> or env TABPFN35_CKPT '
                     '(file tabpfn-v3.5-20260909.safetensors from Hugging Face Prior-Labs/tabpfn_3_5). '
                     'Nothing is downloaded automatically.')
        return None
    p = Path(raw)
    if require and not p.is_file():
        sys.exit(f'TabPFN-3.5 weights NOT found: {p}')
    if 'v3.5' not in p.name:
        sys.exit(f'Weights file name must contain "v3.5" (tabpfn picks the model version from it): {p.name}')
    if require and p.stat().st_size < 1_000_000:
        sys.exit(f'{p} is too small ({p.stat().st_size} B) - incomplete download.')
    exp = os.environ.get('TABPFN35_EXPECTED_SHA')
    if require and exp and sha256_of(p).lower() != exp.lower():
        sys.exit(f'SHA256 mismatch for {p.name}')
    return p


if __name__ == '__main__':
    import http.client
    import urllib.request

    def must_block(fn, label):
        BLOCKED_ATTEMPTS.clear()
        try:
            fn()
        except ConnectionError:
            pass
        except Exception as e:  # urllib wraps in URLError(OSError)
            if not isinstance(getattr(e, 'reason', None), ConnectionError):
                sys.exit(f'SELF-CHECK FAILED ({label}): unexpected {e!r}')
        else:
            sys.exit(f'SELF-CHECK FAILED ({label}): not blocked')
        if not BLOCKED_ATTEMPTS:
            sys.exit(f'SELF-CHECK FAILED ({label}): not recorded')

    def raw(addr, ex=False):
        s_ = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            (s_.connect_ex if ex else s_.connect)(addr)
        finally:
            s_.close()

    def udp(addr):
        s_ = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s_.sendto(b'x', addr)
        finally:
            s_.close()

    must_block(lambda: socket.getaddrinfo('api.priorlabs.ai', 443), 'getaddrinfo priorlabs')
    must_block(lambda: socket.gethostbyname('api.priorlabs.ai'), 'gethostbyname')
    must_block(lambda: socket.gethostbyname_ex('us.i.posthog.com'), 'gethostbyname_ex posthog')
    must_block(lambda: socket.create_connection(('example.com', 80)), 'create_connection')
    must_block(lambda: raw(('8.8.8.8', 53)), 'raw connect IP')
    must_block(lambda: raw(('8.8.8.8', 53), ex=True), 'raw connect_ex IP')
    must_block(lambda: udp(('8.8.8.8', 53)), 'sendto UDP')
    must_block(lambda: urllib.request.urlopen('http://us.i.posthog.com/', timeout=3), 'urllib')
    must_block(lambda: http.client.HTTPSConnection('huggingface.co', timeout=3).request('GET', '/'), 'http.client')
    BLOCKED_ATTEMPTS.clear()
    # loopback must remain allowed (incl. socketpair / AF_UNIX local IPC)
    srv = socket.socket()
    srv.bind(('127.0.0.1', 0))
    srv.listen(1)
    cli = socket.socket()
    cli.connect(srv.getsockname())
    cli.close()
    srv.close()
    socket.getaddrinfo('localhost', 80)
    socket.gethostbyname('127.0.0.1')
    a_, b_ = socket.socketpair()
    a_.close()
    b_.close()
    if hasattr(socket, 'AF_UNIX'):
        u = socket.socket(socket.AF_UNIX)
        try:
            u.connect('/nonexistent-d35-sock')
        except OSError as e:
            assert not isinstance(e, GuardConnectionError)
        finally:
            u.close()
    assert not BLOCKED_ATTEMPTS, BLOCKED_ATTEMPTS
    assert 'TABPFN_TOKEN' not in os.environ
    print('guard self-check OK (all non-loopback network blocked + recorded, loopback allowed, token removed, HF offline).')
