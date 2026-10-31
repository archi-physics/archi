"""The code-text secret screen (archi.sources.code_text.secret_classes).

A file is left out of the graph when its text holds a literal secret; a file
that only names or passes a secret around, or holds public material, is kept.
Every secret-shaped value here is invented and assembled at run time from
pieces, so the test file itself holds no complete secret-shaped string.
"""
import time

import pytest

from archi.sources.code_text import secret_classes

_PW = "Zx9v" + "Q2mT7rLp"  # an invented password
_LONG = _PW + "W4"
_B = "-----BEGIN "
_GH = "gh" + "p_" + "Q7x" * 12
_AWS = "AK" + "IA" + "Q2W3E4R5T6Y7U8I9"
_SLACK = "xo" + "xb-" + "1234567890-abcdefghij"
_JWT = "ey" + "J" + "hbGciOiJIUzI1NiJ9" + ".ey" + "J" + "zdWIiOiIxMjM0NTY3ODkwIn0" + "." + "c2lnbmF0dXJlLXZhbHVl"
_BLOB = "Qk9HVVM" + "rZm9vYmFy" * 12  # 80+ base64 characters, not a real key
_BASIC = "YWxp" + "Y2U6" + "WnhOdlEy"


LITERAL = {
    "private key block": (_B + "RSA PRIVATE KEY-----\nabc\n", "private_key"),
    "openssh private key": (_B + "OPENSSH PRIVATE KEY-----\n", "private_key"),
    "pgp private key": (_B + "PGP PRIVATE KEY BLOCK-----\n", "private_key"),
    "github token in a script": (f"curl -H 'X: {_GH}' https://example.org\n", "token"),
    "aws key id": (f"aws_access_key_id = {_AWS}\n", "token"),
    "slack token": (f"SLACK={_SLACK}\n", "token"),
    "jwt": (f"cookie: {_JWT}\n", "jwt"),
    "base64 blob": (f"tls.key: {_BLOB}\n", "base64_blob"),
    "url with a password": (f"DB=postgres://archi:{_PW}@db01/archi\n", "credential_url"),
    "url with a plain-word password": ("mysql://root:" + "admin" + "@db01/test\n", "credential_url"),
    "url-encoded password": ("postgres://u:" + "Zx9%40v" + "@h/db\n", "credential_url"),
    "url with an empty user": ("redis://:" + _PW + "@cache:6379/0\n", "credential_url"),
    "authorization header": ('headers = {"Authorization": "Bearer ' + _LONG + '"}\n', "authorization_header"),
    "basic auth header": (f"curl -H 'Authorization: Basic {_BASIC}' x\n", "authorization_header"),
    "curl user": (f"curl -u admin:{_PW} https://example.org\n", "authorization_header"),
    "yaml password": (f"db:\n  password: {_PW}\n", "secret_assignment"),
    "quoted python password": (f'PASSWORD = "{_PW}"\n', "secret_assignment"),
    "annotated constant": (f'API_TOKEN: str = "{_LONG}"\n', "secret_assignment"),
    "client secret in json": (f'{{"client_secret": "{_LONG}"}}\n', "secret_assignment"),
    "camel case api key": (f'apiKey: "{_LONG}"\n', "secret_assignment"),
    "shell export": (f"export ACCESS_TOKEN={_LONG}\n", "secret_assignment"),
    "cli flag with equals": (f"run --password={_PW}\n", "secret_assignment"),
    "cli flag with a space": (f"run --password {_PW}\n", "secret_assignment"),
    "db pass key": (f"DB_PASS: {_PW}\n", "secret_assignment"),
    "passphrase": (f"passphrase = '{_PW}'\n", "secret_assignment"),
    "docker auth": ('{"auths": {"r.example.org": {"auth": "' + _BASIC + "Q2bVQ3" + '"}}}\n', "secret_assignment"),
    "netrc": (f"machine example.org login archi password {_PW}\n", "secret_assignment"),
    "xml element": (f"<password>{_PW}</password>\n", "secret_assignment"),
    "dollar inside a quoted password": ('password = "Pa' + "$$" + 'w0rd!x"\n', "secret_assignment"),
    "symbol first": ("password: '!" + "Qaz2wsx3edc'\n", "secret_assignment"),
    "unquoted identifier-shaped yaml password": ("POSTGRES_PASSWORD: archi_" + _PW + "_db\n", "secret_assignment"),
    "dotted env password": ("DB_PASSWORD=prod." + _PW + "\n", "secret_assignment"),
    "ini password": ("[db]\npassword = Prod_" + _PW + "\n", "secret_assignment"),
    "quoted all-caps password": ('password = "XK9P' + 'Q2MZ7RT4"\n', "secret_assignment"),
    "puppet arrow": ("password => '" + _PW + "',\n", "secret_assignment"),
    "quoted key arrow": ('"db_password" => "' + _PW + '"\n', "secret_assignment"),
    "passphrase of words": ('passphrase = "correct horse battery staple"\n', "secret_assignment"),
    "client secret flag": ("login --client-secret " + _PW + "\n", "secret_assignment"),
    "access token flag": ("run --access-token " + _LONG + "\n", "secret_assignment"),
    "mysql -p": ("mysql -u root -p" + _PW + " ops\n", "secret_assignment"),
    "data uri that is not an image": ("x = data:application/octet-stream;base64," + "eyJh" * 30 + "\n", "base64_blob"),
}

REFERENCE = {
    "template": "password: ${DB_PASSWORD}\n",
    "shell variable": "password: $DB_PASSWORD\n",
    "helm template": "password: {{ .Values.db.password }}\n",
    "yaml alias": "password: *db_password\n",
    "environment lookup": 'token = os.environ["ARCHI_TOKEN"]\n',
    "environment variable name": 'api_key = getenv("API_KEY")\n',
    "call": "secret = read_secret(path)\n",
    "dict lookup": 'password = config["password_file"]\n',
    "attribute": "self.token = token\n",
    "percent format": 'headers["Authorization"] = "Bearer %s" % token\n',
    "f-string header": 'headers = {"Authorization": f"Bearer {token}"}\n',
    "url with a variable password": "url = f'postgres://{user}:{password}@{host}/db'\n",
    "url with a templated password": "DB=postgres://archi:${DB_PASS}@db/archi\n",
    "path to a secret": "password_file: /etc/archi/db.pass\n",
    "key that names a secret": "secretName: archi-db-credentials\n",
    "two-line yaml secret block": "volumes:\n  - secret:\n      secretName: archi-db-credentials\n",
    "token url": "token_url: https://auth.example.org/oauth/token\n",
    "number": "max_tokens: 16000\n",
    "type annotation": "token: Optional[str] = None\n",
    "short value": "pwd: abc\n",
    "placeholder": "password: changeme123\n",
    "plain word": "password: required\n",
    "auth setting": "auth: kerberos5\n",
    "comparison": 'if token == "expected":\n',
    "prose in quotes": 'password = "enter your password here"\n',
    "flag followed by another flag": "run --password --verbose\n",
    "hex digest": "sha512: " + "0123456789abcdef" * 8 + "\n",
    "ssh public key": "ssh-rsa " + "AAAAB3NzaC1yc2E" + "A" * 90 + " ops@example.org\n",
    "package integrity hash": '"integrity": "sha512-' + "Qk9HVVMrZm9v" * 8 + '=="\n',
    "inline image": "![logo](data:image/png;base64," + "iVBORw0KGgo" * 10 + ")\n",
    "notebook image": '"image/png": "' + "iVBORw0KGgo" * 10 + '"\n',
    "cluster ca": "certificate-authority-data: " + "LS0tLS1CRUdJTi" * 8 + "\n",
    "plain prose": "Rotate the token every 90 days; never commit a password.\n",
    "python pass": "def f():\n    pass\n",
    "flag help text": "parser.add_argument('--password password')\n",
    "another tool's -u url": "pip install -u https://example.org/simple/pkg.tar\n",
    "f-string with a call": 'token = f"tk-{time.time()}"\n',
    "quote that spans code": 'password = ask("password", getpass("x"))\n',
    "upper-case variable name": "PASSWORD=DB_PASSWORD_VAR\n",
    "loopback url in docs": "postgresql://postgres:okg@127.0.0.1:5491/packtest\n",
    "example host url": "https://alice:" + _PW + "@example.com/repo.git\n",
    "placeholder url password": "scheme://user:pass@host/path\n",
    "comparison arrow": "if ($token >= 3) {\n",
}


@pytest.mark.parametrize("name", sorted(LITERAL))
def test_a_literal_secret_is_found(name):
    text, kind = LITERAL[name]
    assert kind in secret_classes(text)


@pytest.mark.parametrize("name", sorted(REFERENCE))
def test_a_reference_or_public_material_is_not_a_secret(name):
    assert secret_classes(REFERENCE[name]) == []


def test_empty_text_has_no_secret():
    assert secret_classes("") == []


@pytest.mark.parametrize("unit", [
    "a_", "a-b.", "Q7x_", "ab" * 3 + "=", "x" * 7 + "/",
    "a:", "://", "Authorization: ", "password ", "<password>", "-u x:",
    "machine a login b password ", "--x-token ", "mysql ", "a => ",
])
def test_a_long_line_costs_linear_time(unit):
    """A 1 MB line built from one pattern's own pieces screens in under two
    seconds (the key pattern once took minutes on 80 KB)."""
    line = unit * (1_000_000 // len(unit))
    start = time.perf_counter()
    secret_classes(line)
    assert time.perf_counter() - start < 2.0
