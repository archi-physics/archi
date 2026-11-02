#!/usr/bin/env python3
"""Interactive CERN SSO login: Kerberos + TOTP, saved as session cookies.

Moved from okg-deployments ``cms/scripts/sso-login.py`` (main ``32f2b4e3c2``).
It uses one shared session, so you enter your TOTP code once: after the first
service authenticates, Keycloak reuses the SSO session for the others. The
cookie files it writes are the ones ``archi.auth.cookies`` reads.

Usage::

    kinit <username>@CERN.CH
    python -m archi.downloaders.sso_login                 # every service
    python -m archi.downloaders.sso_login hypernews
    python -m archi.downloaders.sso_login --check         # verify saved cookies

Needs ``requests-kerberos`` (and ``python-dotenv`` if you keep settings such
as ``CERN_GRID_CA`` in a ``.env`` file in the current directory).

Changes from the okg-deployments copy: cookie files go to ``--cookie-dir``,
else ``ARCHI_COOKIE_DIR``, else ``./.cookies`` (the old copy wrote to
``<repo>/.cookies``), and ``.env`` is read from the current directory only
(the old copy also read the repository and deployment directories). The login
flow is unchanged.
"""

import argparse
import http.cookiejar
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

COOKIE_DIR_ENV = "ARCHI_COOKIE_DIR"
COOKIE_DIR = Path(".cookies")


def cookie_dir(explicit: str | None = None) -> Path:
    """``--cookie-dir``, else ``ARCHI_COOKIE_DIR``, else ``./.cookies``."""
    if explicit:
        return Path(explicit).expanduser()
    raw = os.environ.get(COOKIE_DIR_ENV)
    return Path(raw).expanduser() if raw else COOKIE_DIR


def _load_dotenv() -> None:
    """Read ``./.env`` when python-dotenv is installed (optional convenience)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()

# Each service: url to trigger SSO, and a predicate that returns True on success.
# Services that use CERN Keycloak SSO (Kerberos + TOTP).
SERVICES = {
    "hypernews": {
        "url":  "https://hypernews.cern.ch/HyperNews/CMS/get/comp-ops.html",
        "ok":   lambda code, text: code == 200 and "auth.cern.ch" not in text[:500],
    },
    "cmsmonit_docs": {
        "url":  "https://cmsmonit-docs.web.cern.ch/",
        "ok":   lambda code, text: code == 200 and "auth.cern.ch" not in text[:500],
    },
    "cmssi_docs": {
        "url":  "https://cmssi.docs.cern.ch/",
        "ok":   lambda code, text: code == 200 and "auth.cern.ch" not in text[:500],
    },
    "cms_http_group_docs": {
        "url":  "https://cms-http-group.docs.cern.ch/",
        "ok":   lambda code, text: code == 200 and "auth.cern.ch" not in text[:500],
    },
    "cms_conddb": {
        # CondDB browser — backs the `conddb_global_tags` source.
        "url":  "https://cms-conddb.cern.ch/cmsDbBrowser/",
        "ok":   lambda code, text: code == 200 and "auth.cern.ch" not in text[:500],
    },
    # NOTE: CERNBox was tried here but removed — it uses OAuth bearer
    # tokens / Basic auth for its WebDAV endpoints, not the oauth2-proxy
    # cookie scheme this script drives. Trying to SSO-login to CERNBox
    # silently saves zero cookies. EOS data from a Mac must go via lxplus
    # rsync (see notes/rsync_snapshot.sh) or XRootD.
}

ENV_VARS = {
    "hypernews": "HYPERNEWS_COOKIE_FILE",
    "cmsmonit_docs": "CMSMONIT_DOCS_COOKIE_FILE",
    "cmssi_docs": "CMSSI_DOCS_COOKIE_FILE",
    "cms_http_group_docs": "CMS_HTTP_GROUP_DOCS_COOKIE_FILE",
    "cms_conddb": "CONDDB_COOKIE_FILE",
}


def _complete_oidc(session, service_url: str, krb, *, totp_code: str | None = None) -> bool:
    """Drive the full CERN SSO OIDC flow for one service URL.

    If totp_code is None and 2FA is required, prompts the user.
    Returns True if we end up authenticated (no SSO redirect on final check).
    """
    resp = session.get(service_url, auth=krb, timeout=90)

    # Check if we landed on the Keycloak login page
    on_sso = "auth.cern.ch" in resp.url or "auth.cern.ch" in resp.text[:500]

    if on_sso:
        # Find the Kerberos broker link on the Keycloak login page
        krb_link = None
        for m in re.finditer(r'href="([^"]*kerberos[^"]*)"', resp.text, re.I):
            krb_link = m.group(1).replace("&amp;", "&")
            if not krb_link.startswith("http"):
                krb_link = "https://auth.cern.ch" + krb_link
            break

        if not krb_link:
            # Keycloak may have auto-selected Kerberos (no link on page).
            pass
        else:
            resp2 = session.get(krb_link, auth=krb, timeout=90)

            # Check if 2FA is needed
            if 'name="otp"' in resp2.text:
                form_action = None
                for m in re.finditer(r'<form[^>]+action="([^"]+)"', resp2.text):
                    form_action = m.group(1).replace("&amp;", "&")
                    break

                if not form_action:
                    print("  ERROR: Could not find TOTP form action.")
                    return False

                code = totp_code or input("  Enter your CERN TOTP code: ").strip()
                if not code:
                    print("  Skipped.")
                    return False

                resp3 = session.post(form_action, data={"otp": code, "login": "Sign In"}, timeout=90)

                if 'name="otp"' in resp3.text:
                    print("  ERROR: Wrong TOTP code.")
                    return False

        # After Kerberos (and optional TOTP), re-request the service URL.
        final = session.get(service_url, auth=krb, timeout=90)
    else:
        final = resp

    authenticated = "auth.cern.ch" not in final.url and "auth.cern.ch" not in final.text[:500]
    return authenticated


def _save_service_cookies(
    session, service_name: str, service_url: str, *, directory: Path
) -> tuple[int, int]:
    """Extract cookies for the service domain (+ auth.cern.ch) and save."""
    directory.mkdir(parents=True, exist_ok=True)
    cookie_file = directory / f"{service_name}.txt"

    service_domain = urlparse(service_url).hostname
    keep_domains = {service_domain, "auth.cern.ch"}

    jar = http.cookiejar.MozillaCookieJar(str(cookie_file))
    for cookie in session.cookies:
        if any(cookie.domain.endswith(d) for d in keep_domains):
            jar.set_cookie(cookie)

    jar.save(ignore_discard=True, ignore_expires=True)
    cookie_file.chmod(0o600)

    svc_count = sum(1 for c in jar if c.domain.endswith(service_domain))
    return len(jar), svc_count


def _check_cookies(service_name: str, service: dict, *, directory: Path) -> bool:
    """Return True if existing cookies still authenticate the service."""
    import requests, time

    cookie_file = directory / f"{service_name}.txt"
    if not cookie_file.exists():
        print(f"[{service_name}] No cookie file")
        return False

    jar = http.cookiejar.MozillaCookieJar(str(cookie_file))
    jar.load(ignore_discard=True, ignore_expires=True)

    service_domain = urlparse(service["url"]).hostname
    svc_cookies = [c for c in jar if c.domain.endswith(service_domain)]
    if not svc_cookies:
        print(f"[{service_name}] ✗ No cookies for {service_domain}")
        return False

    session = requests.Session()
    session.cookies = jar

    try:
        resp = session.get(service["url"], timeout=15, allow_redirects=True)
        ok = service["ok"](resp.status_code, resp.text)
        age_h = (time.time() - cookie_file.stat().st_mtime) / 3600
        status = "✓" if ok else "✗"
        if ok:
            reason = f"age {age_h:.1f}h, {len(svc_cookies)} service cookies"
        elif "auth.cern.ch" in resp.url:
            reason = f"HTTP {resp.status_code}, redirected to SSO (cookies expired?)"
        else:
            reason = f"HTTP {resp.status_code}"
        print(f"[{service_name}] {status} {reason}")
        return ok
    except Exception as e:
        print(f"[{service_name}] ✗ {e}")
        return False


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "services", nargs="*", metavar="SERVICE",
        help=f"Services: {', '.join(SERVICES)} (default: all)",
    )
    parser.add_argument("--check", action="store_true", help="Verify existing cookies only")
    parser.add_argument(
        "--cookie-dir",
        default=None,
        help=f"Where cookie files go (default: ${COOKIE_DIR_ENV}, else ./.cookies)",
    )
    args = parser.parse_args(argv)
    directory = cookie_dir(args.cookie_dir)
    _load_dotenv()

    targets = list(SERVICES) if not args.services or args.services == ["all"] else args.services
    unknown = [s for s in targets if s not in SERVICES]
    if unknown:
        parser.error(f"Unknown: {', '.join(unknown)}. Choose from: {', '.join(SERVICES)}")

    if args.check:
        results = [_check_cookies(n, SERVICES[n], directory=directory) for n in targets]
        sys.exit(0 if all(results) else 1)

    # Verify Kerberos
    import subprocess, requests
    from requests_kerberos import HTTPKerberosAuth, OPTIONAL

    if subprocess.run(["klist", "-s"], capture_output=True).returncode != 0:
        print("ERROR: No Kerberos ticket. Run: kinit <username>@CERN.CH")
        sys.exit(1)

    for line in subprocess.run(["klist"], capture_output=True, text=True).stdout.splitlines():
        if "principal" in line.lower():
            print(f"Kerberos: {line.strip()}")
            break

    print(f"\nLogging in to: {', '.join(targets)}")
    print("You will need your CERN TOTP code (only once — session is shared).\n")

    # Single shared session — Keycloak SSO cookies carry over between services
    session = requests.Session()
    # Build a CA bundle that covers BOTH public CAs (for auth.cern.ch and
    # most *.cern.ch) AND the CERN Grid CA (for cms-conddb.cern.ch etc.).
    # Setting session.verify to just the Grid CA makes auth.cern.ch fail;
    # just the default makes cms-conddb fail. Merged = both work.
    cern_ca = os.environ.get("CERN_GRID_CA")
    if cern_ca and Path(cern_ca).is_file():
        import tempfile
        import certifi
        merged = tempfile.NamedTemporaryFile(
            mode="w", suffix=".pem", delete=False, prefix="cern-merged-ca-"
        )
        with open(certifi.where()) as default_ca:
            merged.write(default_ca.read())
        merged.write("\n")
        with open(cern_ca) as grid_ca:
            merged.write(grid_ca.read())
        merged.close()
        session.verify = merged.name
    krb = HTTPKerberosAuth(mutual_authentication=OPTIONAL)

    success, failed = [], []

    for name in targets:
        svc = SERVICES[name]
        print(f"[{name}] Authenticating...")

        try:
            ok = _complete_oidc(session, svc["url"], krb)
        except Exception as exc:
            print(f"[{name}] ✗ {type(exc).__name__}: {exc}")
            failed.append(name)
            continue

        if ok:
            total, svc_count = _save_service_cookies(
                session, name, svc["url"], directory=directory
            )
            print(f"[{name}] ✓ {total} cookies saved ({svc_count} service-specific) → {directory / name}.txt")
            success.append(name)
        else:
            print(f"[{name}] ✗ Authentication failed")
            failed.append(name)

    print(f"\n{'='*50}")
    if success:
        print(f"✓ Logged in: {', '.join(success)}")
        print(f"\nCookie files saved to {directory}/")
        print("Add to .env:")
        for name in success:
            env_var = ENV_VARS.get(name, f"{name.upper()}_COOKIE_FILE")
            print(f"  {env_var}={directory / name}.txt")
    if failed:
        print(f"✗ Failed:    {', '.join(failed)}")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
