"""Ping Supabase pour eviter l'hibernation du free tier (~7j d'inactivite).

Le schema pocket_arbiter dans la DB Supabase nsvjaebywqdusznthayk est un MIRROR
du corpus local corpus_v2_fr.db, pousse le 16/04/2026 dans le seul but de generer
de l'activite et empecher Supabase de supprimer la DB pour inactivite.

A lancer manuellement au moins une fois par semaine, ou via Task Scheduler Windows.

Usage:
    python scripts/maintenance/ping_supabase.py

Aucune dependance externe (stdlib uniquement).
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path


def load_env(env_path: Path) -> dict[str, str]:
    """Parse minimal .env file (KEY=VALUE, ignore comments and blank lines)."""
    env: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def main() -> int:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        print(f"[ERROR] .env introuvable: {env_path}", file=sys.stderr)
        return 1

    env = load_env(env_path)
    url = env.get("SUPABASE_URL")
    key = env.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        print(
            "[ERROR] SUPABASE_URL ou SUPABASE_SERVICE_ROLE_KEY manquant dans .env",
            file=sys.stderr,
        )
        return 1

    # ISO 27001 — validation explicite du scheme avant urllib.urlopen (cf. ruff S310).
    # On accepte uniquement HTTPS pour eviter file:// et schemes custom potentiellement dangereux.
    if not url.startswith("https://"):
        print(
            f"[ERROR] SUPABASE_URL doit utiliser HTTPS, recu: {url[:20]}...",
            file=sys.stderr,
        )
        return 1

    # Hit du root PostgREST : force un round-trip Postgres pour introspect le schema,
    # donc compte comme activite cote Supabase free tier (anti-hibernation).
    # Pas de SELECT explicite car le schema 'pocket_arbiter' n'est pas expose via PostgREST
    # (configuration Supabase par defaut : seul 'public' est expose).
    target_url = f"{url}/rest/v1/"
    req = urllib.request.Request(  # noqa: S310 — scheme validated above
        target_url,
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 — scheme validated above
            size = r.headers.get("Content-Length", "?")
            print(
                f"[OK] HTTP {r.status} | PostgREST root pinged ({size} bytes) — activite enregistree"
            )
            return 0
    except urllib.error.HTTPError as e:
        print(
            f"[HTTP {e.code}] {e.reason} — DB peut-etre en cours de restore, retenter dans qq minutes",
            file=sys.stderr,
        )
        return 2
    except urllib.error.URLError as e:
        print(
            f"[NETWORK ERROR] {e.reason} — DB probablement hibernee, attendre auto-restore",
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    sys.exit(main())
