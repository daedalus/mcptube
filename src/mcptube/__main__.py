"""CLI entry point for mcptube."""

from mcptube.cli import app


def main() -> int:
    return app()


if __name__ == "__main__":
    raise SystemExit(main())
