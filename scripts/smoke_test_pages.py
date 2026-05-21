"""Smoke test des pages Streamlit via streamlit.testing.AppTest.

Verifie que chaque page execute sans exception avec le theme par defaut.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from streamlit.testing.v1 import AppTest


PAGES = [
    "app/streamlit_app.py",
    "app/pages/2_prediction_dossier.py",
    "app/pages/3_alertes.py",
]


def test_page(page_path: str) -> tuple[bool, list]:
    at = AppTest.from_file(page_path, default_timeout=60)
    at.run()
    return not at.exception, list(at.exception)


def main() -> int:
    failed = 0
    for page in PAGES:
        ok, exceptions = test_page(page)
        status = "[OK]" if ok else "[FAIL]"
        print(f"{status} {page}")
        if not ok:
            failed += 1
            for exc in exceptions:
                print(f"  -> {exc.value}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
