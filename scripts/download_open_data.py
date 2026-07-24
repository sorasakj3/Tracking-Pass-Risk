from __future__ import annotations

import urllib.request
from pathlib import Path

BASE = (
    "https://github.com/metrica-sports/sample-data/raw/refs/heads/master/"
    "data/Sample_Game_1"
)
FILES = {
    "events.csv": "Sample_Game_1_RawEventsData.csv",
    "home.csv": "Sample_Game_1_RawTrackingData_Home_Team.csv",
    "away.csv": "Sample_Game_1_RawTrackingData_Away_Team.csv",
}


def main() -> None:
    target = Path("data/raw")
    target.mkdir(parents=True, exist_ok=True)
    for local_name, remote_name in FILES.items():
        destination = target / local_name
        if not destination.exists():
            print(f"Downloading {remote_name}")
            urllib.request.urlretrieve(f"{BASE}/{remote_name}", destination)


if __name__ == "__main__":
    main()

