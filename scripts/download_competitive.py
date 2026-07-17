"""Download competitive battling data (English only).

Sources:
  - Smogon/Showdown moveset sets via the pkmn project's curated mirror
    (https://github.com/pkmn/smogon -> data.pkmn.cc), MIT-licensed tooling
  - Tier placements parsed from Pokemon Showdown's formats-data.ts
    (https://github.com/smogon/pokemon-showdown, MIT)

Outputs (in data/raw/):
  smogon_sets_gen9.json   -- species -> format -> set name -> set details
  showdown_tiers.json     -- showdown id -> {"tier": ..., "doublesTier": ...}
"""

import json
import re
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

SETS_URL = "https://data.pkmn.cc/sets/gen9.json"
TIERS_URL = ("https://raw.githubusercontent.com/smogon/pokemon-showdown/"
             "master/data/formats-data.ts")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)

    dest = RAW / "smogon_sets_gen9.json"
    if not dest.exists():
        print(f"[get ] {SETS_URL}")
        urllib.request.urlretrieve(SETS_URL, dest)
    else:
        print(f"[skip] {dest.name}")

    dest = RAW / "showdown_tiers.json"
    if not dest.exists():
        print(f"[get ] {TIERS_URL}")
        ts = urllib.request.urlopen(TIERS_URL).read().decode()
        tiers = {}
        # entries look like:  garchomp: {\n\t\ttier: "UUBL",\n\t\tdoublesTier: "DUU",
        for m in re.finditer(
                r"^\t(\w+): \{([^}]*)\}", ts, re.M | re.S):
            body = m.group(2)
            t = re.search(r'\btier: "([^"]+)"', body)
            dt = re.search(r'\bdoublesTier: "([^"]+)"', body)
            if t or dt:
                tiers[m.group(1)] = {
                    **({"tier": t.group(1)} if t else {}),
                    **({"doublesTier": dt.group(1)} if dt else {})}
        json.dump(tiers, open(dest, "w"), indent=0)
        print(f"parsed {len(tiers)} tier entries")
    else:
        print(f"[skip] {dest.name}")
    print(f"Done. Files in {RAW}")


if __name__ == "__main__":
    main()
