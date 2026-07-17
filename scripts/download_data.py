"""Download the multilingual Pokemon CSV tables from the PokeAPI repo.

All files come from https://github.com/PokeAPI/pokeapi (data/v2/csv),
which is the same veekun-derived database that powers pokeapi.co.
"""

import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/PokeAPI/pokeapi/master/data/v2/csv"

FILES = [
    "languages.csv",
    "pokemon_species.csv",
    "pokemon_species_names.csv",       # names + genus, all languages
    "pokemon_species_flavor_text.csv", # Pokedex entries, all languages
    "pokemon.csv",                     # height / weight / default forms
    "pokemon_types.csv",
    "pokemon_stats.csv",
    "pokemon_abilities.csv",
    "types.csv",
    "type_names.csv",                  # localized type names
    "type_efficacy.csv",               # attack/defense damage factors
    "abilities.csv",
    "ability_names.csv",               # localized ability names
    "versions.csv",
    "version_names.csv",
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for fname in FILES:
        dest = DATA_DIR / fname
        if dest.exists():
            print(f"[skip] {fname}")
            continue
        url = f"{BASE}/{fname}"
        print(f"[get ] {url}")
        urllib.request.urlretrieve(url, dest)
    print(f"Done. Files in {DATA_DIR}")


if __name__ == "__main__":
    main()
