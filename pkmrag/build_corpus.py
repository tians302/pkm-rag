"""Build a linked multilingual corpus from the raw PokeAPI CSVs.

Outputs (in data/processed/):
  names_link.json  -- species_id -> names in en / ja / ja-hrkt / zh-hans / zh-hant
                      (the explicit translation linkage table)
  corpus.jsonl     -- one RAG document per (species, language):
                      a localized "fact card" (names in all languages, genus,
                      types, abilities, stats) + deduplicated Pokedex entries.

Language IDs in the PokeAPI database:
  9 = en, 11 = ja (kanji/kana), 1 = ja-hrkt (katakana), 12 = zh-hans, 4 = zh-hant
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT = Path(__file__).resolve().parent.parent / "data" / "processed"

LANGS = {9: "en", 11: "ja", 1: "ja-hrkt", 12: "zh-hans", 4: "zh-hant"}
# Languages we build documents in (ja-hrkt is kept only as a linked alias)
DOC_LANGS = ["en", "ja", "zh-hans", "zh-hant"]

STAT_LABELS = {1: "HP", 2: "Attack", 3: "Defense",
               4: "Sp. Atk", 5: "Sp. Def", 6: "Speed"}

TEMPLATES = {
    "en": ("{name} (National Dex #{sid}) is the {genus}. "
           "Type: {types}. Abilities: {abilities}. "
           "Height {h} m, weight {w} kg. Base stats: {stats}. "
           "Names: English {en}, Japanese {ja}, "
           "Simplified Chinese {zhs}, Traditional Chinese {zht}."),
    "ja": ("{name}（全国図鑑 No.{sid}）は{genus}。"
           "タイプ: {types}。特性: {abilities}。"
           "高さ {h} m、重さ {w} kg。種族値: {stats}。"
           "名前: 英語 {en}、日本語 {ja}、簡体字中国語 {zhs}、繁体字中国語 {zht}。"),
    "zh-hans": ("{name}（全国图鉴 #{sid}）是{genus}。"
                "属性: {types}。特性: {abilities}。"
                "身高 {h} 米，体重 {w} 千克。种族值: {stats}。"
                "名字: 英语 {en}，日语 {ja}，简体中文 {zhs}，繁体中文 {zht}。"),
    "zh-hant": ("{name}（全國圖鑑 #{sid}）是{genus}。"
                "屬性: {types}。特性: {abilities}。"
                "身高 {h} 米，體重 {w} 公斤。種族值: {stats}。"
                "名字: 英語 {en}，日語 {ja}，簡體中文 {zhs}，繁體中文 {zht}。"),
}


def read_csv(name: str):
    with open(RAW / name, newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def clean(text: str) -> str:
    """Flavor text in the games contains hard line breaks and soft hyphens."""
    return (text.replace("\u00ad\n", "").replace("\u00ad\f", "")
                .replace("\n", " ").replace("\f", " ").strip())


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # --- translation linkage: names + genus per species per language -------
    names = defaultdict(dict)   # sid -> lang -> name
    genus = defaultdict(dict)
    for row in read_csv("pokemon_species_names.csv"):
        lang = LANGS.get(int(row["local_language_id"]))
        if lang is None:
            continue
        sid = int(row["pokemon_species_id"])
        names[sid][lang] = row["name"]
        if row.get("genus"):
            genus[sid][lang] = row["genus"]

    with open(OUT / "names_link.json", "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False, indent=1)

    # --- localized type / ability names ------------------------------------
    type_names = defaultdict(dict)
    for row in read_csv("type_names.csv"):
        lang = LANGS.get(int(row["local_language_id"]))
        if lang:
            type_names[int(row["type_id"])][lang] = row["name"]

    ability_names = defaultdict(dict)
    for row in read_csv("ability_names.csv"):
        lang = LANGS.get(int(row["local_language_id"]))
        if lang:
            ability_names[int(row["ability_id"])][lang] = row["name"]

    # --- default pokemon per species: physical + battle data ---------------
    default_pk = {}             # species_id -> pokemon_id (is_default)
    phys = {}                   # pokemon_id -> (height_m, weight_kg)
    for row in read_csv("pokemon.csv"):
        if row["is_default"] == "1":
            default_pk[int(row["species_id"])] = int(row["id"])
            phys[int(row["id"])] = (int(row["height"]) / 10,
                                    int(row["weight"]) / 10)

    pk_types = defaultdict(list)
    for row in read_csv("pokemon_types.csv"):
        pk_types[int(row["pokemon_id"])].append(
            (int(row["slot"]), int(row["type_id"])))

    pk_abilities = defaultdict(list)
    for row in read_csv("pokemon_abilities.csv"):
        pk_abilities[int(row["pokemon_id"])].append(
            (int(row["slot"]), int(row["ability_id"])))

    pk_stats = defaultdict(dict)
    for row in read_csv("pokemon_stats.csv"):
        pk_stats[int(row["pokemon_id"])][int(row["stat_id"])] = \
            int(row["base_stat"])

    # --- Pokedex flavor text, deduplicated per (species, language) ---------
    flavor = defaultdict(lambda: defaultdict(list))  # sid -> lang -> [texts]
    for row in read_csv("pokemon_species_flavor_text.csv"):
        lang = LANGS.get(int(row["language_id"]))
        if lang not in DOC_LANGS:
            continue
        sid = int(row["species_id"])
        text = clean(row["flavor_text"])
        if text and text not in flavor[sid][lang]:
            flavor[sid][lang].append(text)

    # --- emit documents -----------------------------------------------------
    n_docs = 0
    with open(OUT / "corpus.jsonl", "w", encoding="utf-8") as out:
        for sid in sorted(names):
            pkid = default_pk.get(sid)
            if pkid is None:
                continue
            h, w = phys.get(pkid, (None, None))
            stats = pk_stats.get(pkid, {})
            for lang in DOC_LANGS:
                if lang not in names[sid]:
                    continue
                nm = names[sid]
                tps = ", ".join(
                    type_names[t].get(lang, type_names[t].get("en", "?"))
                    for _, t in sorted(pk_types.get(pkid, [])))
                abl = ", ".join(
                    ability_names[a].get(lang, ability_names[a].get("en", "?"))
                    for _, a in sorted(pk_abilities.get(pkid, [])))
                st = ", ".join(f"{STAT_LABELS[k]} {v}"
                               for k, v in sorted(stats.items()))
                card = TEMPLATES[lang].format(
                    name=nm[lang], sid=sid,
                    genus=genus[sid].get(lang, ""),
                    types=tps, abilities=abl, h=h, w=w, stats=st,
                    en=nm.get("en", "?"), ja=nm.get("ja", "?"),
                    zhs=nm.get("zh-hans", "?"), zht=nm.get("zh-hant", "?"))
                dex = " ".join(flavor[sid][lang][:6])   # cap doc length
                doc = {
                    "id": f"{sid}-{lang}",
                    "species_id": sid,
                    "lang": lang,
                    "name": nm[lang],
                    "text": card + (" Pokedex: " + dex if dex else ""),
                }
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                n_docs += 1
    print(f"Wrote {n_docs} documents -> {OUT / 'corpus.jsonl'}")
    print(f"Linked names for {len(names)} species -> {OUT / 'names_link.json'}")


if __name__ == "__main__":
    build()
