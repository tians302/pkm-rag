"""Build a linked multilingual corpus from the raw PokeAPI CSVs.

Outputs (in data/processed/):
  names_link.json  -- species_id -> names in en / ja / ja-hrkt / zh-hans / zh-hant
                      (the explicit translation linkage table)
  corpus.jsonl     -- one RAG document per (species, language):
                      a localized "fact card" (names in all languages, genus,
                      types, abilities, stats, type matchups) + deduplicated
                      Pokedex entries. Plus one type-chart document per
                      (type, language) ("kind": "type", species_id 0) so
                      effectiveness questions have something to retrieve.

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


# Localized fragments for type-matchup text. The multiplier buckets use the
# official game wording (super effective / こうかばつぐん / 效果绝佳) so that
# natural questions share vocabulary with the documents.
SEP = {"en": ", ", "ja": "・", "zh-hans": "、", "zh-hant": "、"}

MATCHUP_DEF = {   # per-species defensive line, appended to the fact card
    "en": (" Type matchup (defense): {parts}."),
    "ja": ("相性（防御）: {parts}。"),
    "zh-hans": ("属性相性（防御）: {parts}。"),
    "zh-hant": ("屬性相性（防禦）: {parts}。"),
}
DEF_BUCKETS = {   # multiplier -> localized phrase, {t} = type list
    "en": {400: "biggest weakness is {t} (double weakness, takes 4x damage)",
           200: "weak to {t} (2x damage)",
           50: "resists {t} (0.5x damage)", 25: "strongly resists {t} (0.25x)",
           0: "immune to {t} (no damage)"},
    "ja": {400: "最大の弱点は{t}（4倍のダメージ）",
           200: "弱点は{t}（こうかばつぐん、2倍）",
           50: "{t}はこうかいまひとつ（0.5倍）", 25: "{t}は0.25倍",
           0: "{t}はこうかなし（無効）"},
    "zh-hans": {400: "最大弱点是{t}（受到4倍伤害）",
                200: "弱点是{t}（效果绝佳，2倍）",
                50: "抵抗{t}（效果不好，0.5倍）", 25: "强抵抗{t}（0.25倍）",
                0: "免疫{t}（无效）"},
    "zh-hant": {400: "最大弱點是{t}（受到4倍傷害）",
                200: "弱點是{t}（效果絕佳，2倍）",
                50: "抵抗{t}（效果不好，0.5倍）", 25: "強抵抗{t}（0.25倍）",
                0: "免疫{t}（無效）"},
}

TYPE_DOC_NAME = {"en": "{t} type", "ja": "{t}タイプ",
                 "zh-hans": "{t}属性", "zh-hant": "{t}屬性"}
# Phrased the way questions are asked ("the most effective attacks against X
# are ..."), not the way charts are read ("X is weak to ..."): small local
# LLMs reliably answer from the first phrasing but flip directions on the
# second (gemma2:2b answered the offense row for a defense question).
TYPE_DOC = {      # standalone type-chart document
    "en": ("{t} type matchup chart. The most effective attacks against "
           "{t}-type Pokemon are {weak} attacks: they are super effective "
           "and deal 2x damage to {t} Pokemon. {res} attacks are not very "
           "effective against {t}-type Pokemon (0.5x damage){imm_d}. "
           "In the other direction, {t}-type attacks are super effective "
           "(2x damage) against {str_} Pokemon, and not very effective "
           "(0.5x damage) against {nve} Pokemon{imm_o}."),
    "ja": ("{t}タイプの相性表。{t}タイプのポケモンに最も効果的な技は{weak}タイプ: "
           "こうかばつぐんで、{t}ポケモンに2倍のダメージを与える。{res}タイプの技は"
           "{t}タイプにはこうかいまひとつ（0.5倍）{imm_d}。逆に、{t}タイプの技は"
           "{str_}タイプにこうかばつぐん（2倍）、{nve}タイプにはこうかいまひとつ"
           "（0.5倍）{imm_o}。"),
    "zh-hans": ("{t}属性相性表。对{t}属性宝可梦最有效的招式是{weak}属性: "
                "效果绝佳，对{t}宝可梦造成2倍伤害。{res}属性招式对{t}属性"
                "效果不好（0.5倍）{imm_d}。反过来，{t}属性招式对{str_}属性宝可梦"
                "效果绝佳（2倍伤害），对{nve}属性效果不好（0.5倍）{imm_o}。"),
    "zh-hant": ("{t}屬性相性表。對{t}屬性寶可夢最有效的招式是{weak}屬性: "
                "效果絕佳，對{t}寶可夢造成2倍傷害。{res}屬性招式對{t}屬性"
                "效果不好（0.5倍）{imm_d}。反過來，{t}屬性招式對{str_}屬性寶可夢"
                "效果絕佳（2倍傷害），對{nve}屬性效果不好（0.5倍）{imm_o}。"),
}
TYPE_DOC_IMM_D = {"en": "; {t} attacks have no effect on {t2}-type Pokemon",
                  "ja": "。{t}タイプの技は{t2}タイプにこうかなし（無効）",
                  "zh-hans": "。{t}属性招式对{t2}属性无效",
                  "zh-hant": "。{t}屬性招式對{t2}屬性無效"}
TYPE_DOC_IMM_O = {"en": ", and have no effect on {t} Pokemon",
                  "ja": "。{t}タイプにはこうかなし",
                  "zh-hans": "，对{t}属性没有效果",
                  "zh-hant": "，對{t}屬性沒有效果"}

MOVE_DOC = {      # one document per (move, language)
    "en": ("{name} is a{n} {type}-type {cls} move (power {power}, "
           "accuracy {acc}, PP {pp}). Names: English {en}, Japanese {ja}, "
           "Simplified Chinese {zhs}, Traditional Chinese {zht}. "
           "Effect: {desc}"),
    "ja": ("{name}は{type}タイプの{cls}わざ（威力 {power}、命中 {acc}、"
           "PP {pp}）。名前: 英語 {en}、日本語 {ja}、簡体字中国語 {zhs}、"
           "繁体字中国語 {zht}。効果: {desc}"),
    "zh-hans": ("{name}是{type}属性的{cls}招式（威力 {power}，命中 {acc}，"
                "PP {pp}）。名字: 英语 {en}，日语 {ja}，简体中文 {zhs}，"
                "繁体中文 {zht}。效果: {desc}"),
    "zh-hant": ("{name}是{type}屬性的{cls}招式（威力 {power}，命中 {acc}，"
                "PP {pp}）。名字: 英語 {en}，日語 {ja}，簡體中文 {zhs}，"
                "繁體中文 {zht}。效果: {desc}"),
}
LEARNSET = {      # appended to the species fact card
    "en": " Moves learned by leveling up: {mv}.",
    "ja": "レベルアップで覚えるわざ: {mv}。",
    "zh-hans": "升级学会的招式: {mv}。",
    "zh-hant": "升級學會的招式: {mv}。",
}
MOVE_ENTRY = {"en": "{n} (Lv. {lv})", "ja": "{n}（Lv.{lv}）",
              "zh-hans": "{n}（Lv.{lv}）", "zh-hant": "{n}（Lv.{lv}）"}


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

    # --- type effectiveness matrix (damage_factor: 0 / 50 / 100 / 200) -----
    efficacy = {}               # (attack_type_id, defend_type_id) -> factor
    for row in read_csv("type_efficacy.csv"):
        efficacy[(int(row["damage_type_id"]),
                  int(row["target_type_id"]))] = int(row["damage_factor"])
    battle_types = sorted({a for a, _ in efficacy})   # the 18 battle types

    def tname(tid: int, lang: str) -> str:
        return type_names[tid].get(lang, type_names[tid].get("en", "?"))

    def defense_line(def_types: list[int], lang: str) -> str:
        """Localized weakness/resistance sentence for a (dual-)type combo."""
        buckets = defaultdict(list)   # combined multiplier (x100) -> type ids
        for atk in battle_types:
            m = 100
            for d in def_types:
                m = m * efficacy.get((atk, d), 100) // 100
            if m != 100:
                buckets[m].append(atk)
        if not buckets:
            return ""
        parts = [DEF_BUCKETS[lang][m].format(
                     t=SEP[lang].join(tname(t, lang) for t in buckets[m]))
                 for m in (400, 200, 50, 25, 0) if m in buckets]
        joiner = "; " if lang == "en" else "；"
        return MATCHUP_DEF[lang].format(parts=joiner.join(parts))

    # --- moves: stats, localized names/descriptions, learnsets -------------
    move_info = {}              # move_id -> (type_id, damage_class_id, p, a, pp)
    for row in read_csv("moves.csv"):
        mid = int(row["id"])
        if mid >= 10000 or int(row["type_id"]) > 18:
            continue            # shadow / special internal moves
        move_info[mid] = (int(row["type_id"]), int(row["damage_class_id"]),
                          row["power"] or "—", row["accuracy"] or "—",
                          row["pp"] or "—")

    move_names = defaultdict(dict)
    for row in read_csv("move_names.csv"):
        lang = LANGS.get(int(row["local_language_id"]))
        if lang and int(row["move_id"]) in move_info:
            move_names[int(row["move_id"])][lang] = row["name"]

    class_names = defaultdict(dict)   # 1 status / 2 physical / 3 special
    for row in read_csv("move_damage_class_prose.csv"):
        lang = LANGS.get(int(row["local_language_id"]))
        if lang:
            class_names[int(row["move_damage_class_id"])][lang] = row["name"]

    # newest description per (move, lang)
    move_desc = defaultdict(dict)     # move_id -> lang -> (vg, text)
    for row in read_csv("move_flavor_text.csv"):
        lang = LANGS.get(int(row["language_id"]))
        mid = int(row["move_id"])
        if lang not in DOC_LANGS or mid not in move_info:
            continue
        vg = int(row["version_group_id"])
        if vg >= move_desc[mid].get(lang, (0, ""))[0]:
            move_desc[mid][lang] = (vg, clean(row["flavor_text"]))

    # level-up learnset from each pokemon's newest version group
    lv_moves = defaultdict(lambda: defaultdict(list))  # pkid -> vg -> entries
    for row in read_csv("pokemon_moves.csv"):
        if row["pokemon_move_method_id"] != "1":
            continue                                   # level-up only
        lvl = int(row["level"])
        mid = int(row["move_id"])
        if lvl >= 1 and mid in move_info:
            lv_moves[int(row["pokemon_id"])][int(
                row["version_group_id"])].append((lvl, mid))

    def learnset_line(pkid: int, lang: str) -> str:
        if not lv_moves.get(pkid):
            return ""
        vg = max(lv_moves[pkid])
        seen, entries = set(), []
        for lvl, mid in sorted(lv_moves[pkid][vg]):
            nm = move_names[mid].get(lang, move_names[mid].get("en"))
            if nm and mid not in seen:
                seen.add(mid)
                entries.append(MOVE_ENTRY[lang].format(n=nm, lv=lvl))
        if not entries:
            return ""
        return LEARNSET[lang].format(mv=SEP[lang].join(entries))

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
                matchup = defense_line(
                    [t for _, t in sorted(pk_types.get(pkid, []))], lang)
                dex = " ".join(flavor[sid][lang][:6])   # cap doc length
                doc = {
                    "id": f"{sid}-{lang}",
                    "species_id": sid,
                    "kind": "species",
                    "lang": lang,
                    "name": nm[lang],
                    "text": card + matchup + learnset_line(pkid, lang)
                            + (" Pokedex: " + dex if dex else ""),
                }
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                n_docs += 1

        # --- one type-chart document per (type, language) -------------------
        for tid in battle_types:
            weak = [a for a in battle_types if efficacy[(a, tid)] == 200]
            res = [a for a in battle_types if efficacy[(a, tid)] == 50]
            imm_d = [a for a in battle_types if efficacy[(a, tid)] == 0]
            str_ = [d for d in battle_types if efficacy[(tid, d)] == 200]
            nve = [d for d in battle_types if efficacy[(tid, d)] == 50]
            imm_o = [d for d in battle_types if efficacy[(tid, d)] == 0]
            for lang in DOC_LANGS:
                j = lambda ts: SEP[lang].join(tname(t, lang) for t in ts)
                text = TYPE_DOC[lang].format(
                    t=tname(tid, lang), weak=j(weak), res=j(res),
                    str_=j(str_), nve=j(nve),
                    imm_d=TYPE_DOC_IMM_D[lang].format(t=j(imm_d),
                                                      t2=tname(tid, lang))
                          if imm_d else "",
                    imm_o=TYPE_DOC_IMM_O[lang].format(t=j(imm_o))
                          if imm_o else "")
                doc = {
                    "id": f"type-{tid}-{lang}",
                    "species_id": 0,          # sentinel: not a species doc
                    "kind": "type",
                    "lang": lang,
                    "name": TYPE_DOC_NAME[lang].format(t=tname(tid, lang)),
                    # all-language names, for the retriever's alias boost
                    # (same trick as species names in names_link.json)
                    "aliases": sorted({type_names[tid][l].lower()
                                       for l in DOC_LANGS
                                       if l in type_names[tid]}),
                    "text": text,
                }
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                n_docs += 1

        # --- one document per (move, language) ------------------------------
        for mid in sorted(move_info):
            tid, cls, power, acc, pp = move_info[mid]
            mnm = move_names[mid]
            for lang in DOC_LANGS:
                if lang not in mnm:
                    continue
                tn = tname(tid, lang)
                text = MOVE_DOC[lang].format(
                    name=mnm[lang], type=tn,
                    n="n" if lang == "en" and tn[:1] in "AEIOU" else "",
                    cls=class_names[cls].get(lang, class_names[cls]["en"]),
                    power=power, acc=acc, pp=pp,
                    en=mnm.get("en", "?"), ja=mnm.get("ja", "?"),
                    zhs=mnm.get("zh-hans", "?"), zht=mnm.get("zh-hant", "?"),
                    desc=move_desc[mid].get(lang, (0, ""))[1])
                doc = {
                    "id": f"move-{mid}-{lang}",
                    "species_id": 0,
                    "kind": "move",
                    "lang": lang,
                    "name": mnm[lang],
                    "aliases": sorted({n.lower() for n in mnm.values()}),
                    "text": text,
                }
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                n_docs += 1
    print(f"Wrote {n_docs} documents -> {OUT / 'corpus.jsonl'}")
    print(f"Linked names for {len(names)} species -> {OUT / 'names_link.json'}")


if __name__ == "__main__":
    build()
