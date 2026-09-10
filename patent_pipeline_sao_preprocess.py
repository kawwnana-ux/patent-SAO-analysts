# patent_pipeline.py
# 日本語特許請求項 SAO 分析パイプライン
#
# 方針:
# 1. GiNZAで形態素・係り受けを解析
# 2. 助詞を捨てずに文法役割として利用
# 3. 連体修飾「～に設けられたB」「～に接続されたC」を関係化
# 4. 受動表現をSAO向けに正規化
# 5. 「前記」「同」「当該」などの先行構成要素を参照
# 6. 「Aと、Bと、Cとを有する」の列挙を処理
# 7. 複合名詞を構成要素としてまとめる
# 8. 元文を壊さず、抽出結果と正規化結果を分離
#
# 必要:
#   pip install spacy ginza ja-ginza
#
# 例:
#   from patent_pipeline import analyze_claim
#   components, relations = analyze_claim(text)
#
# 返却:
#   components: 構成要素のリスト
#   relations : SAO関係のリスト
#
# relation の形式:
#   {
#       "subject": "...",
#       "action": "...",
#       "object": "...",
#       "type": "...",
#       "source": "..."
#   }

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import spacy


# ============================================================
# GiNZA
# ============================================================

_NLP = None


def get_nlp():
    """GiNZAモデルを遅延ロードする。"""
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("ja_ginza")
        except Exception as e:
            raise RuntimeError(
                "GiNZAを読み込めませんでした。"
                " `pip install -U spacy ginza ja-ginza` を確認してください。"
            ) from e
    return _NLP


# ============================================================
# データ構造
# ============================================================

@dataclass
class Relation:
    subject: str
    action: str
    object: str
    type: str = "sao"
    source: str = ""

    def as_dict(self) -> Dict[str, str]:
        return {
            "subject": self.subject,
            "action": self.action,
            "object": self.object,
            "type": self.type,
            "source": self.source,
        }


# ============================================================
# 基本ユーティリティ
# ============================================================

REFERENCE_PREFIXES = (
    "前記",
    "前述の",
    "上述の",
    "上記の",
    "同",
    "当該",
)

CLAUSE_ENDINGS = ("。", "；", ";")

# 特許で関係として扱いやすい表現
RELATION_NORMALIZATION = {
    "設けられる": "設ける",
    "設けられた": "設ける",
    "設けられている": "設ける",
    "配置される": "配置する",
    "配置された": "配置する",
    "配置されている": "配置する",
    "接続される": "接続する",
    "接続された": "接続する",
    "接続されている": "接続する",
    "結合される": "結合する",
    "結合された": "結合する",
    "結合されている": "結合する",
    "形成される": "形成する",
    "形成された": "形成する",
    "形成されている": "形成する",
    "含まれる": "含む",
    "備えられる": "備える",
    "備えた": "備える",
    "有する": "有する",
    "有した": "有する",
    "構成される": "構成する",
    "構成された": "構成する",
    "取り付けられる": "取り付ける",
    "取り付けられた": "取り付ける",
    "収容される": "収容する",
    "収容された": "収容する",
    "載置される": "載置する",
    "載置された": "載置する",
    "接触する": "接触する",
    "接触された": "接触する",
}


def clean_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_action(action: str) -> str:
    action = clean_text(action)
    return RELATION_NORMALIZATION.get(action, action)


def remove_reference_prefix(text: str) -> str:
    s = clean_text(text)
    for p in REFERENCE_PREFIXES:
        if s.startswith(p):
            return s[len(p):].strip()
    return s


def is_reference(text: str) -> bool:
    s = clean_text(text)
    return any(s.startswith(p) for p in REFERENCE_PREFIXES)


def unique_preserve(items: List[str]) -> List[str]:
    out = []
    seen = set()
    for x in items:
        x = clean_text(x)
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


# ============================================================
# 名詞句取得
# ============================================================

def noun_phrase(token) -> str:
    """
    可能な範囲で複合名詞・数量表現等をまとめる。
    係り受けを壊しすぎないよう、tokenのsubtree全体ではなく
    名詞中心の局所範囲を優先する。
    """
    doc = token.doc

    if token.pos_ not in ("NOUN", "PROPN"):
        return token.text

    # 連続する名詞・接頭辞・記号等をまとめる
    start = token.i
    end = token.i + 1

    # 左側の名詞連鎖
    i = token.i - 1
    while i >= 0:
        t = doc[i]
        if t.pos_ in ("NOUN", "PROPN") or t.pos_ == "NUM":
            start = i
            i -= 1
            continue
        if t.pos_ == "ADJ" and i == start - 1:
            # 技術用語の形容詞修飾を限定的に含める
            start = i
            i -= 1
            continue
        break

    # 右側の名詞連鎖
    i = token.i + 1
    while i < len(doc):
        t = doc[i]
        if t.pos_ in ("NOUN", "PROPN") or t.pos_ == "NUM":
            end = i + 1
            i += 1
            continue
        break

    phrase = doc[start:end].text.strip()

    # 不要な空白を除去
    phrase = re.sub(r"\s+", "", phrase)
    return phrase


def expand_noun_phrase(token) -> str:
    """
    名詞の子要素から「第1の基板」「半導体スイッチング素子」などを
    できるだけ自然な構成要素として取得する。
    """
    doc = token.doc
    base = noun_phrase(token)

    # 「前記」等は構成要素名から除去
    base = remove_reference_prefix(base)

    # tokenにかかる名詞修飾を限定的に追加
    modifiers = []
    for child in token.children:
        if child.i >= token.i:
            continue
        if child.pos_ in ("ADJ", "NOUN", "PROPN", "NUM"):
            # すでにbaseに含まれている場合は追加しない
            c = clean_text(child.text)
            if c and c not in base:
                modifiers.append((child.i, c))

    if modifiers:
        modifiers.sort()
        extra = "".join(x[1] for x in modifiers)
        if extra and extra not in base:
            base = extra + base

    return clean_text(base)


# ============================================================
# 参照解決
# ============================================================

def build_component_memory(doc) -> List[str]:
    """
    文書内に出現した構成要素候補を左から記憶する。
    完全な共参照解析ではなく、特許請求項で頻出する
    「前記X」の軽量な先行詞解決を行う。
    """
    memory = []

    for token in doc:
        if token.pos_ not in ("NOUN", "PROPN"):
            continue

        phrase = expand_noun_phrase(token)
        if not phrase:
            continue

        # 明らかな一般語はある程度抑える
        if phrase in {"こと", "もの", "場合", "ため", "よう", "ところ"}:
            continue

        memory.append(phrase)

    return unique_preserve(memory)


def resolve_reference(text: str, memory: List[str]) -> str:
    """
    「前記第2の半導体層」→「第2の半導体層」
    ただし、メモリに完全一致する候補があればそれを優先。
    """
    raw = clean_text(text)
    stripped = remove_reference_prefix(raw)

    if stripped in memory:
        return stripped

    # 記号・空白を除いて比較
    compact = stripped.replace(" ", "")
    for m in reversed(memory):
        if m.replace(" ", "") == compact:
            return m

    return stripped


# ============================================================
# 助詞ベースの役割判定
# ============================================================

def particle_of(token):
    """
    tokenに直接接続する助詞を探す。
    """
    for child in token.children:
        if child.pos_ == "ADP":
            return child.text
    return None


def role_from_particle(token) -> Optional[str]:
    p = particle_of(token)

    if p == "が":
        return "subject"

    if p == "を":
        return "object"

    # 「に」は目的地・対象・場所など複数の意味があるため、
    # 一律objectにはしない。連体修飾関係で別途処理する。
    if p == "に":
        return "oblique"

    return None


# ============================================================
# 連体修飾「～されたB」の解析
# ============================================================

PASSIVE_PATTERNS = [
    (re.compile(r"(.+?)に(.+?)接続された(.+)"), "接続する"),
    (re.compile(r"(.+?)に(.+?)接続される(.+)"), "接続する"),
    (re.compile(r"(.+?)に(.+?)設けられた(.+)"), "設ける"),
    (re.compile(r"(.+?)に(.+?)設けられる(.+)"), "設ける"),
    (re.compile(r"(.+?)に(.+?)配置された(.+)"), "配置する"),
    (re.compile(r"(.+?)に(.+?)配置される(.+)"), "配置する"),
    (re.compile(r"(.+?)に(.+?)形成された(.+)"), "形成する"),
    (re.compile(r"(.+?)に(.+?)形成される(.+)"), "形成する"),
    (re.compile(r"(.+?)に(.+?)結合された(.+)"), "結合する"),
    (re.compile(r"(.+?)に(.+?)結合される(.+)"), "結合する"),
]

# より単純な「Aに接続されたB」等
SIMPLE_MODIFIER_PATTERNS = [
    (re.compile(r"(?P<obj>.+?)に(?P<verb>接続された|接続される)(?P<subj>[^、，,。;；]+)"), "接続する"),
    (re.compile(r"(?P<obj>.+?)に(?P<verb>設けられた|設けられる)(?P<subj>[^、，,。;；]+)"), "設ける"),
    (re.compile(r"(?P<obj>.+?)に(?P<verb>配置された|配置される)(?P<subj>[^、，,。;；]+)"), "配置する"),
    (re.compile(r"(?P<obj>.+?)に(?P<verb>形成された|形成される)(?P<subj>[^、，,。;；]+)"), "形成する"),
    (re.compile(r"(?P<obj>.+?)に(?P<verb>結合された|結合される)(?P<subj>[^、，,。;；]+)"), "結合する"),
    (re.compile(r"(?P<obj>.+?)に(?P<verb>収容された|収容される)(?P<subj>[^、，,。;；]+)"), "収容する"),
]


def extract_modifier_relations(text: str, memory: List[str]) -> List[Relation]:
    relations = []

    for pattern, action in SIMPLE_MODIFIER_PATTERNS:
        for m in pattern.finditer(text):
            obj = resolve_reference(m.group("obj"), memory)
            subj = resolve_reference(m.group("subj"), memory)

            # 余計な接続語を削る
            subj = re.sub(r"^(と|及び|または|又は)\s*", "", subj).strip()

            if subj and obj and subj != obj:
                relations.append(
                    Relation(
                        subject=subj,
                        action=action,
                        object=obj,
                        type="modifier",
                        source=m.group(0),
                    )
                )

    return relations


# ============================================================
# 受動態の文法解析
# ============================================================

def extract_dependency_relations(sent, memory: List[str]) -> List[Relation]:
    relations = []

    for token in sent:
        if token.pos_ != "VERB":
            continue

        action = normalize_action(token.lemma_ if token.lemma_ else token.text)

        # 子要素からsubject/objectを取得
        subjects = []
        objects = []

        for child in token.children:
            if child.dep_ in ("nsubj", "nsubj:pass", "csubj"):
                if child.pos_ in ("NOUN", "PROPN"):
                    subjects.append(expand_noun_phrase(child))

            elif child.dep_ in ("obj", "iobj"):
                if child.pos_ in ("NOUN", "PROPN"):
                    objects.append(expand_noun_phrase(child))

        # 受動態では「によって」が意味上の主体になる場合がある
        by_candidates = []
        for child in token.children:
            if child.text == "によって":
                for gchild in child.children:
                    if gchild.pos_ in ("NOUN", "PROPN"):
                        by_candidates.append(expand_noun_phrase(gchild))

        if by_candidates:
            subjects = by_candidates

        # 通常のSVO
        for s in subjects:
            for o in objects:
                s = resolve_reference(s, memory)
                o = resolve_reference(o, memory)

                if s and o and s != o:
                    relations.append(
                        Relation(
                            subject=s,
                            action=action,
                            object=o,
                            type="dependency",
                            source=sent.text,
                        )
                    )

    return relations


# ============================================================
# 助詞からSVOを作る
# ============================================================

def extract_particle_sao(sent, memory: List[str]) -> List[Relation]:
    """
    GiNZAの係り受けだけに依存せず、
    「が」「を」を使ってS/O候補を拾う。

    S + が + O + を + V
    → S + V + O
    """
    relations = []

    verbs = [t for t in sent if t.pos_ == "VERB"]

    for verb in verbs:
        subjects = []
        objects = []

        for token in sent:
            if token.pos_ not in ("NOUN", "PROPN"):
                continue

            # 動詞に係る候補のみ
            if not (
                token.head == verb
                or token in list(verb.subtree)
            ):
                continue

            role = role_from_particle(token)
            phrase = expand_noun_phrase(token)

            if role == "subject":
                subjects.append(phrase)
            elif role == "object":
                objects.append(phrase)

        action = normalize_action(verb.lemma_ or verb.text)

        for s in subjects:
            for o in objects:
                s = resolve_reference(s, memory)
                o = resolve_reference(o, memory)

                if s and o and s != o:
                    relations.append(
                        Relation(
                            subject=s,
                            action=action,
                            object=o,
                            type="particle_sao",
                            source=sent.text,
                        )
                    )

    return relations


# ============================================================
# 「Aと、Bと、Cとを有する」列挙処理
# ============================================================

def split_enumeration(text: str) -> List[str]:
    """
    特許で頻出する列挙を分割する。
    例:
      Aと、Bと、Cとを有する
    → A / B / C
    """
    s = clean_text(text)

    # 「と、」「、」等を統一
    s = s.replace("，", "、")

    # 末尾の「とを」「と、」等を除外しながら分割
    parts = re.split(r"(?:と、|、|及び|並びに)", s)

    cleaned = []
    for p in parts:
        p = p.strip(" 、,，")
        p = re.sub(r"^(?:第[0-9０-９一二三四五六七八九十]+の)?\s*", lambda m: m.group(0), p)
        if p:
            cleaned.append(p)

    return unique_preserve(cleaned)


def extract_has_relations(text: str, memory: List[str]) -> List[Relation]:
    relations = []

    # 「装置がAと、Bと、Cとを有する」
    patterns = [
        re.compile(
            r"(?P<subject>[^。；;]+?)(?:が|は|であって、)"
            r"(?P<objects>[^。；;]+?)"
            r"(?:と)?(?:を)?(?P<action>有する|備える|含む|具備する)"
        ),
        re.compile(
            r"(?P<subject>[^。；;]+?)"
            r"(?:と|を)?(?P<objects>[^。；;]+?)"
            r"(?P<action>有する|備える|含む|具備する)"
        ),
    ]

    for pattern in patterns:
        for m in pattern.finditer(text):
            subject = resolve_reference(m.group("subject"), memory)
            object_text = m.group("objects")
            action = normalize_action(m.group("action"))

            # 明らかな長文誤爆を抑える
            if len(subject) > 120:
                continue

            # 列挙を分解
            candidates = split_enumeration(object_text)

            for obj in candidates:
                obj = re.sub(r"^(?:と|を)\s*", "", obj).strip()
                obj = resolve_reference(obj, memory)

                # 動詞や接続表現だけの候補を除外
                if not obj or obj in {"こと", "もの"}:
                    continue
                if obj == subject:
                    continue

                relations.append(
                    Relation(
                        subject=subject,
                        action=action,
                        object=obj,
                        type="has",
                        source=m.group(0),
                    )
                )

    return relations


# ============================================================
# 「AにBを設ける」等の直接関係
# ============================================================

def extract_oblique_relations(sent, memory: List[str]) -> List[Relation]:
    relations = []

    for verb in sent:
        if verb.pos_ != "VERB":
            continue

        action = normalize_action(verb.lemma_ or verb.text)

        # verbに直接係る名詞
        nouns = [
            t for t in sent
            if t.pos_ in ("NOUN", "PROPN")
            and (t.head == verb or t in list(verb.subtree))
        ]

        subjects = []
        objects = []

        for token in nouns:
            p = particle_of(token)
            phrase = expand_noun_phrase(token)

            if p == "を":
                objects.append(phrase)
            elif p == "が":
                subjects.append(phrase)
            elif p == "に":
                # 「設ける」「配置する」等では「に」が対象/配置先
                if action in {
                    "設ける", "配置する", "接続する",
                    "形成する", "結合する", "収容する",
                    "載置する", "取り付ける",
                }:
                    objects.append(phrase)

        # 明示主語がない場合、直前の構成要素を主体候補にする
        if not subjects and objects:
            before = [
                t for t in sent
                if t.i < verb.i and t.pos_ in ("NOUN", "PROPN")
                and particle_of(t) in ("が", "は")
            ]
            subjects = [expand_noun_phrase(t) for t in before]

        for s in subjects:
            for o in objects:
                s = resolve_reference(s, memory)
                o = resolve_reference(o, memory)
                if s and o and s != o:
                    relations.append(
                        Relation(
                            subject=s,
                            action=action,
                            object=o,
                            type="oblique",
                            source=sent.text,
                        )
                    )

    return relations


# ============================================================
# 構成要素抽出
# ============================================================

def extract_components(doc, relations: List[Relation]) -> List[str]:
    components = []

    for token in doc:
        if token.pos_ not in ("NOUN", "PROPN"):
            continue

        phrase = expand_noun_phrase(token)
        phrase = remove_reference_prefix(phrase)

        if not phrase:
            continue

        # 抽象的すぎる語を抑制
        if phrase in {
            "こと", "もの", "場合", "ため", "よう",
            "ところ", "各部", "一方", "以上",
        }:
            continue

        components.append(phrase)

    # SAOに出現した語も構成要素として追加
    for r in relations:
        components.extend([r.subject, r.object])

    components = unique_preserve(components)

    # 長すぎる候補を後段で除外
    components = [
        c for c in components
        if len(c) <= 100
    ]

    return components


# ============================================================
# 重複除去
# ============================================================

def relation_key(r: Relation) -> Tuple[str, str, str]:
    return (
        clean_text(r.subject),
        clean_text(r.action),
        clean_text(r.object),
    )


def deduplicate_relations(relations: List[Relation]) -> List[Relation]:
    out = []
    seen = set()

    for r in relations:
        r.subject = remove_reference_prefix(r.subject)
        r.object = remove_reference_prefix(r.object)
        r.action = normalize_action(r.action)

        key = relation_key(r)
        if key in seen:
            continue

        if not r.subject or not r.action or not r.object:
            continue

        if r.subject == r.object:
            continue

        seen.add(key)
        out.append(r)

    return out


# ============================================================
# SAO向け語順正規化
# ============================================================

def build_normalized_sao_text(relations: List[Relation]) -> str:
    """
    抽出済み関係を
      S A O
    の順に並べた解析用テキストにする。

    原文を置き換えるのではなく、別物として保持する。
    """
    lines = []

    for r in relations:
        lines.append(
            f"{r.subject} {r.action} {r.object}"
        )

    return "\n".join(lines)


# ============================================================
# メイン
# ============================================================

def analyze_claim(text: str):
    """
    特許請求項を解析。

    Returns
    -------
    components : list[str]
        構成要素
    relations : list[dict]
        SAO関係
    """
    text = clean_text(text)

    if not text:
        return [], []

    nlp = get_nlp()
    doc = nlp(text)

    memory = build_component_memory(doc)

    all_relations: List[Relation] = []

    # 1. 文単位の解析
    for sent in doc.sents:
        all_relations.extend(
            extract_dependency_relations(sent, memory)
        )
        all_relations.extend(
            extract_particle_sao(sent, memory)
        )
        all_relations.extend(
            extract_oblique_relations(sent, memory)
        )

    # 2. 連体修飾
    all_relations.extend(
        extract_modifier_relations(text, memory)
    )

    # 3. 有する・備える等
    all_relations.extend(
        extract_has_relations(text, memory)
    )

    # 4. 重複除去
    all_relations = deduplicate_relations(all_relations)

    # 5. 構成要素
    components = extract_components(doc, all_relations)

    # 6. 関係に登場する構成要素を優先的に残す
    relation_components = []
    for r in all_relations:
        relation_components.extend([r.subject, r.object])

    relation_components = unique_preserve(relation_components)

    final_components = unique_preserve(
        relation_components + components
    )

    return final_components, [
        r.as_dict() for r in all_relations
    ]


# ============================================================
# 詳細解析
# ============================================================

def analyze_claim_detailed(text: str) -> Dict:
    """
    デバッグ・研究用。
    原文、GiNZA解析、構成要素、SAO、正規化テキストを返す。
    """
    text = clean_text(text)

    if not text:
        return {
            "original": "",
            "sentences": [],
            "tokens": [],
            "components": [],
            "relations": [],
            "normalized": "",
        }

    nlp = get_nlp()
    doc = nlp(text)

    memory = build_component_memory(doc)
    components, relations = analyze_claim(text)

    tokens = []
    for token in doc:
        tokens.append({
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "tag": token.tag_,
            "dep": token.dep_,
            "head": token.head.text,
            "particle": particle_of(token),
        })

    normalized = build_normalized_sao_text(
        [Relation(**r) for r in relations]
    )

    return {
        "original": text,
        "sentences": [s.text for s in doc.sents],
        "tokens": tokens,
        "components": components,
        "relations": relations,
        "normalized": normalized,
        "reference_memory": memory,
    }


# ============================================================
# 従来コードとの互換用
# ============================================================

def print_analysis(components, relations):
    print("========== 構成要素 ==========")
    for i, c in enumerate(components, 1):
        print(f"{i}. {c}")

    print("\n========== SAO関係 ==========")
    for i, r in enumerate(relations, 1):
        if isinstance(r, dict):
            print(
                f"{i}. "
                f"S={r.get('subject', '')} | "
                f"A={r.get('action', '')} | "
                f"O={r.get('object', '')}"
            )
        else:
            print(
                f"{i}. "
                f"S={r.subject} | "
                f"A={r.action} | "
                f"O={r.object}"
            )


# ============================================================
# テスト
# ============================================================

if __name__ == "__main__":

    test_claims = [
        "制御部が半導体スイッチング素子を駆動する。",
        "半導体装置は、第1の基板と、前記第1の基板上に設けられた第2の半導体層と、前記第2の半導体層に接続された電極とを有する。",
        "第1の基板上に設けられた第2の半導体層に接続された電極を有する半導体装置。",
    ]

    for idx, claim in enumerate(test_claims, 1):
        print("\n" + "=" * 70)
        print(f"TEST {idx}")
        print("=" * 70)
        print("原文:")
        print(claim)

        try:
            components, relations = analyze_claim(claim)
            print_analysis(components, relations)

            print("\n========== SAO正規化 ==========")
            detailed = analyze_claim_detailed(claim)
            print(detailed["normalized"])

        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")
