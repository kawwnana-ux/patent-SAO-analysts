import spacy
import ginza
import ja_ginza
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import os
import glob

# ============================================================
# GiNZAモデルの読み込み
# ============================================================

# Webサーバー環境では、インストール済みのGiNZAモデルを直接読み込む
nlp = spacy.load("ja_ginza")
print("GiNZAの読み込みに成功しました！")

# ============================================================
# 日本語フォント
# ============================================================

# Web環境で利用できる日本語フォントを探索
_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansJP-Regular.ttf",
]
_FONT_PATH = next((p for p in _FONT_CANDIDATES if os.path.exists(p)), None)
if _FONT_PATH:
    fm.fontManager.addfont(_FONT_PATH)
    FONT_PROP = fm.FontProperties(fname=_FONT_PATH)
else:
    FONT_PROP = fm.FontProperties()


RELATION_WORDS = {
    "間", "側", "上", "下", "内部", "外部",
    "周囲", "近傍", "前", "後", "間隔", "位置",
    "上部", "下部", "底部", "前部", "後部", "左側", "右側",
    "上端", "下端", "先端", "基端", "端部",
    "外周面", "内周面", "外面", "内面", "表面", "裏面", "上面", "下面",
    "中央", "中間", "周辺", "周縁",
    "前方", "後方", "上方", "下方", "側面",
}

# 「有する」と同じ意味で使われる動詞（「Ａを備える」「Ａを具備する」等）
HAS_LEMMAS = {"有する", "備える", "具備する"}

# 「ことを特徴とする」のような決まり文句に出てくる、実在の構成要素ではない
# 一般的な語（構成要素としては登録しない）
GENERIC_NOUNS = {"こと", "もの", "とき", "場合", "特徴", "ため"}


def _is_generic_relation_word_bigram(doc, i):
    """
    「外周面」（外周＋面）のように、GiNZAが2トークンに分割してしまう
    複合位置語を判定する。
    """
    if i + 1 >= len(doc):
        return False
    combined = doc[i].text + doc[i + 1].text
    if combined not in RELATION_WORDS:
        return False
    return doc[i + 1].head.pos_ == "VERB"


def _is_generic_relation_word(token):
    """
    「側」「内部」などが、一般的な位置関係の語（Ａの上に／Ａの間に、のように
    動詞に係る用法）として使われているか判定する。
    係り先が動詞であれば一般的な位置関係語（構成要素名からは除外する）。
    係り先が名詞であれば「円盤状カッター側」「破砕槽内部側」のように、
    どちらの面・方向かを表す複合語の一部（構成要素名に含める）とみなす。
    """
    if token.text not in RELATION_WORDS:
        return False
    return token.head.pos_ == "VERB"


def _is_counter_word(token):
    """
    「２枚」「３個」のような助数詞（数を数える単位語）かどうかを判定する。
    NUM（数詞）を直接の子に持つ短い名詞は、助数詞である可能性が高い。
    """
    if len(token.text) > 2:
        return False
    return any(child.pos_ == "NUM" and child.dep_ == "nummod" for child in token.children)


def _normalize_component_text(phrase):
    """
    「前記メタデータ生成部」のように、何らかの理由で「前記」「該」が
    スキップされずに構成要素名の先頭に残ってしまった場合の保険。
    「メタデータ生成部」（前記なし）の表記と食い違って、
    同じものが別ノードとして扱われてしまうのを防ぐため、
    先頭の「前記」「該」を取り除く。
    """
    for prefix in ("前記", "該"):
        if phrase.startswith(prefix) and phrase != prefix:
            phrase = phrase[len(prefix):]
    return phrase


# ============================================================
# ① 構成要素抽出
# ============================================================

def extract_patent_components_general(doc):
    components = []
    i = 0
    while i < len(doc):
        token = doc[i]

        if token.text in ("前記", "該"):
            i += 1
            continue

        if _is_generic_relation_word_bigram(doc, i):
            i += 2
            continue

        if token.text == "第" and i + 1 < len(doc) and doc[i + 1].pos_ == "NUM":
            start = i
            words = [doc[i].text]
            i += 1
            words.append(doc[i].text)
            i += 1
            if i < len(doc) and doc[i].text == "の":
                words.append(doc[i].text)
                i += 1
            while i < len(doc) and doc[i].pos_ in {"NOUN", "PROPN"}:
                if _is_generic_relation_word(doc[i]):
                    break
                words.append(doc[i].text)
                i += 1
            end = i - 1
            phrase = _normalize_component_text("".join(words))
            if phrase not in RELATION_WORDS and phrase not in GENERIC_NOUNS:
                components.append({"text": phrase, "start": start, "end": end})
            continue

        if token.pos_ in {"NOUN", "PROPN"}:
            if _is_counter_word(token):
                i += 1
                continue
            start = i
            words = [token.text]
            i += 1
            while i < len(doc) and doc[i].pos_ in {"NOUN", "PROPN"}:
                if _is_generic_relation_word(doc[i]):
                    break
                words.append(doc[i].text)
                i += 1
            end = i - 1
            phrase = _normalize_component_text("".join(words))
            if phrase not in RELATION_WORDS and phrase not in GENERIC_NOUNS:
                components.append({"text": phrase, "start": start, "end": end})
            continue

        i += 1

    unique_components = []
    seen = set()
    for c in components:
        key = (c["start"], c["end"])
        if key in seen:
            continue
        seen.add(key)
        unique_components.append(c)
    return unique_components


# ============================================================
# ② 関係語（上・間 など）抽出
# ============================================================

def extract_relation_words_general(doc):
    results = []
    for token in doc:
        if token.pos_ != "VERB":
            continue
        for child in token.children:
            if child.pos_ != "NOUN":
                continue

            relation_word_text = None

            if child.text in RELATION_WORDS:
                relation_word_text = child.text
            elif child.i - 1 >= 0:
                prev = doc[child.i - 1]
                combined = prev.text + child.text
                if combined in RELATION_WORDS and prev.head.i == child.i:
                    # 「外周面」（外周＋面）のように2トークンに分割された
                    # 複合位置語。ラベルは結合した形にする。
                    relation_word_text = combined

            if relation_word_text is None:
                continue

            results.append({
                "relation_word": relation_word_text,
                "relation_index": child.i,
                "verb": token.text,
                "verb_index": token.i,
                "dependency": child.dep_,
            })
    return results


# ============================================================
# 共通ヘルパー
# ============================================================

def find_component_by_token(components, token_index):
    for c in components:
        if c["start"] <= token_index <= c["end"]:
            return c
    return None


def find_referenced_component(components, token):
    component = find_component_by_token(components, token.i)
    if component is not None:
        return component
    if token.pos_ not in {"NOUN", "PROPN"}:
        # 動詞などは、たまたま文字列が構成要素名と重なっていても
        # 参照とはみなさない（例：動詞「シール」と名詞「リングシール」）
        return None
    word = token.text
    for c in reversed(components):
        if word in c["text"] and c["end"] < token.i:
            return c
    return None


def find_previous_component_by_word(components, token):
    component = find_component_by_token(components, token.i)
    if component is not None:
        return component
    if token.pos_ not in {"NOUN", "PROPN"}:
        return None
    word = token.text
    for c in reversed(components):
        if c["end"] >= token.i:
            continue
        if word in c["text"]:
            return c
    return None


def find_target_component_from_verb(components, verb):
    targets = []
    current = verb
    visited = set()
    while True:
        if current.i in visited:
            break
        visited.add(current.i)
        component = find_component_by_token(components, current.i)
        if component is not None:
            if component not in targets:
                targets.append(component)
            break
        if current.head == current:
            break
        current = current.head
    return targets


def _find_outermost_component_from_verb(components, verb):
    """
    「Ｘを用いてＹに対応するＺを生成するＷ」のように、
    「用いる」の係り先を辿ると先に「Ｚ」（合成音声データ）に行き当たるが、
    本当の動作主はさらに奥にある「Ｗ」（合成音声データ生成部）である、
    というような何重にも入れ子になった文に対応する。

    find_target_component_from_verb は最初に見つかった構成要素で
    止まってしまうが、こちらは1つ目が見つかった後もさらに1段階だけ
    奥を探し、2つ目が見つかればそちらを採用する
    （そのまま際限なく奥まで辿ると、文書全体の最後の語＝請求項の
    タイトルに行き着いてしまうため、2つ目までで止める）。
    """
    found = []
    current = verb
    visited = set()
    while len(found) < 2:
        if current.i in visited:
            break
        visited.add(current.i)
        component = find_component_by_token(components, current.i)
        if component is not None and (not found or component["text"] != found[-1]["text"]):
            found.append(component)
        if current.head == current:
            break
        current = current.head
    if not found:
        return None
    return found[-1]


# ============================================================
# ③ 位置関係の抽出（「〜上に設けられた」等）
# ============================================================

def _is_locative_obl(token):
    """
    「破砕槽内壁面には固定刃を有し」のように、動詞の obl（斜格）が
    場所を表しているかどうかを判定する。
    「により」「によって」のような手段を表す格、「において」のような
    前提・状況を表す格（"より"/"おい"がfixedでついている場合）は
    場所ではないので除外する。
    """
    if token.dep_ != "obl":
        return False
    for child in token.children:
        if child.dep_ == "case":
            for grandchild in child.children:
                if grandchild.dep_ == "fixed" and grandchild.text in ("より", "よって", "おい"):
                    return False
    return True


def extract_has_location_relations(doc, components):
    """
    「Ａには／Ａに、Ｂを有し」のように、場所（RELATION_WORDSの
    固定リストにない語も含む）と「有する」の目的語との関係を抽出する。
    """
    relations = []
    for verb in doc:
        if verb.lemma_ not in HAS_LEMMAS or verb.pos_ != "VERB":
            continue

        obj_token = None
        for child in verb.children:
            if child.dep_ == "obj":
                obj_token = child
                break
        if obj_token is None:
            continue

        target = find_component_by_token(components, obj_token.i) or find_referenced_component(components, obj_token)
        if target is None:
            continue

        for child in verb.children:
            if not _is_locative_obl(child):
                continue
            source = find_component_by_token(components, child.i) or find_referenced_component(components, child)
            if source is None or source["text"] == target["text"]:
                continue
            relations.append({
                "source": source["text"],
                "relation": "には有する",
                "target": target["text"],
                "type": "positional",
            })
    return relations


def extract_installation_relations(doc, components):
    """
    「回転軸に設けた２枚のサイドプレート」のように、「設ける」の
    係り先が「間」のような位置語（構成要素として登録されていない語）
    になっている場合、その位置語の compound の子（＝実際に設置された
    構成要素）を探して関係先にする。
    """
    relations = []
    for verb in doc:
        if verb.lemma_ != "設ける" or verb.pos_ != "VERB":
            continue

        obl_child = None
        for child in verb.children:
            if child.dep_ == "obl":
                obl_child = child
                break
        if obl_child is None:
            continue

        source = find_component_by_token(components, obl_child.i) or find_referenced_component(components, obl_child)
        if source is None:
            continue

        head = verb.head
        target = find_component_by_token(components, head.i)
        if target is None:
            for child in head.children:
                if child.dep_ == "compound":
                    t = find_component_by_token(components, child.i)
                    if t is not None:
                        target = t
                        break
        if target is None:
            targets = find_target_component_from_verb(components, verb)
            target = targets[0] if targets else None

        if target is None or target["text"] == source["text"]:
            continue

        relations.append({
            "source": source["text"],
            "relation": "に設けた",
            "target": target["text"],
            "type": "positional",
        })
    return relations


def _merged_modifier_name(token, components):
    """
    「外側の面」のように、「の」で係る nmod の修飾語を語自体の前に
    くっつけた、より具体的な名前を作る。
    """
    comp = find_component_by_token(components, token.i)
    base = comp["text"] if comp is not None else token.text

    prefix = ""
    for child in token.children:
        if child.dep_ != "nmod":
            continue
        has_no = any(c.dep_ == "case" and c.text == "の" for c in child.children)
        if not has_no:
            continue
        child_comp = find_component_by_token(components, child.i)
        prefix = (child_comp["text"] if child_comp is not None else child.text) + "の"
        break

    return prefix + base


def _find_owner_via_acl(token, components):
    """
    「該サイドプレートの円盤状カッター側ではない外側の面」のように、
    否定の連体修飾（acl）を挟んで持ち主（例：サイドプレート）が
    係っている場合、それを辿って見つける。
    """
    for child in token.children:
        if child.dep_ != "acl":
            continue
        for grandchild in child.children:
            if grandchild.dep_ == "nmod":
                comp = find_component_by_token(components, grandchild.i) or find_referenced_component(components, grandchild)
                if comp is not None:
                    return comp
    return None


def extract_contact_relations(doc, components):
    """
    「〜に接するようにＸを接触させて」のように、「接する」節の主語が
    明示されていない場合、その節が係っている動詞（接触させて等）の
    目的語を主語として補う。

    また、接する対象（面など）は「外側の面」のように修飾語を含めた
    名前にし、さらにその面の持ち主（例：サイドプレート）が分かる場合は
    「持ち主 → 面 → 接するもの」という鎖にする
    （＝持ち主の直接の子として「接するもの」がぶら下がる形にするため）。
    """
    relations = []
    for verb in doc:
        if verb.lemma_ != "接する" or verb.pos_ != "VERB":
            continue

        obl_child = None
        for child in verb.children:
            if child.dep_ == "obl":
                obl_child = child
                break
        if obl_child is None:
            continue
        target_comp = find_component_by_token(components, obl_child.i) or find_referenced_component(components, obl_child)
        if target_comp is None:
            continue
        target_name = _merged_modifier_name(obl_child, components)

        parent_verb = verb.head
        agent = None
        if parent_verb.pos_ == "VERB":
            for child in parent_verb.children:
                if child.dep_ == "obj":
                    agent = find_component_by_token(components, child.i) or find_referenced_component(components, child)
                    if agent is not None:
                        break
        if agent is None or agent["text"] == target_name:
            continue

        owner = _find_owner_via_acl(obl_child, components)
        if owner is not None and owner["text"] != target_name:
            relations.append({
                "source": owner["text"],
                "relation": "有する",
                "target": target_name,
                "type": "has",
            })

        # 「面 が 接するもの に接する」という向きにして、
        # 持ち主 → 面 → 接するもの、という鎖になるようにする
        relations.append({
            "source": target_name,
            "relation": "に接する",
            "target": agent["text"],
            "type": "direct",
        })
    return relations


def extract_boundary_relations(doc, components):
    """
    「Ａの内部側とその外側のＢとの境界」のように、「境界」が
    複数のものの間にある場合、nmodの係り受けを辿って
    「境界」とその両側（Ａ・Ｂ）との関係を抽出する。

    「と」でつながっている語（＝対等に並んでいる境界の両側）だけを対象にし、
    「の」でつながっている語（＝単なる修飾語。例：外側の／サイドプレートの）
    は関係先にしない。
    """
    def has_to_marker(token):
        return any(c.dep_ == "case" and c.text == "と" for c in token.children)

    relations = []
    for c in components:
        if not c["text"].endswith("境界"):
            continue
        boundary_token = doc[c["end"]]

        stack = [boundary_token]
        seen = set()
        while stack:
            t = stack.pop()
            if t.i in seen:
                continue
            seen.add(t.i)

            for child in t.children:
                if child.dep_ != "nmod":
                    continue
                comp = find_component_by_token(components, child.i) or find_referenced_component(components, child)

                if has_to_marker(child):
                    if comp is not None and comp["text"] != c["text"]:
                        relations.append({
                            "source": c["text"],
                            "relation": "との境界",
                            "target": comp["text"],
                            "type": "positional",
                        })
                    # 「と」で繋がった語の中に、さらに入れ子で「と」の並列項が
                    # ある場合があるので、見つかった後も奥まで探索を続ける
                    stack.append(child)
                elif comp is None:
                    # まだ構成要素が見つかっていない場合だけ、さらに奥まで辿る
                    stack.append(child)
    return relations


def extract_positional_relations(doc, components, relation_words):
    relations = []
    for relation in relation_words:
        relation_token = doc[relation["relation_index"]]

        source_components = []
        for child in relation_token.children:
            c = find_referenced_component(components, child)
            if c is not None and c not in source_components:
                source_components.append(c)

        verb = relation_token.head
        if verb.pos_ != "VERB":
            continue

        if verb.lemma_ in HAS_LEMMAS:
            # 「Ａ間に、Ｂを有し」のように「有する」が使われている場合は、
            # 文全体の主語（根っこ）ではなく、「有する」の直接の目的語
            # （＝実際にそこに存在するもの）を関係先にする。
            target_components = []
            for child in verb.children:
                if child.dep_ == "obj":
                    t = find_component_by_token(components, child.i) or find_referenced_component(components, child)
                    if t is not None:
                        target_components.append(t)
                    break
            label = relation["relation_word"] + "に" + verb.text
        else:
            target_components = []
            for child in verb.children:
                if child.dep_ == "obj":
                    t = find_component_by_token(components, child.i) or find_referenced_component(components, child)
                    if t is not None:
                        target_components.append(t)
                    break
            if not target_components:
                # 動詞自身に直接の目的語(obj)がない場合（受身形など）だけ、
                # 従来通り動詞連鎖を遡って構成要素を探す
                target_components = find_target_component_from_verb(components, verb)
            aux_texts = "".join(
                c.text for c in sorted(verb.children, key=lambda c: c.i)
                if c.pos_ == "AUX" and c.i > verb.i
            )
            label = relation["relation_word"] + "に" + verb.text + aux_texts

        for target in target_components:
            for source in source_components:
                if source["text"] == target["text"]:
                    continue
                relations.append({
                    "source": source["text"],
                    "relation": label,
                    "target": target["text"],
                    "type": "positional",
                })
    return relations


# ============================================================
# ④ 直接関係の抽出（「Aに接続されたB」等）
# ============================================================

def _is_passive(verb):
    """動詞が受身形（〜られた／〜れた）かどうかを判定する"""
    return any(
        child.pos_ == "AUX" and child.lemma_ in ("れる", "られる")
        for child in verb.children
    )


def _is_instrumental_obl(token):
    """
    「回転カッター式破砕機により破砕する」のように、動詞の obl（斜格）が
    「により」「によって」で手段・道具を表しているかどうかを判定する。
    """
    if token.dep_ != "obl":
        return False
    for child in token.children:
        if child.dep_ == "case":
            for grandchild in child.children:
                if grandchild.dep_ == "fixed" and grandchild.text in ("より", "よって"):
                    return True
    return False


def _find_topic_in_verb_chain(doc, components, verb):
    """
    「前記テキスト翻訳部は、〜を翻訳して〜を生成し」のように、
    動詞が連鎖している場合、その連鎖（advcl/auxで繋がったVERB同士）の
    範囲内だけで「は」で明示された主題を探す。

    長い複文では、GiNZAが複数の「は」付き名詞を同じ動詞のnsubjとして
    （誤って）結びつけてしまうことがあるため、
      ① 対象の動詞より後ろに出てくる「は」は候補にしない
         （主語が動詞より後に来ることはないため）
      ② 複数見つかった場合は、動詞に一番近い（＝一番あとに出てくる）
         ものを採用する
    という2段階で絞り込む。

    「破砕槽内壁面には」のように「に」＋「は」が連続する場合は対象外にする。
    """
    original_verb_i = verb.i
    current = verb
    visited = set()
    candidates = []
    while current.i not in visited:
        visited.add(current.i)
        for child in current.children:
            if child.pos_ not in ("NOUN", "PROPN"):
                continue
            if child.i >= original_verb_i:
                continue
            has_bare_wa = False
            for cc in child.children:
                if cc.dep_ == "case" and cc.text == "は":
                    if cc.i > 0 and doc[cc.i - 1].pos_ == "ADP":
                        continue
                    has_bare_wa = True
                    break
            if has_bare_wa:
                comp = find_component_by_token(components, child.i) or find_referenced_component(components, child)
                if comp is not None:
                    candidates.append((child.i, comp))
        nxt = current.head
        if nxt.i == current.i or nxt.pos_ != "VERB":
            break
        current = nxt

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _find_nsubj_up_chain(verb):
    """
    「テキスト翻訳部は…翻訳して…生成し」のように、動詞が連鎖している場合、
    その動詞自身に主語(nsubj)がなくても、連鎖を遡った先の動詞
    （最終的にROOTに近い動詞）に本当の主語が付いていることが多い。
    それを探して返す。

    ただし、主語が動詞より後ろに来ることはないので、
    動詞より後ろにある nsubj は候補にしない
    （GiNZAが超長文で、無関係な後方の「は」付き名詞を
      同じ動詞のnsubjとして誤って結びつけてしまうことがあるため）。
    """
    original_verb_i = verb.i
    current = verb
    visited = set()
    while current.i not in visited:
        visited.add(current.i)
        for child in current.children:
            if child.dep_ == "nsubj" and child.i < original_verb_i:
                return child
        nxt = current.head
        if nxt.i == current.i or nxt.pos_ != "VERB":
            break
        current = nxt
    return None


def _find_nearest_topic_before_text(doc, components, verb):
    """
    最終手段：動詞連鎖・nsubj連鎖のどちらでも見つからない場合に、
    依存構造を無視して、テキスト上で動詞より手前にある一番近い
    裸の「は」を探す。

    GiNZAが超長文で「Ａは」を、途中の動詞を全部飛び越えて
    文末の語に直接結びつけてしまうことがあり（例：「メタデータ生成部は」が
    「通信端末」に直接nsubjとして付く）、動詞の連鎖を辿る方法では
    原理的に見つけられないため。

    句点（。）をまたいで探さない。「には」のような複合格助詞の
    「は」は対象外にする。また、候補と対象の動詞との間に
    別の「有する」系動詞（＝その節がすでに完結している印）が
    挟まっている場合は、その候補は別の節の主題とみなしてスキップする。
    """
    for i in range(verb.i - 1, -1, -1):
        t = doc[i]
        if t.text == "。":
            break
        if t.text == "は" and t.dep_ == "case" and t.head.pos_ in ("NOUN", "PROPN"):
            if i > 0 and doc[i - 1].pos_ == "ADP":
                continue
            has_boundary = any(
                doc[j].pos_ == "VERB" and doc[j].lemma_ in HAS_LEMMAS
                for j in range(i + 1, verb.i)
            )
            if has_boundary:
                continue
            comp = find_component_by_token(components, t.head.i) or find_referenced_component(components, t.head)
            if comp is not None:
                return comp
    return None


def _find_following_capability_owner(doc, components, obj_token, limit_i):
    """
    「Ａを…受信可能な受信部」のように、目的語（Ａ）のすぐ後ろに
    「〜可能な◯◯部」という形が続いている場合、それを本当の持ち主として返す。

    GiNZAの解析では、こうした目的語が「受信部」を飛び越して
    外側の動詞（例：含む）に直接繋がってしまうことがあるため、
    テキストの並び順で「次に出てくる〜可能な部」を優先的に探す。
    limit_i より手前（次のリスト項目の区切りが来る前）までしか探さない。
    """
    for i in range(obj_token.i + 1, limit_i):
        t = doc[i]
        if t.text == "可能" and t.pos_ == "ADJ":
            head_noun = t.head
            if head_noun.pos_ in ("NOUN", "PROPN"):
                comp = find_component_by_token(components, head_noun.i) or find_referenced_component(components, head_noun)
                if comp is not None:
                    return comp
    return None


def extract_attribute_relations(doc, components):
    """
    「前記粘着剤層の…表面の算術平均粗さＳａが０．０１２μｍ以下である」
    のように、構成要素の数値スペック（属性）を表す文から
    「持ち主 →（属性名）→ 数値」という関係を抽出する。
    """
    COMPARISON_WORDS = {"以下", "以上", "未満", "超", "程度", "以内"}
    relations = []
    for token in doc:
        if token.text not in COMPARISON_WORDS or token.pos_ != "NOUN":
            continue

        nsubj_token = None
        number_token = None
        unit_token = None
        for child in token.children:
            if child.dep_ == "nsubj":
                nsubj_token = child
            elif child.dep_ == "advmod" and any(ch.isdigit() or ch in "．.０１２３４５６７８９" for ch in child.text):
                number_token = child
            elif child.dep_ == "compound":
                unit_token = child
        if nsubj_token is None or number_token is None:
            continue

        # 属性名（例：算術平均粗さＳａ）を、nsubj自身に直接くっついている
        # 修飾語（compound/amod等）を集めて組み立てる
        attr_words = []
        for c in sorted(nsubj_token.children, key=lambda c: c.i):
            if c.dep_ in ("compound", "amod") or c.pos_ == "PART":
                attr_words.append(c.text)
        attr_words.append(nsubj_token.text)
        attribute_name = "".join(attr_words)

        # 持ち主を、nsubjから「の」（nmod）や「から」（acl→obl）で繋がる
        # 連鎖を辿って探す。「算術平均粗さＳａ」→「表面」→「側」→（遠い）→
        # 「基材」→「粘着剤層」のように、途中に比較のための参照点
        # （基材など）を挟んでいることがあるため、見つかった後も
        # さらに奥（「の」で係る本当の持ち主）がないか探し続け、
        # 最後に見つかったものを採用する。
        owner = None
        current = nsubj_token
        visited = set()
        while current.i not in visited:
            visited.add(current.i)
            next_token = None
            for child in current.children:
                if child.dep_ == "nmod":
                    next_token = child
                    break
                if child.dep_ == "acl" and child.pos_ == "ADJ":
                    for gc in child.children:
                        if gc.dep_ == "obl":
                            next_token = gc
                            break
                    if next_token is not None:
                        break
            if next_token is None:
                break
            comp = find_component_by_token(components, next_token.i) or find_referenced_component(components, next_token)
            if comp is not None:
                owner = comp
            current = next_token
        if owner is None:
            continue

        value_text = number_token.text + (unit_token.text if unit_token is not None else "") + token.text

        relations.append({
            "source": owner["text"],
            "relation": attribute_name,
            "target": value_text,
            "type": "attribute",
        })
    return relations


def extract_composition_relations(doc, components):
    """
    「金属からなる導電部」「群から選択される金属」のように、
    「〜から」＋「なる／選択される／選ばれる」で材料・由来を表す
    パターンから関係を抽出する（マーカッシュ形式でよく使われる）。
    """
    COMPOSITION_LEMMAS = {"なる", "選択", "選ぶ"}
    relations = []
    for verb in doc:
        if verb.lemma_ not in COMPOSITION_LEMMAS or verb.pos_ != "VERB":
            continue

        from_child = None
        for child in verb.children:
            if child.dep_ != "obl":
                continue
            has_kara = any(
                c.dep_ == "case" and c.text == "から" for c in child.children
            )
            if has_kara:
                from_child = child
                break
        if from_child is None:
            continue

        source_comps = []
        stack = [from_child]
        seen = set()
        while stack:
            t = stack.pop()
            if t.i in seen:
                continue
            seen.add(t.i)
            comp = find_component_by_token(components, t.i) or find_referenced_component(components, t)
            if comp is not None and comp not in source_comps:
                source_comps.append(comp)
            for child in t.children:
                if child.dep_ == "nmod":
                    stack.append(child)
        if not source_comps:
            continue

        head_noun = verb.head
        target_comp = None
        if head_noun.i != verb.i:
            target_comp = find_component_by_token(components, head_noun.i)
            if target_comp is None:
                fallback = find_target_component_from_verb(components, verb)
                target_comp = fallback[0] if fallback else None
        if target_comp is None:
            # GiNZAがこの動詞を誤って文全体の根っこ（head=自分自身）だと
            # 解析してしまっている場合の保険。この動詞は連体修飾
            # （〜される◯◯）として使われていることが多いので、
            # すぐ後ろに出てくる構成要素を係り先とみなす。
            for i in range(verb.i + 1, min(verb.i + 8, len(doc))):
                comp = find_component_by_token(components, i)
                if comp is not None:
                    target_comp = comp
                    break
        if target_comp is None:
            continue

        label = "からなる" if verb.lemma_ == "なる" else "から選択される"
        for source_comp in source_comps:
            if target_comp["text"] == source_comp["text"]:
                continue
            relations.append({
                "source": target_comp["text"],
                "relation": label,
                "target": source_comp["text"],
                "type": "direct",
            })
    return relations


def extract_capability_relations(doc, components):
    """
    「音を出力可能な音出力部」のように、動詞ではなく「〜可能な」という
    形容詞の形で能力を表すパターンから関係を抽出する。

    「Ｘ可能」（ＡＤＪ）が名詞Ｙ（例：音出力部）を修飾している場合、
    その目的語（例：音、ＸのＮＯＵＮ compound「出力」が動詞的働きをする）は
    Ｙ自身の直接の子（obj）としてGiNZAに解析されることが多いため、
    Ｙの子から探す。
    """
    relations = []
    for adj in doc:
        if adj.text != "可能" or adj.pos_ != "ADJ":
            continue

        verb_stem = None
        for child in adj.children:
            if child.dep_ == "compound":
                verb_stem = child.text
                break
        if verb_stem is None:
            continue

        head_noun = adj.head
        if head_noun.pos_ not in ("NOUN", "PROPN"):
            continue

        source = find_component_by_token(components, head_noun.i) or find_referenced_component(components, head_noun)
        if source is None:
            continue

        obj_token = None
        for child in head_noun.children:
            if child.dep_ == "obj":
                obj_token = child
                break
        if obj_token is None:
            for child in adj.children:
                if child.dep_ == "obj":
                    obj_token = child
                    break
        if obj_token is None:
            continue

        target = find_component_by_token(components, obj_token.i) or find_referenced_component(components, obj_token)
        if target is None or target["text"] == source["text"]:
            continue

        relations.append({
            "source": source["text"],
            "relation": verb_stem,
            "target": target["text"],
            "type": "direct",
        })
    return relations


def extract_direct_relations(doc, components):
    """
    「Ａに接続されたＢ」（受身）と「Ｂを破砕するＡ」（能動）の
    両方に対応する。受身なら修飾先の名詞(head)が動作の受け手＝target、
    能動なら修飾先の名詞(head)が動作の主体＝sourceになる。

    能動の場合、優先順位は次の通り：
      1) 動詞連鎖を遡って見つかる本当の主語（nsubj）
         例：「テキスト翻訳部は…翻訳して…生成し」の「テキスト翻訳部」
      2) 「により／によって」で明示された手段・道具
         例：「回転カッター式破砕機により破砕する」の「回転カッター式破砕機」
      3) どちらもなければ、修飾先の名詞(head)
    """
    relations = []
    for verb in doc:
        if verb.pos_ != "VERB":
            continue
        if verb.lemma_ in HAS_LEMMAS:
            # 「有する」「備える」「具備する」は extract_has_relations /
            # extract_has_location_relations / extract_positional_relations の
            # 方で別途処理しているのでここでは扱わない
            continue
        if verb.lemma_ == "接触" and any(
            child.dep_ == "advcl" and child.lemma_ == "接する" for child in verb.children
        ):
            # 「〜に接するように…接触させて」は extract_contact_relations の方で
            # 別途処理しているのでここでは扱わない（重複防止）
            continue

        head_component = find_component_by_token(components, verb.head.i)
        used_fallback = head_component is None
        if head_component is None:
            # 係り先が構成要素でない場合（別の動詞に連なっている等）は、
            # さらに上まで遡って構成要素を探す
            fallback = find_target_component_from_verb(components, verb)
            head_component = fallback[0] if fallback else None
        if head_component is None:
            continue

        if _is_passive(verb):
            # 受身：headが受け手（target）。「に」で係る語などが動作主（source）。
            for child in verb.children:
                if child.dep_ not in ("obl", "nsubj"):
                    continue
                if child.text == "場合":
                    # 「場合」は条件節の目印であって、動作主ではないので除外する
                    # （GiNZAが超長文でここに主語を誤って結びつけることがある）
                    continue
                source = (
                    find_previous_component_by_word(components, child)
                    or find_referenced_component(components, child)
                )
                if source is None or source["text"] == head_component["text"]:
                    continue
                relations.append({
                    "source": source["text"],
                    "relation": verb.text,
                    "target": head_component["text"],
                    "type": "direct",
                })
        else:
            # 能動：headが直接の係り先として構成要素そのものであれば、それを
            # 主語(source)として使う（例：「Ｘを表すＹ」のＹ＝head）。
            # headがfallback（動詞連鎖を遡って）でしか見つからなかった場合、
            # または head が「文書の最後の語＝請求項タイトル」で、かつ
            # 動詞自身に場所(obl)がある場合（順次列挙形式で複数の動詞が
            # みな最後の装置名にacl接続されてしまうケース）だけ、
            # ①「は」で明示された主題 → ②動詞連鎖を遡った主語(nsubj)
            # → ③動詞自身（またはその1つ先の連なった動詞）の「Ｘに」
            #   （場所を表すobl） → ④テキスト上の直前の「は」
            #   → ⑤「により」の道具、の優先順位で本当の主語を探し直す。
            effective_source = head_component

            last_real_token_i = len(doc) - 1
            while last_real_token_i > 0 and doc[last_real_token_i].pos_ == "PUNCT":
                last_real_token_i -= 1
            head_is_claim_title = head_component["end"] == last_real_token_i

            if used_fallback or head_is_claim_title:
                topic = _find_topic_in_verb_chain(doc, components, verb)
                if topic is not None:
                    effective_source = topic
                else:
                    nsubj_token = _find_nsubj_up_chain(verb)
                    if nsubj_token is not None:
                        subj = (
                            find_component_by_token(components, nsubj_token.i)
                            or find_referenced_component(components, nsubj_token)
                        )
                        if subj is not None:
                            effective_source = subj
                    else:
                        text_topic = _find_nearest_topic_before_text(doc, components, verb)
                        if text_topic is not None:
                            effective_source = text_topic
                        else:
                            own_locative = None
                            if verb.lemma_ != "用いる":
                                # 「用いる」は下の専用フォールバックの方で
                                # 別途処理しているのでここでは対象外にする（重複防止）。
                                # 動詞自身の子、または（それで見つからなければ）
                                # 1つだけ先の連なった動詞の子から場所(obl)を探す
                                # （例：「取り付け」の場所「部」は、共通の親「形成」の
                                #  子になっていて、取り付け自身の直接の子ではないため）。
                                # 遡りすぎるとGiNZAの誤解析を拾ってしまうので1段階まで。
                                # また、既にテキスト上の「は」探しで見つからなかった
                                # 場合の最終手段の1つとして使う（優先度を低くする）。
                                for candidate_v in (verb, verb.head if verb.head.pos_ == "VERB" else None):
                                    if candidate_v is None or own_locative is not None:
                                        continue
                                    for child in candidate_v.children:
                                        if _is_locative_obl(child):
                                            comp = (
                                                find_component_by_token(components, child.i)
                                                or find_referenced_component(components, child)
                                            )
                                            if comp is None:
                                                # 「先端」のように、oblの語自体が位置語として
                                                # 構成要素から除外されている場合、その「の」
                                                # 修飾語（例：「アーム」の先端の「アーム」）を見る
                                                for gc in child.children:
                                                    if gc.dep_ == "nmod":
                                                        comp = (
                                                            find_component_by_token(components, gc.i)
                                                            or find_referenced_component(components, gc)
                                                        )
                                                        if comp is not None:
                                                            break
                                            if comp is not None:
                                                own_locative = comp
                                                break
                            if own_locative is not None:
                                effective_source = own_locative
                            elif verb.lemma_ == "用いる":
                                # 「Ｘを用いて〜する部」のように何重にも入れ子に
                                # なっている場合、鎖の一番奥まで辿り直す
                                outer = _find_outermost_component_from_verb(components, verb)
                                if outer is not None:
                                    effective_source = outer

            if effective_source is head_component:
                for child in verb.children:
                    if _is_instrumental_obl(child):
                        tool = (
                            find_component_by_token(components, child.i)
                            or find_referenced_component(components, child)
                        )
                        if tool is not None:
                            effective_source = tool
                        break

            for child in verb.children:
                if child.dep_ != "obj":
                    continue
                target = (
                    find_component_by_token(components, child.i)
                    or find_referenced_component(components, child)
                )
                if target is None or target["text"] == effective_source["text"]:
                    continue

                real_owner = effective_source
                real_relation = verb.text
                if verb.lemma_ == "含む":
                    # 「Ａを…受信可能な受信部」のように、目的語の直後に
                    # 「〜可能な部」が続く場合は、そちらを本当の持ち主にする
                    cap_owner = _find_following_capability_owner(doc, components, child, verb.i)
                    if cap_owner is not None and cap_owner["text"] != target["text"]:
                        real_owner = cap_owner
                if real_owner["text"] == target["text"]:
                    continue
                relations.append({
                    "source": real_owner["text"],
                    "relation": real_relation,
                    "target": target["text"],
                    "type": "direct",
                })

            # 「Ａを送信可能な送信部と、Ｂを受信可能な受信部とを含む通信部」
            # のように、「含む」の対象が複数のリスト項目になっている場合、
            # 単一のobjだけでなく、直前の「であって」区切りから動詞までの
            # 範囲にあるリスト項目もすべて対象にする。
            # 「であって」の区切りが実際に見つかった場合だけ適用する
            # （見つからない場合に文書の先頭まで遡って無関係な項目まで
            #  拾ってしまうのを防ぐため）。
            if verb.lemma_ == "含む":
                scope_start = None
                for j in range(verb.i - 1, -1, -1):
                    if doc[j].text == "。":
                        break
                    if doc[j].text == "て" and doc[j].dep_ == "mark":
                        head_noun = doc[j].head
                        is_de_atte = False
                        for hc in head_noun.children:
                            if hc.dep_ == "cop" and hc.text == "で":
                                if any(gc.dep_ == "fixed" and gc.text in ("あっ", "あり") for gc in hc.children):
                                    is_de_atte = True
                                break
                        if is_de_atte:
                            # 「であって」節の主語が今の所有者と同じものを
                            # 指している場合だけ、この区切りを採用する
                            # （例：「通信部であって」の通信部と、
                            #  「含む」の所有者である通信部が一致する場合）。
                            head_comp = (
                                find_component_by_token(components, head_noun.i)
                                or find_referenced_component(components, head_noun)
                            )
                            if head_comp is not None and head_comp["text"] == effective_source["text"]:
                                scope_start = j + 1
                            break
                if scope_start is not None:
                    for comp in components:
                        if not (scope_start <= comp["end"] < verb.i):
                            continue
                        if not _is_list_item_component(doc, comp):
                            continue
                        if comp["text"] == effective_source["text"]:
                            continue
                        relations.append({
                            "source": effective_source["text"],
                            "relation": verb.text,
                            "target": comp["text"],
                            "type": "direct",
                        })

    unique = []
    seen = set()
    for r in relations:
        key = (r["source"], r["relation"], r["target"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique


# ============================================================
# ⑤ 「有する」関係の抽出
# ============================================================

def _is_list_item_component(doc, comp):
    """
    「Ａと、Ｂと、Ｃと、…を有する（備える）」のような並列列挙で、
    その構成要素の直後に「と、」（区切りの格助詞＋読点）が来ているかどうかを
    判定する。「Ａと通信する」のような、単に「と」で係る場合（直後が
    読点でない）は対象外にする。
    """
    end = comp["end"]
    nxt = end + 1
    if (
        nxt + 1 < len(doc)
        and doc[nxt].text == "と"
        and doc[nxt].dep_ == "case"
        and doc[nxt + 1].text in ("、", "を")
    ):
        return True
    return False


def extract_has_relations(doc, components):
    """
    「Ａを有する」「Ａは〜を有し」のような文から (所有者, 有する, 対象) を抽出する。

    3段階で所有者(owner)を判定する:
      1) 明示的な主語（〜は/〜が）がある節 → その主語
      2) 「〜を有する＜名詞＞」のように＜名詞＞を修飾する形（属格用法） →
         その＜名詞＞。対象は、それより前に出てきた構成要素すべて
         （「Ａと、Ｂと、Ｃと、を有するＸ」という並列列挙のパターン用）
      3) どちらでもない（連用形で他の動詞に連なっている場合など） →
         文全体の主語（＝依存構造上のROOTが属する構成要素）
    """
    root_token = None
    for t in doc:
        if t.head == t:
            root_token = t
            break
    root_component = (
        find_component_by_token(components, root_token.i) if root_token is not None else None
    )

    relations = []
    for verb in doc:
        if verb.lemma_ not in HAS_LEMMAS or verb.pos_ != "VERB":
            continue

        subj_token = None
        obj_token = None
        for child in verb.children:
            if child.dep_ == "nsubj" and subj_token is None:
                subj_token = child
            if child.dep_ == "obj" and obj_token is None:
                obj_token = child

        head_component = find_component_by_token(components, verb.head.i)

        targets = []
        owner = None

        if subj_token is not None:
            owner = (
                find_component_by_token(components, subj_token.i)
                or find_referenced_component(components, subj_token)
            )
            if obj_token is not None:
                t = (
                    find_component_by_token(components, obj_token.i)
                    or find_referenced_component(components, obj_token)
                )
                if t is not None:
                    targets.append(t)
        elif head_component is not None:
            owner = head_component
            # 「Ａと、Ｂと、Ｃと、…を有する」のような並列列挙のパターン用。
            # 直後に「と」が付くリスト項目だけを対象にする
            # （そうしないと、文中の無関係な名詞まで全部拾ってしまうため）。
            list_targets = [
                c for c in components
                if c["end"] < verb.i and _is_list_item_component(doc, c)
            ]
            targets = list_targets if list_targets else [c for c in components if c["end"] < verb.i]
        else:
            owner = root_component
            if obj_token is not None:
                t = (
                    find_component_by_token(components, obj_token.i)
                    or find_referenced_component(components, obj_token)
                )
                if t is not None:
                    targets.append(t)

        if owner is None:
            continue

        for target in targets:
            if target is None or target["text"] == owner["text"]:
                continue
            relations.append({
                "source": owner["text"],
                "relation": "有する",
                "target": target["text"],
                "type": "has",
            })

    unique = []
    seen = set()
    for r in relations:
        key = (r["source"], r["relation"], r["target"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique


# ============================================================
# ⑥ 全関係を統合
# ============================================================

def combine_all_relations(positional, direct, has):
    all_relations = list(positional) + list(direct) + list(has)
    unique = []
    seen = set()
    for r in all_relations:
        key = (r["source"], r["relation"], r["target"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique


def _simplify_hierarchy(relations, doc=None, components=None):
    """
    根（root）から全ノードへ直接「有する」で繋ぐのではなく、
    より具体的な位置関係・直接関係の鎖（例：破砕槽→内壁面→固定刃）が
    既にある場合は、そちらを優先して根からの重複した「有する」を消す。
    また、位置関係の起点になっているが誰からも指されていないノード
    （例：サイドプレート）は、根の直接の子として補って繋ぎ直す。

    「有する」「備える」が1つも使われていない請求項（順次列挙形式など）
    では has_edges が空になるが、その場合は文末の語（＝発明の名称、
    例：照明装置）を根とみなし、他のどこからも指されていないノードを
    その直接の子として補う。
    """
    has_edges = [r for r in relations if r["type"] == "has"]

    if not has_edges:
        if doc is None or components is None:
            return relations
        last_i = len(doc) - 1
        while last_i > 0 and doc[last_i].pos_ == "PUNCT":
            last_i -= 1
        claim_title = find_component_by_token(components, last_i)
        if claim_title is None:
            return relations

        all_targets = set(r["target"] for r in relations)
        extra = []
        added = set()
        for r in relations:
            if r["type"] not in ("positional", "direct"):
                continue
            s = r["source"]
            if s == claim_title["text"] or s in all_targets or s in added:
                continue
            extra.append({
                "source": claim_title["text"],
                "relation": "有する",
                "target": s,
                "type": "has",
            })
            added.add(s)
        return relations + extra

    owners = set(r["source"] for r in has_edges)
    all_targets = set(r["target"] for r in relations)
    roots = [o for o in owners if o not in all_targets]
    root = roots[0] if roots else next(iter(owners))

    incoming = {}
    for r in relations:
        incoming.setdefault(r["target"], []).append(r)

    # ① 根からの「有する」より具体的な鎖がある場合は、根からの分を消す。
    #    ただし「には有する」「間に有し」「含む」のような“そこに含まれる”系
    #    の関係だけを対象にする（「上に設けられた」「接続」のような
    #    単なる並び関係は対象にしない＝半導体装置の例を壊さないため）。
    to_remove = []
    for r in has_edges:
        if r["source"] != root:
            continue
        n = r["target"]
        more_specific = [
            x for x in incoming.get(n, [])
            if x is not r
            and x["type"] in ("positional", "direct")
            and x["source"] != root
            and ("有" in x["relation"] or "含" in x["relation"] or "からなる" in x["relation"] or "選択される" in x["relation"])
        ]
        if more_specific:
            to_remove.append(r)

    simplified = [r for r in relations if r not in to_remove]

    # ② 位置関係・直接関係の起点になっているのに、誰からも指されていない
    #    ノードは、根の直接の子として補って繋ぐ
    incoming2 = {}
    for r in simplified:
        incoming2.setdefault(r["target"], []).append(r)

    extra = []
    added = set()
    for r in simplified:
        if r["type"] not in ("positional", "direct"):
            continue
        s = r["source"]
        if s == root or s in added:
            continue
        if s not in incoming2:
            extra.append({"source": root, "relation": "有する", "target": s, "type": "has"})
            added.add(s)

    return simplified + extra


# ============================================================
# ⑦ パイプライン本体：請求項テキスト → 構成要素・関係（単文形式）
# ============================================================

import re as _re_module


def _clean_claim_text(text):
    """
    請求項テキストの前処理。改行・タブ・連続した空白などを取り除く。

    日本語の請求項は単語間にスペースを必要としないため、
    改行やインデント（字下げ）をそのまま含んだテキストを渡されると、
    GiNZAが「\\n  前記」のように改行・空白を「前記」と同じ1つの
    トークンとしてくっつけてしまうことがある。すると「前記」だけを
    狙った完全一致チェックがすり抜けてしまい、「前記」が構成要素名の
    先頭に残ってしまう（例：「\\n  前記メタデータ生成部」）。
    これを防ぐため、解析前にすべての空白文字を取り除く。
    """
    return _re_module.sub(r"\s+", "", text)


def analyze_claim(text):
    """単文形式の請求項テキストを渡すと (構成要素リスト, 関係リスト) を返す"""
    text = _clean_claim_text(text)
    doc = nlp(text)
    components = extract_patent_components_general(doc)
    relation_words = extract_relation_words_general(doc)

    positional = extract_positional_relations(doc, components, relation_words)
    location = extract_has_location_relations(doc, components)
    installation = extract_installation_relations(doc, components)
    contact = extract_contact_relations(doc, components)
    boundary = extract_boundary_relations(doc, components)
    capability = extract_capability_relations(doc, components)
    composition = extract_composition_relations(doc, components)
    attribute = extract_attribute_relations(doc, components)
    direct = extract_direct_relations(doc, components)
    has = extract_has_relations(doc, components)

    final_relations = combine_all_relations(
        positional + location + installation + boundary,
        direct + contact + capability + composition + attribute,
        has,
    )
    final_relations = _simplify_hierarchy(final_relations, doc, components)
    return components, final_relations


# ============================================================
# ⑧ 自動レイアウト（マインドマップ風：左→右の階層配置）
# ============================================================

from matplotlib.path import Path


def _box_size(text):
    """ノードのラベル文字列から、四角い箱の幅・高さを見積もる"""
    lines = text.split("\n")
    w = max(len(l) for l in lines) * 0.32 + 0.6
    h = 0.5 * len(lines) + 0.5
    return w, h


def _wrap_label(text, max_chars=6):
    """長いノード名は2行に折り返す"""
    if len(text) <= max_chars:
        return text
    mid = len(text) // 2
    return text[:mid] + "\n" + text[mid:]


def compute_layout(G):
    """
    「有する」関係を軸にした、左→右のマインドマップ風レイアウト。
    根（root）を一番左に置き、階層が深くなるほど右に配置する。
    グラフが複数の孤立したグループ（連結成分）に分かれている場合は、
    それぞれを別グループとして縦に並べて配置する。
    """
    undirected = G.to_undirected()
    pos = {}
    x_gap = 2.0
    y_gap = 0.7
    y_cursor = 0.0

    for component_nodes in nx.connected_components(undirected):
        subG = G.subgraph(component_nodes)

        has_edges = [(u, v) for u, v, d in subG.edges(data=True) if d.get("type") == "has"]
        if has_edges:
            owners = set(u for u, v in has_edges)
            all_targets = set(v for u, v, d in subG.edges(data=True))
            roots = [n for n in owners if n not in all_targets]
            root = roots[0] if roots else next(iter(owners))
        else:
            in_deg = dict(subG.in_degree())
            no_incoming = [n for n in component_nodes if in_deg.get(n, 0) == 0]
            root = no_incoming[0] if no_incoming else next(iter(component_nodes))

        lengths = nx.single_source_shortest_path_length(subG.to_undirected(), root)
        layers = {}
        for node, depth in lengths.items():
            layers.setdefault(depth, []).append(node)

        # x位置：各深さ（列）ごとに、その列で一番幅の広い箱に合わせて
        # 次の列の開始位置をずらしていく
        depth_x = {}
        cx = 0.0
        max_depth = max(layers.keys())
        for depth in range(max_depth + 1):
            nodes = layers.get(depth, [])
            if not nodes:
                continue
            max_w = max(_box_size(_wrap_label(n))[0] for n in nodes)
            depth_x[depth] = cx
            cx += max_w + x_gap

        comp_pos = {}
        comp_top = 0.0
        for depth, nodes in layers.items():
            heights = [_box_size(_wrap_label(n))[1] for n in nodes]
            total_h = sum(heights) + y_gap * (len(nodes) - 1)
            y = total_h / 2
            for node, h in zip(nodes, heights):
                comp_pos[node] = (depth_x[depth], y - h / 2)
                y -= h + y_gap
            comp_top = max(comp_top, total_h / 2)

        for node, (x, y) in comp_pos.items():
            pos[node] = (x, y + y_cursor)

        y_cursor -= (comp_top * 2 + 3.0)

    for node in G.nodes():
        if node not in pos:
            pos[node] = (0, y_cursor)
            y_cursor -= 3.0

    return pos


# ============================================================
# ⑨ 可視化（マインドマップ風：四角ノード＋曲線）
# ============================================================

TYPE_STYLE = {
    "has":        {"color": "#4C87C6", "label": "階層関係（有する）"},
    "positional": {"color": "#1f77b4", "label": "位置関係"},
    "direct":     {"color": "#2ca02c", "label": "直接関係（接続など）"},
    "attribute":  {"color": "#d18a1a", "label": "属性（数値スペック）"},
}


def _bezier_path(x0, y0, x1, y1):
    dx = (x1 - x0) * 0.5
    verts = [(x0, y0), (x0 + dx, y0), (x1 - dx, y1), (x1, y1)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    return Path(verts, codes)


def visualize_relations(final_relations, title="特許請求項の構成要素間関係"):
    """analyze_claim()等が返した関係リストを渡すと、マインドマップ風の図を描画する"""
    G = nx.DiGraph()
    for r in final_relations:
        G.add_node(r["source"])
        G.add_node(r["target"])
        G.add_edge(r["source"], r["target"], relation=r["relation"], type=r["type"])

    if len(G.nodes()) == 0:
        print("関係が抽出できませんでした。構成要素や依存構造を確認してください。")
        return

    labels = {n: _wrap_label(n) for n in G.nodes()}
    pos = compute_layout(G)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    fig_w = max(12, (max(xs) - min(xs)) * 1.3)
    fig_h = max(6, (max(ys) - min(ys)) * 1.3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    used_types = set(nx.get_edge_attributes(G, "type").values())

    # ------------------------------------------------------
    # 辺（ベジェ曲線）を先に描く
    # ------------------------------------------------------
    for u, v, d in G.edges(data=True):
        style = TYPE_STYLE.get(d["type"], {"color": "gray"})
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        w0, h0 = _box_size(labels[u])
        w1, h1 = _box_size(labels[v])

        if abs(x1 - x0) < 0.01:
            # 同じ列（兄弟ノード）同士の接続：右側に迂回する縦方向の曲線にする
            sx, sy = x0 + w0 / 2, y0
            tx, ty = x1 + w1 / 2, y1
            bulge = 0.7 + abs(y1 - y0) * 0.12
            verts = [(sx, sy), (sx + bulge, sy), (tx + bulge, ty), (tx, ty)]
            path = Path(verts, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
        elif x1 >= x0:
            sx, sy = x0 + w0 / 2, y0
            tx, ty = x1 - w1 / 2, y1
            path = _bezier_path(sx, sy, tx, ty)
        else:
            sx, sy = x0 - w0 / 2, y0
            tx, ty = x1 + w1 / 2, y1
            path = _bezier_path(sx, sy, tx, ty)

        patch = mpatches.PathPatch(path, facecolor="none", edgecolor=style["color"], lw=1.8, zorder=1)
        ax.add_patch(patch)

        arrow_dx = 0.15 if tx > sx else -0.15
        ax.annotate("", xy=(tx, ty), xytext=(tx - arrow_dx, ty),
                    arrowprops=dict(arrowstyle="-|>", color=style["color"], lw=1.8))

        mx, my = (sx + tx) / 2, (sy + ty) / 2
        ddx, ddy = tx - sx, ty - sy
        dd = (ddx ** 2 + ddy ** 2) ** 0.5
        if dd > 0:
            off_x, off_y = -ddy / dd * 0.22, ddx / dd * 0.22
        else:
            off_x, off_y = 0, 0
        ax.text(mx + off_x, my + off_y, d["relation"], fontsize=9, fontproperties=FONT_PROP,
                ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.2), zorder=3)

    # ------------------------------------------------------
    # ノード（角丸四角）
    # ------------------------------------------------------
    for n in G.nodes():
        x, y = pos[n]
        w, h = _box_size(labels[n])
        box = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.05,rounding_size=0.12",
            linewidth=1.5, edgecolor="#2f5f96", facecolor="#eaf1fb", zorder=4
        )
        ax.add_patch(box)
        ax.text(x, y, labels[n], fontsize=10.5, fontproperties=FONT_PROP,
                ha="center", va="center", zorder=5, color="#1b3350")

    legend_handles = [
        mpatches.Patch(color=TYPE_STYLE[t]["color"], label=TYPE_STYLE[t]["label"])
        for t in used_types if t in TYPE_STYLE
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, prop=FONT_PROP, fontsize=10, loc="lower left",
                  bbox_to_anchor=(0, 1.02), frameon=False)

    ax.set_title(title, fontproperties=FONT_PROP, fontsize=18, pad=25)
    ax.set_xlim(min(xs) - 3, max(xs) + 3)
    ax.set_ylim(min(ys) - 2, max(ys) + 2)
    ax.axis("off")
    plt.tight_layout()
    return fig


# ============================================================
# ⑩ 箇条書き形式（「識別子：説明文。」の並び）に対応した解析
# ============================================================

import re

BULLET_LINE_RE = re.compile(r'^\s*([^\s：:。]{1,10})[：:]\s*(.+?)\s*$')


def _split_bullets(text):
    """
    テキストをヘッダー行と箇条書き行（識別子：説明文）に分ける。
    箇条書きが2つ未満なら None を返す（＝箇条書き形式ではない）。
    """
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]

    header_lines = []
    bullets = []

    for line in lines:
        m = BULLET_LINE_RE.match(line)
        if m:
            identifier, desc = m.group(1), m.group(2)
            if not desc.endswith("。"):
                desc += "。"
            bullets.append((identifier, desc))
        else:
            header_lines.append(line)

    if len(bullets) < 2:
        return None, None

    return "".join(header_lines), bullets


def _main_noun_of(text):
    """
    文の主要な語（＝その文が定義している対象）を返す。
    「〜の破砕槽。」のような名詞述語文では、依存構造上のROOTが
    その定義対象になっていることを利用する。
    （文中に同じ文字列が別の場所で先に出てきていても、
      ROOTベースで判定するので誤って重複除去されない）
    """
    doc = nlp(text)

    root = None
    for t in doc:
        if t.head == t:
            root = t
            break

    if root is not None and root.pos_ in {"NOUN", "PROPN"}:
        start = root.i
        i = root.i - 1
        while i >= 0 and doc[i].dep_ == "compound" and doc[i].head.i == start:
            start = i
            i -= 1
        return "".join(tok.text for tok in doc[start:root.i + 1])

    # ROOTが名詞でない場合のフォールバック
    components = extract_patent_components_general(doc)
    if components:
        return components[-1]["text"]
    nouns = [t.text for t in doc if t.pos_ in {"NOUN", "PROPN"}]
    return nouns[-1] if nouns else text.strip("。")


def _find_reference_relation(identifier, desc):
    """
    説明文の中に他の識別子（例：装置Ｂ）への言及があれば、
    そこで使われている動詞（得た、得られた等）を関係名として返す。
    """
    m = re.search(re.escape(identifier) + r'(?:で|から|より)?(得られた|得た)', desc)
    if m:
        return m.group(1)
    if identifier in desc:
        return "由来"
    return None


def analyze_claim_with_bullets(text, include_internal_detail=False):
    """
    箇条書き形式（識別子：説明文）にも対応した解析。
    箇条書きが見つからない場合は、通常の analyze_claim() にフォールバックする。
    """
    header_text, bullets = _split_bullets(text)

    if bullets is None:
        return analyze_claim(text)

    # ------------------------------------------------------
    # ① ヘッダー文から全体を表す装置名（コンテナ）を特定
    # ------------------------------------------------------
    container = _main_noun_of(header_text) if header_text.strip() else "全体"

    # ------------------------------------------------------
    # ② 各箇条書きの「実質的な名前（エイリアス）」を決定
    # ------------------------------------------------------
    alias = {}
    for identifier, desc in bullets:
        alias[identifier] = _main_noun_of(desc)

    relations = []

    # container --有する--> 各項目
    for identifier, _ in bullets:
        relations.append({
            "source": container,
            "relation": "有する",
            "target": alias[identifier],
            "type": "has",
        })

    # ------------------------------------------------------
    # ③ 項目間の参照関係（例：装置Ｃは装置Ｂ由来）
    # ------------------------------------------------------
    for identifier, desc in bullets:
        for other_id, _ in bullets:
            if other_id == identifier:
                continue
            if other_id in desc:
                rel_label = _find_reference_relation(other_id, desc) or "由来"
                relations.append({
                    "source": alias[other_id],
                    "relation": rel_label,
                    "target": alias[identifier],
                    "type": "direct",
                })

    # ------------------------------------------------------
    # ④ （オプション）各項目の内部構造も展開する場合
    # ------------------------------------------------------
    components = [{"text": container, "start": -1, "end": -1}]
    for identifier, desc in bullets:
        components.append({"text": alias[identifier], "start": -1, "end": -1})

        if include_internal_detail:
            sub_components, sub_relations = analyze_claim(desc)
            other_ids = [oid for oid, _ in bullets if oid != identifier]
            for sr in sub_relations:
                # 他の項目識別子（装置Ｂなど）への言及に由来する関係は、
                # ③ですでに扱っているのでここでは除外する
                if sr["source"] in other_ids or sr["target"] in other_ids:
                    continue
                relations.append(sr)
            components.extend(sub_components)

    # 重複整理
    unique_relations = []
    seen = set()
    for r in relations:
        key = (r["source"], r["relation"], r["target"])
        if key in seen:
            continue
        seen.add(key)
        unique_relations.append(r)

    return components, unique_relations


# ============================================================
# ⑪ 箇条書きの中の1項目だけを詳細展開する
# ============================================================

def get_bullet_detail(text, identifier):
    """
    箇条書き形式の請求項から、指定した識別子（例："装置Ｂ"）の
    説明文だけを取り出して、その内部構造（構成要素・関係）を返す。

    戻り値: (components, relations, alias)
        alias … その項目の実質的な名前（例:"破砕槽"）
    """
    _, bullets = _split_bullets(text)
    if bullets is None:
        raise ValueError("箇条書き形式が見つかりませんでした。")

    other_ids = [oid for oid, _ in bullets if oid != identifier]

    target_desc = None
    for oid, desc in bullets:
        if oid == identifier:
            target_desc = desc
            break

    if target_desc is None:
        raise ValueError(f"識別子 '{identifier}' が見つかりませんでした。")

    alias = _main_noun_of(target_desc)
    sub_components, sub_relations = analyze_claim(target_desc)

    # 他の項目識別子への言及に由来する関係は除外（この図には不要なので）
    filtered_relations = [
        r for r in sub_relations
        if r["source"] not in other_ids and r["target"] not in other_ids
    ]

    return sub_components, filtered_relations, alias


# ============================================================
# ⑫ 汎用エントリーポイント：どんな請求項でも自動で全体像＋各項目の詳細を出す
# ============================================================

def _has_internal_structure(desc, min_has_count=2):
    """
    説明文の中に「有する」（活用形含む）が複数回出てくる場合、
    内部にさらに構成要素があるとみなす。
    """
    doc = nlp(desc)
    count = sum(1 for t in doc if t.lemma_ in HAS_LEMMAS and t.pos_ == "VERB")
    return count >= min_has_count


def analyze_and_visualize(text, min_has_count=2):
    """
    どんな請求項テキストを渡しても対応する、汎用のエントリーポイント。

    - 箇条書き形式でなければ：1枚の図をそのまま描画する。
    - 箇条書き形式であれば：
        ① まず全体像（各項目間の「有する」「得た」等の関係）を1枚描画し、
        ② 内部に構成要素をさらに持っていそうな項目（"有する"が複数回出る説明文）
           を自動検出し、それぞれについて内部構造の詳細図をもう1枚ずつ描画する。

    戻り値: 描画したタイトルのリスト（確認用）
    """
    _, bullets = _split_bullets(text)
    titles = []

    if bullets is None:
        # 単文形式：今まで通り1枚
        components, relations = analyze_claim(text)
        visualize_relations(relations, title="構成要素間関係")
        titles.append("構成要素間関係")
        return titles

    # ① 全体像
    components, relations = analyze_claim_with_bullets(text, include_internal_detail=False)
    visualize_relations(relations, title="全体構成")
    titles.append("全体構成")

    # ② 内部構造を持っていそうな項目を自動検出して、それぞれ詳細図を描画
    for identifier, desc in bullets:
        if _has_internal_structure(desc, min_has_count=min_has_count):
            try:
                detail_components, detail_relations, alias = get_bullet_detail(text, identifier)
            except ValueError:
                continue
            if not detail_relations:
                continue
            title = f"{alias}（{identifier}）の内部構造"
            visualize_relations(detail_relations, title=title)
            titles.append(title)

    return titles


# ============================================================
# ⑬ 手動で親子関係を付け替える（自動判定できない場合の補正用）
# ============================================================

def reparent_nodes(relations, mapping):
    """
    指定したノードの「有する」による親を、別のノードに付け替える。
    元々あった「有する」の親は取り除き、指定した新しい親からの
    「有する」を必ず追加する（位置関係など他の種類の辺はそのまま残す）。

    mapping: {子ノード名: 新しい親ノード名} の辞書
    例: reparent_nodes(relations, {"回転軸": "回転カッター式破砕機"})
    """
    filtered = [
        r for r in relations
        if not (r["type"] == "has" and r["target"] in mapping)
    ]
    for child, new_parent in mapping.items():
        filtered.append({
            "source": new_parent,
            "relation": "有する",
            "target": child,
            "type": "has",
        })
    return filtered


def merge_nodes(relations, merges):
    """
    「実は同じもの」を指す2つのノード名を1つに統合する。
    Ａ＝Ｂだと判断した場合、Ｂ側のすべての出現をＡに書き換える。

    merges: {統合して消したい名前: 残す方の名前} の辞書
    例: merge_nodes(relations, {"回転カッター式破砕機": "破砕槽"})
        → 「回転カッター式破砕機」という表記をすべて「破砕槽」に統一する
    """
    def rename(name):
        return merges.get(name, name)

    renamed = []
    for r in relations:
        renamed.append({
            "source": rename(r["source"]),
            "relation": r["relation"],
            "target": rename(r["target"]),
            "type": r["type"],
        })

    # 統合した結果、自分自身への矢印（Ａ→Ａ）や重複は取り除く
    unique = []
    seen = set()
    for r in renamed:
        if r["source"] == r["target"]:
            continue
        key = (r["source"], r["relation"], r["target"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique
    
    
text = "基板本体及び前記基板本体上に配置されたチップを有する基板が上下方向に間隔をあけて複数収容可能な収容容器を支持するロードポートであって、前記基板の状態を測定する基板状態測定部、を備え、前記基板状態測定部は、前記基板のうち前記基板本体の少なくとも一部を検出する第一検出部と、前記基板のうち前記チップの少なくとも一部を検出する第二検出部と、前記第一検出部により検出される、少なくとも前記基板本体の下端及び上端の前記上下方向における位置、及び、前記第二検出部により検出される、少なくとも前記チップの下端及び上端の前記上下方向における位置を取得する制御装置と、を備え、前記第一検出部及び前記第二検出部は、それぞれ画像センサであり、前記第一検出部の焦点位置と前記第二検出部の焦点位置とは、前記上下方向に交差する第一方向において互いに異なっているロードポート。"


_, relations = analyze_claim(text)
visualize_relations(relations, title="通信端末の構成要素間関係")
