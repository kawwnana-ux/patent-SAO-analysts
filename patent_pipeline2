import os
import re
import json
from typing import Dict, Any, List

import spacy
from openai import OpenAI


# =========================================================
# 1. GiNZA
# =========================================================

@staticmethod
def _load_ginza():
    try:
        return spacy.load("ja_ginza")
    except Exception:
        try:
            return spacy.load("ja_ginza_electra")
        except Exception as e:
            raise RuntimeError(
                "GiNZAモデルを読み込めませんでした。"
                "requirements.txt を確認してください。"
            ) from e


_NLP = None


def get_nlp():
    global _NLP

    if _NLP is None:
        _NLP = _load_ginza()

    return _NLP


# =========================================================
# 2. 特許請求項の前処理
# =========================================================

def preprocess_claim(text: str) -> str:
    """
    特許請求項をSAO解析しやすい形に軽く前処理する。

    重要:
    ・意味を変えるような大胆な書き換えはしない
    ・前記などの照応表現はLLM側で処理
    ・少なくとも等も削除しない
    """

    if not text:
        return ""

    text = text.strip()

    # 全角空白
    text = text.replace("\u3000", " ")

    # 改行・連続空白
    text = re.sub(r"\s+", " ", text)

    # 「請求項１」などの見出しを軽く除去
    text = re.sub(
        r"^\s*【?請求項\s*[0-9０-９]+\s*】?\s*",
        "",
        text
    )

    # 全角括弧などを統一
    text = text.replace("（", "(")
    text = text.replace("）", ")")

    return text.strip()


# =========================================================
# 3. GiNZA係り受け解析
# =========================================================

def ginza_parse(text: str) -> Dict[str, Any]:

    nlp = get_nlp()

    doc = nlp(text)

    tokens = []

    for token in doc:
        tokens.append({
            "id": token.i,
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "tag": token.tag_,
            "dep": token.dep_,
            "head": token.head.i,
            "head_text": token.head.text,
        })

    sentences = []

    for sent in doc.sents:
        sentences.append({
            "text": sent.text,
            "start": sent.start,
            "end": sent.end
        })

    return {
        "text": text,
        "tokens": tokens,
        "sentences": sentences
    }


# =========================================================
# 4. LLM用JSON Schema
# =========================================================

SAO_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "claim_subject": {
            "type": "string"
        },
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {
                        "type": "string"
                    },
                    "text": {
                        "type": "string"
                    },
                    "node_type": {
                        "type": "string",
                        "enum": [
                            "claim",
                            "component",
                            "function",
                            "property",
                            "terminal",
                            "location",
                            "other"
                        ]
                    },
                    "parent_id": {
                        "type": ["string", "null"]
                    }
                },
                "required": [
                    "id",
                    "text",
                    "node_type",
                    "parent_id"
                ]
            }
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "subject_id": {
                        "type": "string"
                    },
                    "action": {
                        "type": "string"
                    },
                    "object_id": {
                        "type": "string"
                    },
                    "relation_type": {
                        "type": "string",
                        "enum": [
                            "containment",
                            "function",
                            "location",
                            "position",
                            "property",
                            "connection",
                            "other"
                        ]
                    },
                    "level": {
                        "type": "integer"
                    }
                },
                "required": [
                    "subject_id",
                    "action",
                    "object_id",
                    "relation_type",
                    "level"
                ]
            }
        }
    },
    "required": [
        "claim_subject",
        "nodes",
        "relations"
    ]
}


# =========================================================
# 5. LLMプロンプト
# =========================================================

SYSTEM_PROMPT = r"""
あなたは日本語特許請求項のSAO構造解析専門システムです。

目的は、特許請求項から

Subject - Action - Object

の関係を抽出し、さらにそれらを階層構造として表現することです。

【最重要ルール】

1. 特許請求項に書かれていない関係を推測して追加しない。

2. 「前記」は新しいノードにしない。
   以前に出現した同一対象を参照する。

3. 「少なくとも」「少なくとも１つ」などの数量表現は、
   原則として独立ノードにしない。

4. 技術的に意味のある名詞句はできるだけそのまま保持する。

5. 「放熱装置の主面」は、原則として
   「放熱装置の主面」という名詞句として扱う。

6. 並列された対象は別々のノードとして作る。

例:
「正側電源入力端子、負側電源入力端子および出力端子」

なら、

正側電源入力端子
負側電源入力端子
出力端子

を別々のノードにする。

7. 「Aの一部はBとCとの間に位置する」
の場合、

Subject = Aの一部
Action = 位置する
Object = BとCとの間

とする。

8. 階層構造を重要視する。

例えば、

「インテリジェントパワーモジュールは、
放熱装置と、取り付けフレームと、
パワー半導体モジュールとを備える」

なら、

インテリジェントパワーモジュール
 └─ 備える
    ├─ 放熱装置
    ├─ 取り付けフレーム
    └─ パワー半導体モジュール

とする。

9. 「有する」「含む」は単純に「備える」に置換しない。

例えば、

「取り付けフレームは開口部を有する」

なら、

取り付けフレーム
 └─ 有する
    └─ 開口部

とする。

10. 「パワー半導体モジュールは端子を含む」
なら、

パワー半導体モジュール
 └─ 含む
    ├─ 正側電源入力端子
    ├─ 負側電源入力端子
    └─ 出力端子

とする。

11. 「スイッチング機能を有するパワー半導体モジュール」
なら、

パワー半導体モジュール
 └─ 有する
    └─ スイッチング機能

とする。

12. 「備える」は請求項全体の主要構成を示す上位関係として扱う。

13. 「有する」「含む」は、対応する構成要素の下位関係として扱う。

14. parent_idには、階層上の親となるノードIDを入れる。

15. relationsには実際のSAO関係をすべて記録する。

16. level:
   0 = 請求項の最上位
   1 = 主要構成
   2 = 構成要素の内部構成
   3 = さらに下位
   とする。

17. 受動態でも意味を保持する。

例えば
「Aに配置されたB」
なら、
Bを主体として「配置される」関係を作る。

18. 「位置決めされている」は
   「位置決めされる」
   として正規化してよい。

19. 「含み」は「含む」、
   「有し」は「有する」
   のように動詞の基本形に正規化する。

20. ただし「有する」と「備える」と「含む」を
   同一関係として統合してはいけない。

【特に重要】

以下のような誤りを絶対に避ける。

誤:
パワー半導体モジュール → 有する → 放熱装置

本文にその関係が存在しない場合、作ってはいけない。

また、

放熱装置 → 間に位置する → インテリジェントパワーモジュール

のように、係り受けを誤って逆転させてはいけない。

「取り付けフレームの一部は、
正側電源入力端子、負側電源入力端子および出力端子と、
放熱装置との間に位置する」

なら、

Subject:
取り付けフレームの一部

Action:
位置する

Object:
正側電源入力端子、負側電源入力端子および出力端子と、放熱装置との間

である。

【出力】

JSON Schemaに完全に従って出力すること。
"""


# =========================================================
# 6. OpenAIクライアント
# =========================================================

def get_openai_client():

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY が設定されていません。"
        )

    return OpenAI(api_key=api_key)


# =========================================================
# 7. LLM SAO抽出
# =========================================================

def extract_sao_with_llm(
    claim_text: str,
    ginza_result: Dict[str, Any],
    model: str = "gpt-5.6-luna"
) -> Dict[str, Any]:

    client = get_openai_client()

    # GiNZAの解析結果をLLMへの補助情報として渡す
    ginza_text = json.dumps(
        ginza_result,
        ensure_ascii=False,
        indent=2
    )

    user_prompt = f"""
以下の特許請求項をSAO解析してください。

【特許請求項】
{claim_text}

【GiNZA係り受け解析】
{ginza_text}

GiNZA解析は参考情報です。
最終的なSAO構造は特許文の意味と構文を優先してください。

特に、
・前記の照応解決
・備えるの上位構造
・有する/含むの下位構造
・並列構造
・受動態
・位置関係
・「Aの一部」
を正確に処理してください。
"""

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "patent_sao",
                "strict": True,
                "schema": SAO_SCHEMA
            }
        }
    )

    result_text = response.output_text

    return json.loads(result_text)


# =========================================================
# 8. LLM結果の簡易検証
# =========================================================

def validate_sao(result: Dict[str, Any]) -> Dict[str, Any]:

    node_ids = {
        node["id"]
        for node in result.get("nodes", [])
    }

    valid_relations = []

    for relation in result.get("relations", []):

        s = relation["subject_id"]
        o = relation["object_id"]

        if s not in node_ids:
            continue

        if o not in node_ids:
            continue

        if s == o:
            continue

        valid_relations.append(relation)

    result["relations"] = valid_relations

    return result


# =========================================================
# 9. グラフ用データ作成
# =========================================================

def build_graph_data(
    sao_result: Dict[str, Any]
) -> Dict[str, Any]:

    nodes = sao_result.get("nodes", [])
    relations = sao_result.get("relations", [])

    node_map = {
        node["id"]: node
        for node in nodes
    }

    edges = []

    for relation in relations:

        subject = node_map.get(
            relation["subject_id"]
        )

        object_ = node_map.get(
            relation["object_id"]
        )

        if not subject or not object_:
            continue

        edges.append({
            "source": subject["text"],
            "action": relation["action"],
            "target": object_["text"],
            "level": relation["level"],
            "relation_type": relation["relation_type"]
        })

    return {
        "nodes": nodes,
        "edges": edges
    }


# =========================================================
# 10. 全体パイプライン
# =========================================================

def analyze_claim(
    claim_text: str,
    model: str = "gpt-5.6-luna"
) -> Dict[str, Any]:

    # 前処理
    processed = preprocess_claim(claim_text)

    # GiNZA
    ginza_result = ginza_parse(processed)

    # LLM
    sao_result = extract_sao_with_llm(
        processed,
        ginza_result,
        model=model
    )

    # 検証
    sao_result = validate_sao(sao_result)

    # グラフ
    graph_data = build_graph_data(sao_result)

    return {
        "original_text": claim_text,
        "processed_text": processed,
        "ginza": ginza_result,
        "sao": sao_result,
        "graph": graph_data
    }
