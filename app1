import json

import pandas as pd
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

from patent_pipeline import (
    analyze_claim,
    get_nlp
)


# =========================================================
# ページ設定
# =========================================================

st.set_page_config(
    page_title="日本語特許請求項SAO構造分析",
    page_icon="🪼",
    layout="wide"
)


# =========================================================
# タイトル
# =========================================================

st.title("🪼 日本語特許請求項SAO構造分析")

st.caption(
    "LLMによるSAO抽出＋GiNZA係り受け解析＋階層SAOグラフ"
)


# =========================================================
# サイドバー
# =========================================================

with st.sidebar:

    st.header("⚙️ 設定")

    model = st.selectbox(
        "LLMモデル",
        [
            "gpt-5.6-luna",
            "gpt-5.6-terra",
            "gpt-5.6-sol"
        ],
        index=0
    )

    st.info(
        "LLMがSAO抽出の中心です。"
        "GiNZAは係り受け解析と補助情報として使用します。"
    )


# =========================================================
# 入力
# =========================================================

default_claim = """放熱装置と、
前記放熱装置の主面に配置された少なくとも１つの取り付けフレームと、
スイッチング機能を有する少なくとも１つのパワー半導体モジュールと、
を備え、
前記パワー半導体モジュールは、正側電源入力端子、負側電源入力端子および出力端子を含み、
前記取り付けフレームは、少なくとも１つの開口部を有し、
前記パワー半導体モジュールは、前記開口部により前記取り付けフレームに対して位置決めされており、
前記取り付けフレームの一部は、前記正側電源入力端子、前記負側電源入力端子および前記出力端子と、前記放熱装置との間に位置する、
インテリジェントパワーモジュール。"""


claim = st.text_area(
    "特許請求項",
    value=default_claim,
    height=320
)


# =========================================================
# 解析
# =========================================================

if st.button(
    "🔍 SAO解析を実行",
    type="primary",
    use_container_width=True
):

    if not claim.strip():

        st.warning("特許請求項を入力してください。")
        st.stop()

    try:

        with st.spinner("解析中..."):

            result = analyze_claim(
                claim,
                model=model
            )

        st.session_state["analysis_result"] = result

        st.success("解析が完了しました。")

    except Exception as e:

        st.error("解析中にエラーが発生しました。")

        st.exception(e)

        st.stop()


# =========================================================
# 結果表示
# =========================================================

if "analysis_result" in st.session_state:

    result = st.session_state["analysis_result"]

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🌳 階層SAO",
            "🔗 SAO表",
            "🧩 GiNZA",
            "📄 JSON"
        ]
    )


    # =====================================================
    # 1. 階層SAO
    # =====================================================

    with tab1:

        st.subheader("🌳 階層SAO構造")

        sao = result["sao"]

        nodes = {
            n["id"]: n
            for n in sao["nodes"]
        }

        relations = sao["relations"]

        # 親子構造を表示
        def show_children(
            parent_id,
            depth=0
        ):

            children = [
                r for r in relations
                if r["subject_id"] == parent_id
            ]

            for r in children:

                target = nodes.get(
                    r["object_id"]
                )

                if target is None:
                    continue

                prefix = "　" * depth

                st.markdown(
                    f"{prefix}**{nodes[parent_id]['text']}** "
                    f"→ `{r['action']}` → "
                    f"**{target['text']}**"
                )

                show_children(
                    target["id"],
                    depth + 1
                )


        # ルート候補
        object_ids = {
            r["object_id"]
            for r in relations
        }

        root_nodes = [
            n for n in sao["nodes"]
            if n["id"] not in object_ids
        ]

        for root in root_nodes:

            st.markdown(
                f"### {root['text']}"
            )

            show_children(
                root["id"],
                1
            )


    # =====================================================
    # 2. SAO表
    # =====================================================

    with tab2:

        st.subheader("🔗 SAOトリプル")

        node_map = {
            n["id"]: n["text"]
            for n in sao["nodes"]
        }

        rows = []

        for r in sao["relations"]:

            rows.append({
                "Subject": node_map.get(
                    r["subject_id"],
                    ""
                ),
                "Action": r["action"],
                "Object": node_map.get(
                    r["object_id"],
                    ""
                ),
                "階層": r["level"],
                "関係種別": r["relation_type"]
            })

        if rows:

            df = pd.DataFrame(rows)

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "SAO関係が抽出されませんでした。"
            )


    # =====================================================
    # 3. GiNZA
    # =====================================================

    with tab3:

        st.subheader("🧩 GiNZA係り受け解析")

        ginza = result["ginza"]

        df = pd.DataFrame(
            ginza["tokens"]
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )


    # =====================================================
    # 4. JSON
    # =====================================================

    with tab4:

        st.subheader("📄 LLM SAO JSON")

        st.json(
            result["sao"]
        )


    # =====================================================
    # ネットワークグラフ
    # =====================================================

    st.divider()

    st.subheader("🕸️ SAOネットワーク")

    graph = result["graph"]

    G = nx.DiGraph()

    for node in graph["nodes"]:

        G.add_node(
            node["text"],
            node_type=node["node_type"],
            level=node.get("parent_id")
        )

    for edge in graph["edges"]:

        G.add_edge(
            edge["source"],
            edge["target"],
            action=edge["action"]
        )

    if len(G.nodes) > 0:

        fig, ax = plt.subplots(
            figsize=(16, 10)
        )

        pos = nx.spring_layout(
            G,
            seed=42,
            k=1.8
        )

        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_size=2600
        )

        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            arrows=True,
            arrowsize=20
        )

        nx.draw_networkx_labels(
            G,
            pos,
            ax=ax,
            font_size=9
        )

        edge_labels = {
            (
                u,
                v
            ): data["action"]
            for u, v, data
            in G.edges(data=True)
        }

        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            ax=ax,
            font_size=8
        )

        ax.axis("off")

        st.pyplot(
            fig,
            use_container_width=True
        )
