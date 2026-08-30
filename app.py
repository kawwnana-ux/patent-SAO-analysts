import streamlit as st
import matplotlib.pyplot as plt

from patent_pipeline import analyze_claim, get_bullet_detail, _split_bullets, visualize_relations

st.set_page_config(
    page_title="特許請求項解析システム",
    page_icon="📑",
    layout="wide",
)

st.title("📑 特許請求項解析システム")
st.write("特許請求項を入力すると、構成要素・関係・内部構造を可視化します。")

text = st.text_area(
    "請求項を入力してください",
    height=280,
    placeholder="ここに特許請求項を貼り付けてください。",
)

if st.button("🔍 解析する", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("請求項テキストを入力してください。")
        st.stop()

    try:
        with st.spinner("解析中です…"):
            components, relations = analyze_claim(text)

        st.subheader("1. 構成要素")
        if components:
            st.dataframe(
                [{"構成要素": c["text"]} for c in components],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("構成要素を抽出できませんでした。")

        st.subheader("2. 構成要素間の関係")
        if relations:
            st.dataframe(
                [
                    {
                        "起点": r["source"],
                        "関係": r["relation"],
                        "終点": r["target"],
                        "種類": r["type"],
                    }
                    for r in relations
                ],
                use_container_width=True,
                hide_index=True,
            )

            fig = visualize_relations(relations, title="構成要素間関係")
            if fig is not None:
                st.subheader("3. 関係図")
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("関係を抽出できませんでした。")

        # 箇条書き形式なら各項目の内部構造も表示
        _, bullets = _split_bullets(text)
        if bullets:
            st.subheader("4. 内部構造")
            for identifier, _ in bullets:
                try:
                    _, detail_relations, alias = get_bullet_detail(text, identifier)
                except ValueError:
                    continue

                if not detail_relations:
                    continue

                with st.expander(f"{identifier}：{alias}", expanded=True):
                    fig = visualize_relations(
                        detail_relations,
                        title=f"{alias}（{identifier}）の内部構造",
                    )
                    if fig is not None:
                        st.pyplot(fig)
                        plt.close(fig)

    except Exception as e:
        st.error("解析中にエラーが発生しました。")
        st.exception(e)
