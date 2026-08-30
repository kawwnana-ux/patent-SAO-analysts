# 特許請求項解析 Webアプリ

Google Colabで作成した請求項解析プログラムをStreamlitでWeb化したものです。

## 構成

- `app.py`：Web画面
- `patent_pipeline.py`：既存の解析ロジック
- `requirements.txt`：必要ライブラリ

## 起動

```bash
pip install -r requirements.txt
streamlit run app.py
```

ブラウザで表示されたURLを開きます。

既存の解析ロジックはできるだけ維持し、Colab固有の処理だけWeb実行向けに変更しています。
