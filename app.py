import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import base64
import anthropic
import openai as openai_lib
from duckduckgo_search import DDGS

# ─── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ITBI BH — Análise de Imóveis",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "itbi_2008_2026.parquet")

# ─── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Carregando base ITBI (501 k transações)...")
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["data_quitacao"] = pd.to_datetime(df["data_quitacao"], errors="coerce")
    for col in ("valor_declarado", "valor_base_calculo", "valor_m2",
                "area_construida_adquirida", "area_adquirida_unidades_somadas"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["building_key"] = df["endereco"].str.split(" - ").str[0].str.strip()
    df["building_key_norm"] = df["building_key"].str.upper()
    return df

# ─── Search helpers ────────────────────────────────────────────────────────────

def search_buildings(df: pd.DataFrame, query: str) -> list[str]:
    q = query.upper().strip()
    if len(q) < 3:
        return []
    mask = df["building_key_norm"].str.contains(q, na=False, regex=False)
    return sorted(df[mask]["building_key"].dropna().unique().tolist())

def get_building_df(df: pd.DataFrame, building_key: str) -> pd.DataFrame:
    return (
        df[df["building_key"] == building_key]
        .sort_values("data_quitacao", ascending=False)
        .reset_index(drop=True)
    )

# ─── Photo helpers ─────────────────────────────────────────────────────────────

def file_to_base64(uploaded_file) -> dict:
    data = base64.b64encode(uploaded_file.read()).decode("utf-8")
    name = uploaded_file.name.lower()
    if name.endswith(".png"):
        media_type = "image/png"
    elif name.endswith(".webp"):
        media_type = "image/webp"
    elif name.endswith(".gif"):
        media_type = "image/gif"
    else:
        media_type = "image/jpeg"
    return {"data": data, "media_type": media_type, "name": uploaded_file.name}

def display_photos_grid(photos: list[dict], cols: int = 3):
    if not photos:
        return
    chunks = [photos[i:i+cols] for i in range(0, len(photos), cols)]
    for chunk in chunks:
        c_list = st.columns(cols)
        for j, photo in enumerate(chunk):
            img_bytes = base64.b64decode(photo["data"])
            c_list[j].image(img_bytes, caption=photo.get("name", ""), use_container_width=True)

# ─── AI providers + web search ────────────────────────────────────────────────

CLAUDE_MODELS = ["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5-20251001"]
GPT_MODELS    = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]

_SEARCH_PROMPT = (
    "Pesquise anúncios de imóveis à venda no endereço ou nas proximidades de:\n{address}\nBelo Horizonte, MG\n\n"
    "Faça múltiplas buscas: ZAP Imóveis, Viva Real, OLX, Imovelweb, QuintoAndar.\n"
    "Para cada anúncio encontrado extraia: título, URL, preço (R$), área (m²), preço/m², descrição.\n\n"
    "Retorne SOMENTE um JSON válido com esta estrutura (sem markdown, sem texto fora do JSON):\n"
    '{{"listings": [{{"title":"...","url":"...","price":000000,"area":00.0,"price_m2":0000.0,"description":"..."}}],'
    '"summary":"Breve resumo do mercado nessa localização"}}'
)

def ddg_search(query: str, max_results: int = 8) -> list[dict]:
    try:
        with DDGS() as ddgs:
            return [
                {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
                for r in ddgs.text(query, max_results=max_results)
            ]
    except Exception as e:
        return [{"error": str(e)}]

def _parse_listings_json(raw: str) -> tuple[list[dict], str]:
    try:
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start >= 0:
            data = json.loads(raw[start:end])
            return data.get("listings", []), data.get("summary", "")
    except Exception:
        pass
    return [], raw

def _search_listings_claude(address: str, api_key: str, model: str) -> tuple[list[dict], str]:
    client = anthropic.Anthropic(api_key=api_key)
    tools = [{
        "name": "search_web",
        "description": "Search the web for real estate listings and market data",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }]
    messages = [{"role": "user", "content": _SEARCH_PROMPT.format(address=address)}]
    for _ in range(8):
        response = client.messages.create(model=model, max_tokens=4096, tools=tools, messages=messages)
        if response.stop_reason == "end_turn":
            raw = next((b.text for b in response.content if hasattr(b, "text")), "")
            return _parse_listings_json(raw)
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use" and block.name == "search_web":
                    results = ddg_search(block.input["query"])
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(results, ensure_ascii=False),
                    })
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
    return [], "Limite de buscas atingido."

def _search_listings_gpt(address: str, api_key: str, model: str) -> tuple[list[dict], str]:
    client = openai_lib.OpenAI(api_key=api_key)
    tools = [{
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for real estate listings and market data",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }]
    messages = [{"role": "user", "content": _SEARCH_PROMPT.format(address=address)}]
    for _ in range(8):
        response = client.chat.completions.create(
            model=model, max_tokens=4096, tools=tools, tool_choice="auto", messages=messages,
        )
        choice = response.choices[0]
        if choice.finish_reason == "stop":
            return _parse_listings_json(choice.message.content or "")
        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)
            for tc in (choice.message.tool_calls or []):
                if tc.function.name == "search_web":
                    args = json.loads(tc.function.arguments)
                    results = ddg_search(args["query"])
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(results, ensure_ascii=False),
                    })
    return [], "Limite de buscas atingido."

def search_listings(address: str, provider: str, api_key: str, model: str) -> tuple[list[dict], str]:
    if provider == "Claude":
        return _search_listings_claude(address, api_key, model)
    return _search_listings_gpt(address, api_key, model)

# ─── Competitive analysis ──────────────────────────────────────────────────────

def _ad_text_block(ad: dict, label: str = "") -> str:
    parts = []
    if label:
        parts.append(f"[{label}]")
    parts.append(f"Título: {ad.get('title', 'N/A')}")
    if ad.get("price"):
        parts.append(f"Preço pedido: R$ {float(ad['price']):,.0f}")
    if ad.get("area"):
        parts.append(f"Área: {float(ad['area']):.1f} m²")
    if ad.get("price_m2"):
        parts.append(f"R$/m²: R$ {float(ad['price_m2']):,.0f}")
    if ad.get("floor"):
        parts.append(f"Andar: {ad['floor']}")
    if ad.get("bedrooms"):
        parts.append(f"Quartos: {ad['bedrooms']}")
    if ad.get("parking"):
        parts.append(f"Vagas: {ad['parking']}")
    if ad.get("url"):
        parts.append(f"Link: {ad['url']}")
    if ad.get("notes"):
        parts.append(f"Observações: {ad['notes']}")
    n_photos = len(ad.get("photos") or [])
    if n_photos:
        parts.append(f"Fotos: {n_photos} imagem(ns) incluída(s)")
    return " | ".join(parts)

def _build_competitive_prompt(
    building_df: pd.DataFrame,
    target_ad: dict,
    comparative_ads: list[dict],
    online_listings: list[dict],
) -> str:
    building_key = building_df["building_key"].iloc[0] if len(building_df) > 0 else "N/A"

    valid_m2 = building_df["valor_m2"].dropna()
    valid_total = building_df["valor_declarado"].dropna()
    avg_m2 = valid_m2.mean() if not valid_m2.empty else None
    median_m2 = valid_m2.median() if not valid_m2.empty else None
    p25_m2 = valid_m2.quantile(0.25) if not valid_m2.empty else None
    p75_m2 = valid_m2.quantile(0.75) if not valid_m2.empty else None
    date_min = str(building_df["data_quitacao"].min())[:10] if len(building_df) > 0 else "N/A"
    date_max = str(building_df["data_quitacao"].max())[:10] if len(building_df) > 0 else "N/A"

    recent_cutoff = pd.Timestamp.now() - pd.DateOffset(years=2)
    recent = building_df[building_df["data_quitacao"] >= recent_cutoff].copy()
    recent_m2 = recent["valor_m2"].dropna()

    hist = []
    for _, row in building_df.head(12).iterrows():
        hist.append({
            "data": str(row.get("data_quitacao", ""))[:10],
            "unidade": str(row.get("endereco", "")),
            "valor": float(row["valor_declarado"]) if pd.notna(row.get("valor_declarado")) else None,
            "area_m2": float(row["area_construida_adquirida"]) if pd.notna(row.get("area_construida_adquirida")) else None,
            "valor_m2": float(row["valor_m2"]) if pd.notna(row.get("valor_m2")) else None,
            "andar": str(row["andar"]) if pd.notna(row.get("andar")) else None,
        })

    has_photos = bool(target_ad.get("photos") or any(a.get("photos") for a in comparative_ads))
    photo_note = (
        "\n\nFotos dos imóveis foram incluídas nesta análise. Avalie-as criteriosamente: "
        "padrão de acabamento, estado de conservação, luminosidade, vista, tamanho real percebido "
        "e quaisquer diferenciais ou problemas visíveis."
    ) if has_photos else ""

    target_block = _ad_text_block(target_ad, "ANÚNCIO ALVO")
    comp_blocks = "\n".join(_ad_text_block(ad, f"COMPARATIVO {i+1}") for i, ad in enumerate(comparative_ads)) or "Nenhum comparativo adicionado."
    online_blocks = "\n".join(_ad_text_block(l, f"ONLINE {i+1}") for i, l in enumerate(online_listings[:10])) or "Nenhuma busca online realizada."

    return f"""Você é um consultor imobiliário sênior especializado no mercado de Belo Horizonte. Realize uma ANÁLISE COMPETITIVA detalhada tendo o anúncio alvo como foco central.{photo_note}

## EDIFÍCIO DE REFERÊNCIA
{building_key} — Belo Horizonte, MG

## BASE DE MERCADO — ITBI (transações oficiais da Prefeitura de BH)
- Total de transações: {len(building_df)} | Período: {date_min} → {date_max}
- R$/m² médio histórico: {f"R$ {avg_m2:,.0f}" if avg_m2 else "N/A"} | Mediana: {f"R$ {median_m2:,.0f}" if median_m2 else "N/A"}
- Intervalo interquartil (P25–P75): {f"R$ {p25_m2:,.0f} – R$ {p75_m2:,.0f}" if p25_m2 and p75_m2 else "N/A"}
- Transações últimos 2 anos: {len(recent)} | R$/m² médio recente: {f"R$ {recent_m2.mean():,.0f}" if not recent_m2.empty else "N/A"}

Últimas 12 transações (unidades do edifício):
{json.dumps(hist, ensure_ascii=False, indent=2)}

## ANÚNCIO ALVO
{target_block}

## ANÚNCIOS COMPARATIVOS (adicionados pelo usuário como base)
{comp_blocks}

## ANÚNCIOS ONLINE (mercado atual)
{online_blocks}

---

Gere a ANÁLISE COMPETITIVA completa em markdown com os headers abaixo:

### Posicionamento de Preço
Calcule e apresente em tabela o R$/m² do anúncio alvo vs cada comparativo vs mediana ITBI vs média recente.
Mostre o desvio percentual em cada caso. Veredicto claro: o preço está ACIMA / NA LINHA / ABAIXO do mercado.

### Score Competitivo (0–10)
Atribua uma nota justificada para o anúncio alvo em:
- **Preço** (0 = muito caro; 10 = excelente valor)
- **Área e planta** (relação tamanho × preço × configuração)
- **Andar / localização no edifício** (se disponível)
- **Acabamento e conservação** (baseado nas fotos e observações; se não houver fotos, indique)
- **Score geral ponderado**

### Análise comparativo a comparativo
Para cada comparativo: como se compara ao alvo em preço/m², características e condição. Quem leva vantagem e por quê.

### Valor Justo Estimado
Com base em todos os dados: valor justo mínimo · valor justo máximo · proposta de abertura sugerida.
Mostre o raciocínio de cálculo.

### Pontos Positivos do Anúncio Alvo
Diferenciais que justificam o preço ou que são vantagens reais de compra.
Se houver fotos, mencione aspectos visuais positivos observados.

### Red Flags e Pontos de Atenção
Riscos, inconsistências, itens que justificam desconto ou cautela.
Se houver fotos, mencione aspectos visuais negativos.

### Estratégia de Negociação
- Valor de abertura com justificativa
- Argumentos-chave para negociar desconto (baseados nos dados comparativos)
- Limite máximo recomendado e por quê
- Táticas e timing sugeridos

Seja preciso com R$. Escreva em português. Baseie tudo exclusivamente nos dados fornecidos.
"""

def _build_multimodal_content(
    text_prompt: str,
    target_ad: dict,
    comparative_ads: list[dict],
    provider: str,
) -> list:
    content = []

    def add_images(photos: list[dict], label: str):
        if not photos:
            return
        if provider == "Claude":
            content.append({"type": "text", "text": f"\n--- {label} ---"})
            for p in photos:
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": p["media_type"], "data": p["data"]},
                })
        else:
            content.append({"type": "text", "text": f"\n--- {label} ---"})
            for p in photos:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{p['media_type']};base64,{p['data']}"},
                })

    add_images(target_ad.get("photos") or [], f"Fotos do anúncio alvo: {target_ad.get('title', '')}")
    for i, ad in enumerate(comparative_ads):
        add_images(ad.get("photos") or [], f"Fotos comparativo {i+1}: {ad.get('title', '')}")

    content.append({"type": "text", "text": text_prompt})
    return content

def generate_competitive_analysis(
    building_df: pd.DataFrame,
    target_ad: dict,
    comparative_ads: list[dict],
    online_listings: list[dict],
    provider: str,
    api_key: str,
    model: str,
) -> str:
    text_prompt = _build_competitive_prompt(building_df, target_ad, comparative_ads, online_listings)
    content = _build_multimodal_content(text_prompt, target_ad, comparative_ads, provider)

    if provider == "Claude":
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text
    else:
        client = openai_lib.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": content}],
        )
        return response.choices[0].message.content or ""

# ─── Charts ────────────────────────────────────────────────────────────────────

def build_yearly_avg_chart(building_df: pd.DataFrame) -> go.Figure:
    df = building_df.dropna(subset=["ano", "valor_m2", "valor_declarado"]).copy()
    df["ano"] = df["ano"].astype(int)
    by_year = (
        df.groupby("ano")
        .agg(
            avg_total=("valor_declarado", "mean"),
            avg_m2=("valor_m2", "mean"),
            n=("valor_declarado", "count"),
        )
        .reset_index()
        .sort_values("ano")
    )

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Valor médio total por ano (R$)", "Preço médio R$/m² por ano"),
    )
    fig.add_trace(go.Bar(
        x=by_year["ano"], y=by_year["avg_total"],
        name="Valor médio (R$)", marker_color="#1976D2",
        text=by_year["n"].apply(lambda n: f"{n} trans."), textposition="outside",
        hovertemplate="<b>%{x}</b><br>Média: R$ %{y:,.0f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=by_year["ano"], y=by_year["avg_m2"],
        mode="markers+lines", name="R$/m² médio",
        marker=dict(size=8, color="#FF7043"),
        line=dict(color="#FF7043", width=2),
        hovertemplate="<b>%{x}</b><br>R$/m²: R$ %{y:,.0f}<extra></extra>",
    ), row=1, col=2)
    fig.update_layout(
        height=340, showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=40, b=30, l=70, r=20),
    )
    fig.update_yaxes(tickprefix="R$ ", tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(tickprefix="R$ ", tickformat=",.0f", row=1, col=2)
    fig.update_xaxes(dtick=1, tickformat="d")
    return fig


def build_comparison_chart(
    building_df: pd.DataFrame,
    target_ad: dict | None,
    comparative_ads: list[dict],
    online_listings: list[dict],
) -> go.Figure:
    fig = go.Figure()
    now = pd.Timestamp.now()

    hist = building_df.dropna(subset=["data_quitacao", "valor_m2"])
    if not hist.empty:
        fig.add_trace(go.Scatter(
            x=hist["data_quitacao"], y=hist["valor_m2"],
            mode="markers", name="ITBI (histórico)",
            marker=dict(size=8, color="#1976D2", symbol="circle", opacity=0.55),
            hovertemplate="<b>ITBI</b><br>%{x|%d/%m/%Y}<br>R$/m²: R$ %{y:,.0f}<extra></extra>",
        ))

    valid_online = [l for l in online_listings if l.get("price_m2") and float(l["price_m2"]) > 0]
    if valid_online:
        fig.add_trace(go.Scatter(
            x=[now] * len(valid_online),
            y=[float(l["price_m2"]) for l in valid_online],
            mode="markers", name="Anúncios online",
            marker=dict(size=12, color="#FF5722", symbol="diamond"),
            text=[l.get("title", "")[:50] for l in valid_online],
            hovertemplate="<b>%{text}</b><br>R$/m²: R$ %{y:,.0f}<extra></extra>",
        ))

    valid_comps = [a for a in comparative_ads if a.get("price_m2") and float(a["price_m2"]) > 0]
    if valid_comps:
        fig.add_trace(go.Scatter(
            x=[now] * len(valid_comps),
            y=[float(a["price_m2"]) for a in valid_comps],
            mode="markers", name="Comparativos",
            marker=dict(size=14, color="#9C27B0", symbol="star"),
            text=[a.get("title", "")[:50] for a in valid_comps],
            hovertemplate="<b>%{text}</b><br>R$/m²: R$ %{y:,.0f}<extra></extra>",
        ))

    if target_ad and target_ad.get("price_m2") and float(target_ad["price_m2"]) > 0:
        fig.add_trace(go.Scatter(
            x=[now],
            y=[float(target_ad["price_m2"])],
            mode="markers+text",
            name="🎯 ALVO",
            marker=dict(size=22, color="#F44336", symbol="star",
                        line=dict(color="#000", width=2)),
            text=[f"🎯 {target_ad.get('title', 'ALVO')[:30]}"],
            textposition="top center",
            textfont=dict(size=12, color="#F44336"),
            hovertemplate=f"<b>🎯 ALVO: {target_ad.get('title', '')[:50]}</b><br>R$/m²: R$ %{{y:,.0f}}<extra></extra>",
        ))

    # Add horizontal reference lines for ITBI stats
    valid_m2 = building_df["valor_m2"].dropna()
    if not valid_m2.empty:
        median_val = valid_m2.median()
        fig.add_hline(
            y=median_val, line_dash="dot", line_color="#1976D2", opacity=0.5,
            annotation_text=f"Mediana ITBI: R$ {median_val:,.0f}/m²",
            annotation_position="bottom right",
        )
        recent_cutoff = pd.Timestamp.now() - pd.DateOffset(years=2)
        recent = building_df[building_df["data_quitacao"] >= recent_cutoff]["valor_m2"].dropna()
        if not recent.empty and len(recent) >= 3:
            recent_avg = recent.mean()
            fig.add_hline(
                y=recent_avg, line_dash="dash", line_color="#FF7043", opacity=0.6,
                annotation_text=f"Média ITBI recente (2a): R$ {recent_avg:,.0f}/m²",
                annotation_position="top right",
            )

    fig.update_layout(
        title="Posicionamento de mercado — R$/m² (ITBI histórico × comparativos × anúncio alvo)",
        xaxis_title="Data", yaxis_title="R$/m²",
        height=480, plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(tickprefix="R$ ", tickformat=",.0f"),
        margin=dict(t=60, b=40, l=80, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

# ─── Session state ─────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "building_key": None,
        "building_df": None,
        "online_listings": [],
        "comparative_ads": [],
        "target_ad": None,
        "analysis": None,
        "market_summary": "",
        "search_done": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ─── Helpers ───────────────────────────────────────────────────────────────────

def fmt_r(val) -> str:
    return f"R$ {val:,.0f}" if pd.notna(val) and val else "–"

def fmt_m2(val) -> str:
    return f"{val:.1f} m²" if pd.notna(val) and val else "–"

def _ad_form(form_key: str, submit_label: str, title_placeholder: str = "Ex: Apto 302 – ZAP Imóveis"):
    """Render a reusable ad input form. Returns (submitted: bool, ad_dict: dict)."""
    with st.form(form_key, clear_on_submit=True):
        r1c1, r1c2, r1c3 = st.columns(3)
        ad_title   = r1c1.text_input("Título / Referência *", placeholder=title_placeholder)
        ad_price   = r1c2.number_input("Preço pedido (R$)", min_value=0.0, step=10_000.0, format="%.0f")
        ad_area    = r1c3.number_input("Área (m²)", min_value=0.0, step=1.0, format="%.1f")

        r2c1, r2c2, r2c3 = st.columns(3)
        ad_floor   = r2c1.text_input("Andar", placeholder="Ex: 8 ou Térreo")
        ad_beds    = r2c2.number_input("Quartos", min_value=0, step=1, format="%d")
        ad_parking = r2c3.number_input("Vagas", min_value=0, step=1, format="%d")

        ad_url     = st.text_input("Link do anúncio", placeholder="https://...")
        ad_notes   = st.text_area("Observações / Diferenciais", placeholder="Ex: reformado, 2 suítes, vista livre, condomínio R$ 800/mês…")
        ad_photos  = st.file_uploader(
            "Fotos do imóvel",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            help="Adicione fotos para incluir análise visual no relatório (Claude e GPT-4o suportam imagens).",
        )

        submitted = st.form_submit_button(submit_label)

    if submitted:
        if not ad_title.strip():
            st.warning("Preencha ao menos o título.")
            return False, {}
        pm2 = round(ad_price / ad_area, 2) if ad_area > 0 and ad_price > 0 else None
        photos = [file_to_base64(f) for f in (ad_photos or [])]
        ad = {
            "title":    ad_title.strip(),
            "price":    ad_price or None,
            "area":     ad_area or None,
            "price_m2": pm2,
            "floor":    ad_floor.strip() or None,
            "bedrooms": int(ad_beds) if ad_beds > 0 else None,
            "parking":  int(ad_parking) if ad_parking > 0 else None,
            "url":      ad_url.strip() or None,
            "notes":    ad_notes.strip() or None,
            "photos":   photos,
        }
        return True, ad
    return False, {}

def _render_ad_card(ad: dict, idx: int, collection_key: str | None = None):
    """Display an ad card in an expander. If collection_key is set, shows a remove button."""
    label = ad["title"]
    if ad.get("price"):
        label += f" — R$ {ad['price']:,.0f}"
    if ad.get("price_m2"):
        label += f" · R$ {ad['price_m2']:,.0f}/m²"

    with st.expander(label, expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Preço", fmt_r(ad.get("price")))
        c2.metric("Área", fmt_m2(ad.get("area")))
        c3.metric("R$/m²", fmt_r(ad.get("price_m2")))
        c4.metric("Andar", ad.get("floor") or "–")

        if ad.get("bedrooms") or ad.get("parking"):
            b1, b2 = st.columns(2)
            b1.write(f"**Quartos:** {ad.get('bedrooms') or '–'}")
            b2.write(f"**Vagas:** {ad.get('parking') or '–'}")

        if ad.get("url"):
            st.markdown(f"[🔗 Ver anúncio]({ad['url']})")
        if ad.get("notes"):
            st.caption(f"📝 {ad['notes']}")

        photos = ad.get("photos") or []
        if photos:
            st.write(f"**{len(photos)} foto(s):**")
            display_photos_grid(photos, cols=min(3, len(photos)))
        else:
            st.caption("Sem fotos.")

        if collection_key is not None:
            if st.button("🗑 Remover", key=f"del_{collection_key}_{idx}"):
                st.session_state[collection_key].pop(idx)
                st.rerun()

# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    init_state()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configurações")

        provider = st.radio("Provedor de IA", options=["Claude", "GPT"], horizontal=True)

        if provider == "Claude":
            api_key = st.text_input("Anthropic API Key", type="password",
                                    value=os.environ.get("ANTHROPIC_API_KEY", ""))
            model = st.selectbox("Modelo", CLAUDE_MODELS)
        else:
            api_key = st.text_input("OpenAI API Key", type="password",
                                    value=os.environ.get("OPENAI_API_KEY", ""))
            model = st.selectbox("Modelo", GPT_MODELS)

        if not api_key:
            st.warning("Insira sua API key para busca online e análise.")
        else:
            st.success(f"{provider} — {model}")

        st.divider()
        st.caption("**Base:** ITBI BH 2008–2026 · Prefeitura BH")

        if st.session_state.building_key:
            st.divider()
            st.caption(f"**Edifício ativo:**\n{st.session_state.building_key}")

            # Quick status
            has_target = st.session_state.target_ad is not None
            n_comps    = len(st.session_state.comparative_ads)
            st.caption(
                f"{'✅' if has_target else '⬜'} Anúncio alvo  |  "
                f"{n_comps} comparativo(s)"
            )

            if st.button("Limpar tudo"):
                for k, v in {
                    "building_key": None, "building_df": None,
                    "online_listings": [], "comparative_ads": [],
                    "target_ad": None, "analysis": None,
                    "market_summary": "", "search_done": False,
                }.items():
                    st.session_state[k] = v
                st.rerun()

    # ── Header ─────────────────────────────────────────────────────────────────
    st.title("🏠 ITBI BH — Análise Competitiva de Imóveis")
    st.caption(
        "Encontrou um apartamento de interesse? Pesquise o edifício, defina o anúncio alvo, "
        "adicione comparativos e gere uma análise competitiva completa com IA."
    )

    df = load_data()

    # ── 1. Busca ───────────────────────────────────────────────────────────────
    st.subheader("1. Buscar edifício na base ITBI")
    query = st.text_input(
        "Endereço",
        placeholder="Ex: PATAGONIA 1023  ou  AV AFONSO PENA  ou  RUA GUAJAJARAS",
        label_visibility="collapsed",
    )
    buildings = search_buildings(df, query) if query else []

    if buildings:
        col_sel, col_btn = st.columns([5, 1])
        with col_sel:
            selected_building = st.selectbox(f"{len(buildings)} edifício(s):", options=buildings)
        with col_btn:
            st.write("")
            st.write("")
            if st.button("Selecionar", type="primary", use_container_width=True):
                bdf = get_building_df(df, selected_building)
                st.session_state.building_key    = selected_building
                st.session_state.building_df     = bdf
                st.session_state.online_listings = []
                st.session_state.comparative_ads = []
                st.session_state.target_ad       = None
                st.session_state.analysis        = None
                st.session_state.market_summary  = ""
                st.session_state.search_done     = False
                st.rerun()
    elif query and len(query) >= 3:
        st.warning("Nenhum edifício encontrado. Tente só o nome da rua sem número.")

    if not st.session_state.building_key or st.session_state.building_df is None:
        return

    bdf     = st.session_state.building_df
    bkey    = st.session_state.building_key
    n_total = len(bdf)
    n_units = bdf["endereco"].nunique()

    if n_total == 0:
        st.warning("Nenhuma transação encontrada para este edifício.")
        return

    # ── 2. Overview ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"2. Overview ITBI — {bkey}")

    latest      = bdf.iloc[0]
    valid_m2    = bdf["valor_m2"].dropna()
    valid_total = bdf["valor_declarado"].dropna()
    date_min    = bdf["data_quitacao"].min()
    date_max    = bdf["data_quitacao"].max()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Transações", n_total)
    k2.metric("Unidades únicas", n_units)
    k3.metric("Período", f"{str(date_min)[:7]} → {str(date_max)[:7]}")
    k4.metric("Última venda", str(latest["data_quitacao"])[:10] if pd.notna(latest["data_quitacao"]) else "–")
    k5.metric("R$/m² médio histórico", fmt_r(valid_m2.mean()) if not valid_m2.empty else "–")
    k6.metric("Valor médio histórico", fmt_r(valid_total.mean()) if not valid_total.empty else "–")

    st.plotly_chart(build_yearly_avg_chart(bdf), use_container_width=True)

    col_exp, col_rec = st.columns(2)
    with col_exp:
        st.markdown("**Top 5 — Maiores valores declarados**")
        top5_price = (
            bdf.dropna(subset=["valor_declarado"]).nlargest(5, "valor_declarado")
            [["data_quitacao","endereco","valor_declarado","valor_m2","area_construida_adquirida"]]
            .rename(columns={"data_quitacao":"Data","endereco":"Unidade",
                              "valor_declarado":"Valor (R$)","valor_m2":"R$/m²",
                              "area_construida_adquirida":"Área (m²)"})
        )
        top5_price["Data"]       = top5_price["Data"].dt.strftime("%d/%m/%Y")
        top5_price["Valor (R$)"] = top5_price["Valor (R$)"].apply(lambda v: f"R$ {v:,.0f}" if pd.notna(v) else "–")
        top5_price["R$/m²"]      = top5_price["R$/m²"].apply(lambda v: f"R$ {v:,.0f}" if pd.notna(v) else "–")
        top5_price["Área (m²)"]  = top5_price["Área (m²)"].apply(lambda v: f"{v:.1f}" if pd.notna(v) else "–")
        st.dataframe(top5_price, use_container_width=True, hide_index=True)
    with col_rec:
        st.markdown("**Top 5 — Transações mais recentes**")
        top5_rec = (
            bdf.dropna(subset=["data_quitacao"]).head(5)
            [["data_quitacao","endereco","valor_declarado","valor_m2","area_construida_adquirida"]]
            .rename(columns={"data_quitacao":"Data","endereco":"Unidade",
                              "valor_declarado":"Valor (R$)","valor_m2":"R$/m²",
                              "area_construida_adquirida":"Área (m²)"})
        )
        top5_rec["Data"]       = top5_rec["Data"].dt.strftime("%d/%m/%Y")
        top5_rec["Valor (R$)"] = top5_rec["Valor (R$)"].apply(lambda v: f"R$ {v:,.0f}" if pd.notna(v) else "–")
        top5_rec["R$/m²"]      = top5_rec["R$/m²"].apply(lambda v: f"R$ {v:,.0f}" if pd.notna(v) else "–")
        top5_rec["Área (m²)"]  = top5_rec["Área (m²)"].apply(lambda v: f"{v:.1f}" if pd.notna(v) else "–")
        st.dataframe(top5_rec, use_container_width=True, hide_index=True)

    # Full filterable table
    st.markdown("**Tabela completa**")
    fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 2])
    years = sorted(bdf["ano"].dropna().astype(int).unique().tolist())
    year_range = fc1.select_slider(
        "Ano", options=years,
        value=(years[0], years[-1]) if len(years) >= 2 else (years[0], years[0]),
    ) if years else None
    tipos = ["Todos"] + sorted(bdf["descricao_tipo_ocupacao_unidade"].dropna().unique().tolist())
    tipo_sel = fc2.selectbox("Tipo de ocupação", tipos)
    andares = ["Todos"] + sorted(bdf["andar"].dropna().astype(str).unique().tolist())
    andar_sel = fc3.selectbox("Andar", andares)
    blocos = ["Todos"] + sorted(bdf["bloco"].dropna().astype(str).unique().tolist())
    bloco_sel = fc4.selectbox("Bloco", blocos)

    fdf = bdf.copy()
    if year_range:
        fdf = fdf[fdf["ano"].between(year_range[0], year_range[1])]
    if tipo_sel != "Todos":
        fdf = fdf[fdf["descricao_tipo_ocupacao_unidade"] == tipo_sel]
    if andar_sel != "Todos":
        fdf = fdf[fdf["andar"].astype(str) == andar_sel]
    if bloco_sel != "Todos":
        fdf = fdf[fdf["bloco"].astype(str) == bloco_sel]

    table_cols = {
        "data_quitacao": "Data", "endereco": "Endereço completo",
        "valor_declarado": "Valor declarado (R$)", "valor_base_calculo": "Base cálculo (R$)",
        "valor_m2": "R$/m²", "area_construida_adquirida": "Área (m²)",
        "andar": "Andar", "bloco": "Bloco",
        "descricao_tipo_ocupacao_unidade": "Tipo", "padrao_acabamento_unidade": "Padrão",
        "revendido": "Revendido",
    }
    display_df = fdf[[c for c in table_cols if c in fdf.columns]].rename(columns=table_cols)
    st.caption(f"{len(fdf)} transações de {n_total} total")
    st.dataframe(
        display_df, use_container_width=True, hide_index=True,
        column_config={
            "Data": st.column_config.DateColumn("Data", format="DD/MM/YYYY"),
            "Valor declarado (R$)": st.column_config.NumberColumn("Valor declarado (R$)", format="R$ %.0f"),
            "Base cálculo (R$)": st.column_config.NumberColumn("Base cálculo (R$)", format="R$ %.0f"),
            "R$/m²": st.column_config.NumberColumn("R$/m²", format="R$ %.0f"),
            "Área (m²)": st.column_config.NumberColumn("Área (m²)", format="%.1f m²"),
        },
        height=380,
    )

    # ── 3. Anúncio alvo ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("3. 🎯 Anúncio alvo")
    st.caption(
        "Cadastre o apartamento que você quer avaliar. "
        "Ele será o foco central da análise competitiva."
    )

    if st.session_state.target_ad:
        st.success(f"**Alvo definido:** {st.session_state.target_ad['title']}")
        _render_ad_card(st.session_state.target_ad, idx=0)
        if st.button("✏️ Substituir anúncio alvo"):
            st.session_state.target_ad = None
            st.session_state.analysis  = None
            st.rerun()
    else:
        st.info("Nenhum anúncio alvo definido. Preencha abaixo:")
        submitted, ad = _ad_form(
            "form_target",
            "🎯 Definir como anúncio alvo",
            title_placeholder="Ex: Apto 802 – Ed. Patagônia – ZAP",
        )
        if submitted and ad:
            st.session_state.target_ad = ad
            st.session_state.analysis  = None
            st.rerun()

    # ── 4. Anúncios comparativos ───────────────────────────────────────────────
    st.divider()
    st.subheader("4. Anúncios comparativos")
    st.caption(
        "Adicione outros apartamentos do mercado para servir de base na análise competitiva. "
        "Quanto mais comparativos, mais preciso o posicionamento."
    )

    submitted_comp, new_comp = _ad_form(
        "form_comparative",
        "➕ Adicionar comparativo",
        title_placeholder="Ex: Apto 505 – Concorrente – Viva Real",
    )
    if submitted_comp and new_comp:
        st.session_state.comparative_ads.append(new_comp)
        st.session_state.analysis = None
        st.rerun()

    if st.session_state.comparative_ads:
        st.write(f"**{len(st.session_state.comparative_ads)} comparativo(s) adicionado(s):**")
        for i, ad in enumerate(st.session_state.comparative_ads):
            _render_ad_card(ad, idx=i, collection_key="comparative_ads")
    else:
        st.caption("Nenhum comparativo ainda.")

    # ── 5. Anúncios online ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("5. Busca online de mercado")
    st.caption("Busca adicional via IA + DuckDuckGo para enriquecer o contexto de mercado.")

    col_btn, col_info = st.columns([2, 5])
    with col_btn:
        if st.button(
            f"🔍 Buscar online com {provider}",
            type="secondary",
            disabled=not api_key,
            help=f"Requer API key {provider}" if not api_key else f"DuckDuckGo + {model}",
        ):
            with st.spinner(f"{provider} ({model}) buscando anúncios online..."):
                listings, summary = search_listings(bkey, provider, api_key, model)
                st.session_state.online_listings = listings
                st.session_state.market_summary  = summary
                st.session_state.search_done     = True
            st.rerun()
    with col_info:
        if not api_key:
            st.info(f"Configure a API Key do {provider} no painel lateral.")

    if st.session_state.search_done:
        if st.session_state.market_summary:
            st.info(f"**Resumo de mercado:** {st.session_state.market_summary}")
        listings = st.session_state.online_listings
        if listings:
            st.write(f"**{len(listings)} anúncio(s) encontrado(s) online:**")
            for i, l in enumerate(listings):
                with st.container(border=True):
                    ca, cb, cc, cd = st.columns([4, 2, 2, 2])
                    ca.write(f"**{l.get('title', f'Anúncio {i+1}')}**")
                    price, area, pm2 = l.get("price"), l.get("area"), l.get("price_m2")
                    cb.metric("Preço", f"R$ {price:,.0f}" if price else "–")
                    cc.metric("Área", f"{area:.0f} m²" if area else "–")
                    cd.metric("R$/m²", f"R$ {pm2:,.0f}" if pm2 else "–")
                    if l.get("url"):
                        st.markdown(f"[🔗 Ver anúncio]({l['url']})")
                    if l.get("description"):
                        st.caption(l["description"])
        else:
            st.warning("Nenhum anúncio estruturado retornado pela busca online.")

    # ── Gráfico comparativo ────────────────────────────────────────────────────
    if (st.session_state.target_ad or st.session_state.comparative_ads
            or st.session_state.online_listings):
        st.divider()
        st.plotly_chart(
            build_comparison_chart(
                bdf,
                st.session_state.target_ad,
                st.session_state.comparative_ads,
                st.session_state.online_listings,
            ),
            use_container_width=True,
        )

    # ── 6. Análise competitiva ─────────────────────────────────────────────────
    st.divider()
    st.subheader("6. 🤖 Análise competitiva")

    target = st.session_state.target_ad
    comps  = st.session_state.comparative_ads

    if not target:
        st.warning("Defina o **anúncio alvo** na seção 3 para habilitar a análise.")
    else:
        n_photos_total = len(target.get("photos") or []) + sum(
            len(a.get("photos") or []) for a in comps
        )
        col_meta = st.columns(4)
        col_meta[0].metric("Alvo", target["title"][:30])
        col_meta[1].metric("Comparativos", len(comps))
        col_meta[2].metric("Anúncios online", len(st.session_state.online_listings))
        col_meta[3].metric("Fotos para análise visual", n_photos_total)

        if not api_key:
            st.info(f"Configure a API Key do {provider} no painel lateral para gerar a análise.")
        else:
            if st.button(
                f"🤖 Gerar análise competitiva com {provider} ({model})",
                type="primary",
            ):
                with st.spinner(f"{provider} ({model}) analisando posicionamento competitivo..."):
                    analysis = generate_competitive_analysis(
                        bdf, target, comps,
                        st.session_state.online_listings,
                        provider, api_key, model,
                    )
                    st.session_state.analysis = analysis
                st.rerun()

    if st.session_state.analysis:
        with st.container(border=True):
            st.markdown(st.session_state.analysis)
        col_dl, col_reset = st.columns([3, 1])
        col_dl.download_button(
            "⬇️ Baixar análise (.md)",
            data=st.session_state.analysis,
            file_name=f"analise_{bkey.replace(' ', '_')[:30]}.md",
            mime="text/markdown",
        )
        if col_reset.button("↺ Regenerar"):
            st.session_state.analysis = None
            st.rerun()


if __name__ == "__main__":
    main()
