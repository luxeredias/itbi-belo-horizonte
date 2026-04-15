"""
CKAN Data Layer — Prefeitura de Belo Horizonte ITBI API
Fetches live ITBI transaction data from dados.pbh.gov.br

All 23 resources (01/2008–current) are queried in parallel.
Three field-name schema variants are handled transparently.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

CKAN_API_BASE = "https://dados.pbh.gov.br/api/3/action/datastore_search"
_HEADERS = {"User-Agent": "ITBI-BH-App/1.0"}
_MAX_WORKERS = 10
_REQUEST_TIMEOUT = 20
_FETCH_LIMIT = 32000  # max records per resource per query

# ─── Resource registry ─────────────────────────────────────────────────────────
# Columns: (resource_id, addr_field, date_field, ano_construcao_field)
#
# Schema A — historical + Jun 2024:
#   addr="Endereco"  date="Data Quitacao Transacao"  ano="Ano de Construcao Unidade"
# Schema B — Jul 2024 – Feb 2026:
#   addr="Endereco"  date="Data Quitacao"             ano="Ano de Construcao (Unidade)"
# Schema C — Mar 2026:
#   addr="Endereco Completo"  date="Data Quitacao"    ano="Ano de Construcao (Unidade)"

RESOURCES: list[tuple[str, str, str, str]] = [
    # resource_id                              addr_field          date_field                   ano_field
    ("7f8955aa-0b30-4157-bbc2-7dd444941728", "Endereco",          "Data Quitacao Transacao",   "Ano de Construcao Unidade"),        # 01/2008–05/2024
    ("b21f9785-6b4c-43a1-bd66-dbf89e63efe7", "Endereco",          "Data Quitacao Transacao",   "Ano de Construcao Unidade"),        # Jun 2024
    ("1520a5dc-858a-4eca-89bd-54964c273ce7", "Endereco",          "Data Quitacao",             "Ano de Construcao (Unidade)"),      # Jul 2024
    ("aa948bd0-9554-4e33-87f7-eb61ef0d2903", "Endereco",          "Data Quitacao",             "Ano de Construcao (Unidade)"),      # Aug 2024
    ("f582a331-9608-49a2-be3b-a5a5aa19ecff", "Endereco",          "Data Quitacao",             "Ano de Construcao (Unidade)"),      # Sep 2024
    ("f5c13859-c3ee-4e3e-9c34-c6215f49a4a5", "Endereco",          "Data Quitacao",             "Ano de Construcao (Unidade)"),      # Oct 2024
    ("97db6e95-141a-4402-aa60-50dd06ae9340", "Endereco",          "Data Quitacao",             "Ano de Construcao (Unidade)"),      # Nov 2024
    ("af6d7c58-e1d2-4fab-8d9a-487dddc30a20", "Endereco",          "Data Quitacao",             "Ano de Construcao (Unidade)"),      # Dec 2024
    ("78c33612-ebf5-4a25-af35-d3065f056579", "Endereco",          "Data Quitacao",             "Ano de Construcao (Unidade)"),      # Jan 2025
    ("23e019f9-6537-4561-9935-ca7d19ae3b6b", "Endereco",          "Data Quitacao",             "Ano de Construcao (Unidade)"),      # Feb 2025
    ("df8c42ee-166f-4e5d-9eb3-69fc00f3ae57", "Endereco",          "Data Quitacao",             "Ano de Construcao (Unidade)"),      # Mar 2025
    ("dad2d5af-d9c2-4e09-9766-7f01387fbd9a", "Endereco",          "Data Quitacao",             "Ano de Construcao (Unidade)"),      # Apr 2025
    ("0e353102-585d-4dfb-ab15-84550b550c61", "Endereco",          "Data Quitacao",             "Ano de Construcao (Unidade)"),      # May 2025
    ("6be6874d-6e20-4d68-a904-2351fc0f31f4", "Endereco",          "Data Quitacao",             "Ano de Construcao (Unidade)"),      # Jun 2025
    ("9f8b1538-5f8b-469c-8728-f4902b28456e", "Endereco",          "Data Quitacao",             "Ano de Construcao (Unidade)"),      # Jul 2025
    ("f4e60a70-3bd8-41dc-b031-42b5f3d1671a", "Endereco",          "Data Quitacao",             "Ano de Construcao (Unidade)"),      # Aug 2025
    ("cbdea14d-77de-4c32-a7db-3e327e09257c", "Endereco",          "Data Quitacao",             "Ano de Construcao (Unidade)"),      # Sep 2025
    ("fbd1fe60-f838-4392-92c2-98cc09dfc81c", "Endereco",          "Data Quitacao",             "Ano de Construcao (Unidade)"),      # Oct 2025
    ("9bd075e2-2abe-42d7-a1e2-b68a18245172", "Endereco",          "Data Quitacao",             "Ano de Construcao (Unidade)"),      # Nov 2025
    ("c3dd25fd-ac34-4f3f-afcc-21589d64ce8a", "Endereco",          "Data Quitacao",             "Ano de Construcao (Unidade)"),      # Dec 2025
    ("2deb4632-f226-40a7-b8ac-89c6d9f8aff9", "Endereco",          "Data Quitacao",             "Ano de Construcao (Unidade)"),      # Jan 2026
    ("453b3f6d-ec7f-415c-af5e-c23d1f488e2a", "Endereco",          "Data Quitacao",             "Ano de Construcao (Unidade)"),      # Feb 2026
    ("0773a6b9-b107-4692-9b85-760221ec3abb", "Endereco Completo", "Data Quitacao",             "Ano de Construcao (Unidade)"),      # Mar 2026
]

# ─── Parsing helpers ───────────────────────────────────────────────────────────

def _parse_brl(s: Any) -> float | None:
    """Parse Brazilian numeric string '1.900.000,00' → 1_900_000.0"""
    if s is None or s == "":
        return None
    try:
        return float(str(s).replace(".", "").replace(",", "."))
    except (ValueError, AttributeError):
        return None


def _parse_date(s: Any) -> pd.Timestamp | None:
    """Parse 'DD/MM/YYYY' → pd.Timestamp (NaT on failure)"""
    if not s:
        return None
    ts = pd.to_datetime(str(s), dayfirst=True, errors="coerce")
    return None if pd.isna(ts) else ts


# ─── Record normalization ──────────────────────────────────────────────────────

def _normalize(rec: dict, addr_field: str, date_field: str, ano_field: str) -> dict:
    """Map a raw CKAN record to the internal column schema."""
    endereco = str(rec.get(addr_field) or "").strip()

    area_c = _parse_brl(rec.get("Area Construida Adquirida"))
    area_u = _parse_brl(
        rec.get("Area Adquirida (Unidades Somadas)")
        or rec.get("Area Adquirida Unidades Somadas")
    )
    v_decl = _parse_brl(rec.get("Valor Declarado"))
    v_base = _parse_brl(rec.get("Valor Base Calculo"))
    area_t = _parse_brl(rec.get("Area Terreno Total"))
    frac   = _parse_brl(rec.get("Fracao Ideal Adquirida"))

    dt = _parse_date(rec.get(date_field))

    ano_c_raw = (
        rec.get(ano_field)
        or rec.get("Ano de Construcao (Unidade)")
        or rec.get("Ano de Construcao Unidade")
    )
    padrao = (
        str(rec.get("Padrao Acabamento (Unidade)") or rec.get("Padrao Acabamento Unidade") or "").strip()
    )
    tipo_ocup = (
        str(rec.get("Descricao Tipo Ocupacao (Unidade)") or rec.get("Descricao Tipo Ocupacao Unidade") or "").strip()
    )

    valor_m2      = (v_decl / area_c) if (v_decl and area_c and area_c > 0) else None
    building_key  = endereco.split(" - ")[0].strip() if " - " in endereco else endereco

    return {
        "id":                              rec.get("_id"),
        "endereco":                        endereco,
        "bairro":                          str(rec.get("Bairro") or "").strip(),
        "ano_construcao_unidade":          _parse_brl(ano_c_raw),
        "area_terreno_total":              area_t,
        "area_construida_adquirida":       area_c,
        "area_adquirida_unidades_somadas": area_u,
        "padrao_acabamento_unidade":       padrao,
        "fracao_ideal_adquirida":          frac,
        "tipo_construtivo_preponderante":  str(rec.get("Tipo Construtivo Preponderante") or "").strip(),
        "descricao_tipo_ocupacao_unidade": tipo_ocup,
        "valor_declarado":                 v_decl,
        "valor_base_calculo":              v_base,
        "zona_uso":                        str(rec.get("Zona Uso ITBI") or "").strip(),
        "data_quitacao":                   dt,
        "ano":                             int(dt.year)          if dt is not None else None,
        "mes":                             int(dt.month)         if dt is not None else None,
        "ano_mes":                         dt.strftime("%Y-%m")  if dt is not None else None,
        "valor_m2":                        valor_m2,
        # Not available via API — kept for schema compatibility
        "andar":                           None,
        "unidade":                         None,
        "bloco":                           None,
        "apt_key":                         None,
        "revendido":                       None,
        "building_key":                    building_key,
        "building_key_norm":               building_key.upper(),
    }


# ─── HTTP helper ───────────────────────────────────────────────────────────────

def _fetch_one(resource_id: str, addr_field: str, date_field: str, ano_field: str, query: str) -> list[dict]:
    """Fetch + normalize records from one resource matching `query`. Returns [] on any error."""
    params = {
        "resource_id": resource_id,
        "q":           json.dumps({addr_field: query}),
        "limit":       _FETCH_LIMIT,
    }
    url = CKAN_API_BASE + "?" + urlencode(params)
    req = Request(url, headers=_HEADERS)
    try:
        with urlopen(req, timeout=_REQUEST_TIMEOUT) as f:
            data = json.loads(f.read())
        return [_normalize(r, addr_field, date_field, ano_field) for r in data["result"]["records"]]
    except Exception:
        return []


# ─── Public API ────────────────────────────────────────────────────────────────

def fetch_all_matching(query: str) -> list[dict]:
    """
    Query all resources in parallel for addresses containing `query`.
    Returns a list of normalized record dicts.
    """
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        futures = [
            executor.submit(_fetch_one, rid, af, df, ano_f, query)
            for rid, af, df, ano_f in RESOURCES
        ]
        results: list[dict] = []
        for future in as_completed(futures):
            results.extend(future.result())
    return results


def search_building_names(query: str) -> list[str]:
    """
    Return sorted unique building names (address up to first ' - ')
    from all resources where the address contains `query`.
    Requires at least 3 characters.
    """
    if len(query.strip()) < 3:
        return []
    records = fetch_all_matching(query.strip().upper())
    names: set[str] = {r["building_key"] for r in records if r.get("building_key")}
    return sorted(names)


def get_building_df(building_key: str) -> pd.DataFrame:
    """
    Fetch all ITBI records for `building_key` from every resource.
    Client-side filters to exact building prefix to prevent false positives
    (e.g. 'PATAGONIA 200' matching 'PATAGONIA 2001').
    Returns a DataFrame sorted by data_quitacao descending.
    """
    records = fetch_all_matching(building_key)
    prefix = building_key.upper() + " - "
    exact  = building_key.upper()
    filtered = [
        r for r in records
        if r["building_key_norm"] == exact
        or r["endereco"].upper().startswith(prefix)
    ]
    if not filtered:
        return pd.DataFrame()
    df = pd.DataFrame(filtered)
    df["data_quitacao"] = pd.to_datetime(df["data_quitacao"], errors="coerce")
    return df.sort_values("data_quitacao", ascending=False).reset_index(drop=True)
