"""
CKAN Data Layer — Prefeitura de Belo Horizonte ITBI API
Fetches live ITBI transaction data from dados.pbh.gov.br

Resources are discovered dynamically from the CKAN package metadata
(cached for 24 h) so new monthly releases are picked up automatically.
Field-name schema variants across resources are detected per-resource.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

_CKAN_BASE      = "https://dados.pbh.gov.br/api/3/action"
CKAN_API_BASE   = f"{_CKAN_BASE}/datastore_search"
_PACKAGE_ID     = "itbi-relatorios"
_HEADERS        = {"User-Agent": "ITBI-BH-App/1.0"}
_MAX_WORKERS    = 10
_REQUEST_TIMEOUT = 20
_FETCH_LIMIT    = 32000   # max records per resource per query
_RESOURCE_TTL   = 86_400  # re-discover resources every 24 h

# ─── Resource discovery ────────────────────────────────────────────────────────
# Instead of a hardcoded list, we ask CKAN which CSV resources exist and
# detect the field-name schema of each one.  Results are cached in memory
# for _RESOURCE_TTL seconds so new monthly files are picked up automatically.

_resource_cache: tuple[float, list[tuple[str, str, str, str]]] | None = None


def _detect_schema(resource_id: str) -> tuple[str, str, str] | None:
    """
    Call datastore_search with limit=0 to read field names, then return
    (addr_field, date_field, ano_construcao_field).
    Returns None if the resource is unavailable.
    """
    params = {"resource_id": resource_id, "limit": 0}
    url = CKAN_API_BASE + "?" + urlencode(params)
    req = Request(url, headers=_HEADERS)
    try:
        with urlopen(req, timeout=_REQUEST_TIMEOUT) as f:
            data = json.loads(f.read())
        fields = {fld["id"] for fld in data["result"]["fields"]}

        addr = "Endereco Completo" if "Endereco Completo" in fields else "Endereco"
        date = "Data Quitacao Transacao" if "Data Quitacao Transacao" in fields else "Data Quitacao"
        ano  = "Ano de Construcao Unidade" if "Ano de Construcao Unidade" in fields else "Ano de Construcao (Unidade)"
        return addr, date, ano
    except Exception:
        return None


def _fetch_resource_list() -> list[tuple[str, str, str, str]]:
    """
    Query package_show for the ITBI dataset, filter active CSV resources,
    and detect the schema of each one in parallel.
    Returns list of (resource_id, addr_field, date_field, ano_field).
    """
    url = f"{_CKAN_BASE}/package_show?id={_PACKAGE_ID}"
    req = Request(url, headers=_HEADERS)
    with urlopen(req, timeout=_REQUEST_TIMEOUT) as f:
        pkg = json.loads(f.read())

    csv_ids = [
        r["id"]
        for r in pkg["result"]["resources"]
        if r.get("format", "").upper() == "CSV" and r.get("datastore_active", False)
    ]

    result: list[tuple[str, str, str, str]] = []
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        future_to_id = {executor.submit(_detect_schema, rid): rid for rid in csv_ids}
        for future, rid in future_to_id.items():
            schema = future.result()
            if schema:
                result.append((rid, *schema))
    return result


def _get_resources() -> list[tuple[str, str, str, str]]:
    """
    Return the cached resource list, refreshing if older than _RESOURCE_TTL.
    Thread-safe for Streamlit's single-threaded execution model.
    """
    global _resource_cache
    now = time.monotonic()
    if _resource_cache is None or (now - _resource_cache[0]) >= _RESOURCE_TTL:
        _resource_cache = (now, _fetch_resource_list())
    return _resource_cache[1]


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
    """Parse 'DD/MM/YYYY' → pd.Timestamp (None on failure)"""
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
    padrao = str(
        rec.get("Padrao Acabamento (Unidade)") or rec.get("Padrao Acabamento Unidade") or ""
    ).strip()
    tipo_ocup = str(
        rec.get("Descricao Tipo Ocupacao (Unidade)") or rec.get("Descricao Tipo Ocupacao Unidade") or ""
    ).strip()

    valor_m2     = (v_decl / area_c) if (v_decl and area_c and area_c > 0) else None
    building_key = endereco.split(" - ")[0].strip() if " - " in endereco else endereco

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
        "ano":                             int(dt.year)         if dt is not None else None,
        "mes":                             int(dt.month)        if dt is not None else None,
        "ano_mes":                         dt.strftime("%Y-%m") if dt is not None else None,
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
    Query all active resources in parallel for addresses containing `query`.
    Resource list is refreshed from CKAN every 24 h automatically.
    Returns a list of normalized record dicts.
    """
    resources = _get_resources()
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        futures = [
            executor.submit(_fetch_one, rid, af, df, ano_f, query)
            for rid, af, df, ano_f in resources
        ]
        results: list[dict] = []
        for future in as_completed(futures):
            results.extend(future.result())
    return results


def search_building_names(query: str) -> list[str]:
    """
    Return sorted unique building names (address up to first ' - ')
    from all active resources where the address contains `query`.
    Requires at least 3 characters.
    """
    if len(query.strip()) < 3:
        return []
    records = fetch_all_matching(query.strip().upper())
    names: set[str] = {r["building_key"] for r in records if r.get("building_key")}
    return sorted(names)


def get_building_df(building_key: str) -> pd.DataFrame:
    """
    Fetch all ITBI records for `building_key` from every active resource.
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
