from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Optional

CLASSES_ORDER = ["G", "F", "E", "D", "C", "B", "A", "A+"]

@dataclass
class PMADParams:
    vida_util_estrutura_anos: int = 60
    quota_estrutura_no_preco: float = 0.70
    classe_ref: str = "C"
    premio_por_salto_classe: float = 0.075

def compute_pma_d(
    preco_mercado: float,
    area_m2: float,
    ano_construcao: Optional[int] = None,
    classe_energetica: Optional[str] = None,
    capex_manutencao_previsto: float = 0.0,
    wacc: float = 0.05,
    horizonte_capex_anos: int = 10,
    params: PMADParams = PMADParams(),
):
    if area_m2 <= 0:
        raise ValueError("area_m2 must be positive")

    current_year = date.today().year
    idade = None
    depreciacao_pct = 0.0
    ajuste_depreciacao = 0.0
    if ano_construcao:
        idade = max(0, current_year - int(ano_construcao))
        depreciacao_pct = max(0.0, min(idade / params.vida_util_estrutura_anos, 0.9))
        valor_estrutura = preco_mercado * params.quota_estrutura_no_preco
        ajuste_depreciacao = valor_estrutura * depreciacao_pct
    else:
        depreciacao_pct = None

    ajuste_energia = 0.0
    delta_passos = None
    if classe_energetica and classe_energetica in CLASSES_ORDER:
        delta_passos = CLASSES_ORDER.index(classe_energetica) - CLASSES_ORDER.index(params.classe_ref)
        ajuste_energia = preco_mercado * (delta_passos * params.premio_por_salto_classe)

    vp_capex = 0.0
    if capex_manutencao_previsto:
        vp_capex = capex_manutencao_previsto / ((1 + wacc) ** horizonte_capex_anos)

    pma_d = preco_mercado - ajuste_depreciacao + ajuste_energia - vp_capex
    pma_d_m2 = pma_d / area_m2

    desconto_abs = preco_mercado - pma_d
    desconto_pct = desconto_abs / preco_mercado if preco_mercado else 0.0

    if pma_d > preco_mercado * 1.03:
        badge = "Subavaliado"
    elif pma_d < preco_mercado * 0.97:
        badge = "Sobreavaliado"
    else:
        badge = "Justo"

    return {
        "pma_d": pma_d,
        "pma_d_m2": pma_d_m2,
        "desconto_abs": desconto_abs,
        "desconto_pct": desconto_pct,
        "ajuste_depreciacao": ajuste_depreciacao,
        "ajuste_energia": ajuste_energia,
        "vp_capex": vp_capex,
        "idade": idade,
        "depreciacao_pct": depreciacao_pct,
        "badge": badge,
        "delta_passos": delta_passos,
    }
