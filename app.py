# appv3.py — Preco justo: Comprar vs Alugar (Portugal)
# Solver: PuLP (MILP)
# Regime Jovem 2025, IMT/IS, media de aluguel com crescimento,
# P* via igualdade de custos, valorizacao do imovel aplicada APOS P*
# Bloco "Sobrepreco vs. valorizacao" com rótulos e condições integrados.
# Requisitos: pip install streamlit numpy pandas pulp

from __future__ import annotations
import math
from dataclasses import dataclass
import streamlit as st
import numpy as np
import pandas as pd

# ========= MILP solver =========
try:
    import pulp
except Exception:
    st.error("Dependencia ausente: instale com 'pip install pulp' e reinicie o app.")
    st.stop()

# ========= IMT (2025) – aproximado =========
@dataclass
class IMTBracket:
    lower: float
    upper: float     # exclusivo (math.inf para ultimo)
    rate: float      # taxa marginal ou unica (flat=True)
    abat: float      # parcela a abater se flat=False
    flat: bool = False

IMT_TABLES = {
    ("continente", "hpp"): [
        IMTBracket(0, 104_261, 0.00, 0.00, False),
        IMTBracket(104_261, 142_618, 0.02, 2_085.22, False),
        IMTBracket(142_618, 194_458, 0.05, 6_363.76, False),
        IMTBracket(194_458, 324_058, 0.07, 10_252.92, False),
        IMTBracket(324_058, 648_022, 0.08, 13_493.50, False),
        IMTBracket(648_022, 1_128_287, 0.06, 0.00, True),
        IMTBracket(1_128_287, math.inf, 0.075, 0.00, True),
    ],
    ("continente", "habitacao"): [
        IMTBracket(0, 104_261, 0.01, 0.00, False),
        IMTBracket(104_261, 142_618, 0.02, 1_042.61, False),
        IMTBracket(142_618, 194_458, 0.05, 5_321.15, False),
        IMTBracket(194_458, 324_058, 0.07, 9_210.31, False),
        IMTBracket(324_058, 621_501, 0.08, 12_450.89, False),
        IMTBracket(621_501, 1_128_287, 0.06, 0.00, True),
        IMTBracket(1_128_287, math.inf, 0.075, 0.00, True),
    ],
    ("ilhas", "hpp"): [
        IMTBracket(0, 130_326, 0.00, 0.00, False),
        IMTBracket(130_326, 178_273, 0.02, 2_606.52, False),
        IMTBracket(178_273, 243_073, 0.05, 7_954.71, False),  # corrigido separador
        IMTBracket(243_073, 405_073, 0.07, 12_816.17, False),
        IMTBracket(405_073, 810_028, 0.08, 16_866.90, False),
        IMTBracket(810_028, 1_410_359, 0.06, 0.00, True),
        IMTBracket(1_410_359, math.inf, 0.075, 0.00, True),
    ],
}

def compute_imt(price: float, regime: str, territorio: str) -> float:
    br = IMT_TABLES[(territorio, regime)]
    for b in br:
        if price <= b.upper:
            return price * b.rate if b.flat else max(0.0, price * b.rate - b.abat)
    return price * br[-1].rate

def imposto_selo_credito(loan_amount: float, prazo_opt: str) -> float:
    if loan_amount <= 0:
        return 0.0
    if prazo_opt == ">=5y":
        return 0.006 * loan_amount
    elif prazo_opt == "1-5y":
        return 0.005 * loan_amount
    else:
        return 0.0004 * 12 * loan_amount

def monthly_commute_cost(d_you: float, d_sp: float, days_m: int, cost_km: float) -> float:
    return ((d_you + d_sp) * 2.0) * days_m * cost_km

# ========= UI =========
st.set_page_config(page_title="Preco Justo: Comprar vs Alugar", layout="wide")
st.title("Preco justo para comprar vs alugar — custos irrecuperaveis")
st.subheader("Baseado nos conceitos dos seguintes vídeos:")
c1,c2,c3= st.columns(3)
with c1:
    st.link_button("Todos os Custos Associados a compra de casa","https://www.youtube.com/watch?v=twoIYmxGiOE",icon="▶️")
with c2:
    st.link_button("Alugar vs Comprar: Como decidir?","https://www.youtube.com/watch?v=q9Golcxjpi8",icon="▶️")
with c3:
    st.link_button("Alugar vs comprar: A regra dos 5%","https://www.youtube.com/watch?v=Uwl3-jBNEd4",icon="▶️")
with st.sidebar:
    st.header("1) Renda de referencia")
    area = st.number_input("Area util desejada (m2)", 10.0, value=120.0, step=1.0)
    rent_per_m2 = st.number_input("Renda media na regiao (EUR/m2/mes)", 1.0, value=8.5, step=0.1)
    renter_ins_month = st.number_input("Seguro inquilino (EUR/mes)", 0.0, value=8.0, step=1.0)
    rent_utils_month = st.number_input("Utilidades no imovel arrendado (EUR/mes)", 0.0, value=90.0, step=5.0)
    rent_app_rate_pct = st.number_input(
        "Valorizacao anual esperada da renda (%)", -10.0, value=0.0, step=0.25,
        help="Crescimento composto mes a mes sobre a renda base; usamos a media no horizonte."
    )

    st.divider()
    st.header("2) Preco de compra de mercado")
    buy_per_m2_market = st.number_input("Preco medio de compra (EUR/m2)", 100.0, value=2000.0, step=50.0)

    st.divider()
    st.header("3) Deslocamento mensal")
    days = st.number_input("Dias de commuting/mes", 0, value=22, step=1)
    cost_km = st.number_input("Custo por km (EUR/km)", 0.0, value=0.30, step=0.01)
    c1, c2 = st.columns(2)
    with c1:
        d_rent_you = st.number_input("Distancia sua (renda) — km (so ida)", 0.0, value=3.5, step=0.1)
        d_buy_you  = st.number_input("Distancia sua (compra) — km (so ida)", 0.0, value=3.5, step=0.1)
    with c2:
        d_rent_sp  = st.number_input("Distancia conjuge (renda) — km (so ida)", 0.0, value=15.0, step=0.1)
        d_buy_sp   = st.number_input("Distancia conjuge (compra) — km (so ida)", 0.0, value=15.0, step=0.1)

    st.divider()
    st.header("4) Classe energetica & utilidades (compra)")
    energy_class = st.selectbox("Classe energetica", ["A+","A","B","C","D","E","F","G"], index=5)
    util_mode = st.radio("Estimativa de utilidades (compra)", ["Valor fixo (EUR/mes)", "A partir da classe (% vs aluguel)"])
    if util_mode == "Valor fixo (EUR/mes)":
        buy_utils_month = st.number_input("Utilidades no imovel comprado (EUR/mes)", 0.0, value=90.0, step=5.0)
    else:
        factors = {"A+": -0.30, "A": -0.25, "B": -0.18, "C": -0.10, "D": -0.05, "E": 0.00, "F": 0.08, "G": 0.15}
        delta = factors.get(energy_class, 0.0)
        buy_utils_month = rent_utils_month * (1.0 + delta)
        st.caption(f"Heuristica: {int(delta*100)}% vs utilidades do arrendamento.")

    st.divider()
    st.header("5) Parametros de propriedade")
    horizon = st.number_input("Horizonte de permanencia (anos)", 1.0, value=6.0, step=1.0)

    st.subheader("Custo de capital (sem valorizacao do imovel)")
    auto_cap = st.checkbox("Calcular automaticamente via LTV, taxa do credito e retorno alternativo", value=True)
    if auto_cap:
        loan_rate_pct = st.number_input("Taxa do credito (nominal, %)", 0.0, value=4.0, step=0.1)
        opp_return_pct = st.number_input("Retorno alternativo esperado (%)", 0.0, value=5.0, step=0.1)
    else:
        cap_rate_pct = st.number_input("Custo de capital anual (oportunidade/juros) — %", 0.0, value=3.0, step=0.25)
        loan_rate_pct = st.number_input("Taxa do credito (nominal, %) — opcional", 0.0, value=4.0, step=0.1)
        opp_return_pct = st.number_input("Retorno alternativo (%) — opcional", 0.0, value=5.0, step=0.1)

    maint_rate_pct = st.number_input("Manutencao anual — % do preco", 0.0, value=1.5, step=0.25)
    maint_rate = maint_rate_pct / 100.0
    condo_month = st.number_input("Condominio (EUR/mes)", 0.0, value=35.0, step=5.0)
    ins_month = st.number_input("Seguros (vida + multirriscos) (EUR/mes)", 0.0, value=45.0, step=5.0)

    st.subheader("IMI")
    imi_rate = st.number_input("Taxa IMI (%)", 0.0, value=0.35, step=0.01) / 100.0
    vpt_mode = st.radio("VPT para calculo do IMI", ["% do preco", "Valor fixo (EUR)"])
    if vpt_mode == "% do preco":
        vpt_ratio = st.number_input("VPT como % do preco", 0.10, value=0.70, step=0.05)
        vpt_value = None
    else:
        vpt_value = st.number_input("VPT (EUR/ano-base)", 0.0, value=0.0, step=100.0)
        vpt_ratio = None

    st.subheader("Custos one-off")
    upfront_registos = st.number_input("Emolumentos de registo (EUR)", 0.0, value=700.0, step=50.0)
    upfront_outros   = st.number_input("Outros custos fixos (EUR)", 0.0, value=1_300.0, step=100.0)

    st.subheader("IMT / IS / Regime Jovem (2025)")
    territorio = st.selectbox("Territorio", ["continente", "ilhas"], index=0)
    regime = st.selectbox("Regime IMT base", ["hpp", "habitacao"], index=0)
    youth_mode = st.checkbox("Aplicar Credito Habitacao Jovem (isencao IMT/IS compra e emolumentos; garantia)")
    if youth_mode:
        elig = st.selectbox("Elegibilidade", ["Ambos <=35 (1a habitacao)", "Apenas um <=35"], index=0)
        youth_share = 1.0 if "Ambos" in elig else 0.5
    else:
        youth_share = 0.0
    THRESH = 324_058.0
    ltv = st.slider("LTV do credito (%)", 0, 100, value=90, step=5) / 100.0
    is_prazo_opt = st.selectbox("Prazo do credito p/ IS", [">=5y", "1-5y", "<1y"], index=0)
    sell_rate = st.number_input("Custo esperado na venda (% do preco)", 0.0, value=5.00, step=0.25) / 100.0

    st.subheader("Valorizacao do imovel (aplicada APOS obter P*)")
    prop_app_rate_pct = st.number_input("Valorizacao anual esperada do imovel (%)", -10.0, value=0.0, step=0.25)

# ---- custo de capital efetivo (sem valorizacao do imovel) ----
if auto_cap:
    cap_rate = ltv * (loan_rate_pct/100.0) + (1.0 - ltv) * (opp_return_pct/100.0)
else:
    cap_rate = cap_rate_pct / 100.0

# ========= Preparacao =========
months = max(1, int(round(horizon * 12)))

# Alugar — mês 1 e média no horizonte (renda base cresce; util/seguro/commute fixos)
rent_month = rent_per_m2 * area
commute_rent = monthly_commute_cost(d_rent_you, d_rent_sp, days, cost_km)
commute_buy  = monthly_commute_cost(d_buy_you,  d_buy_sp,  days, cost_km)
rent_total   = rent_month + renter_ins_month + rent_utils_month + commute_rent

rent_app_rate = rent_app_rate_pct / 100.0
if abs(rent_app_rate) < 1e-12:
    rent_avg_rent_component = rent_month
else:
    r_m = (1.0 + rent_app_rate) ** (1.0/12.0)
    rent_avg_rent_component = rent_month * (r_m**months - 1.0) / ((r_m - 1.0) * months)
rent_avg_total = rent_avg_rent_component + renter_ins_month + rent_utils_month + commute_rent

# Preco de mercado
P_market = buy_per_m2_market * area

# ========= Owning total =========
STATE = dict(
    vpt_ratio=(vpt_ratio if vpt_value is None else 0.0),
    vpt_value=(None if vpt_value is None else vpt_value),
    ltv=ltv,
    regime=regime,
    territorio=territorio,
    is_prazo_opt=is_prazo_opt,
    horizon=horizon,
    cap_rate=cap_rate,
    maint_rate=maint_rate,
    imi_rate=imi_rate,
    condo_month=condo_month,
    ins_month=ins_month,
    upfront_registos=upfront_registos,
    upfront_outros=upfront_outros,
    sell_rate=sell_rate,
    buy_utils_month=buy_utils_month,
    commute_buy=commute_buy,
    youth_share=youth_share,
    THRESH=THRESH,
)

def owning_monthly_total(price: float) -> tuple[float, dict]:
    vpt = (STATE['vpt_ratio'] * price) if STATE['vpt_value'] is None else STATE['vpt_value']
    loan = STATE['ltv'] * price
    imt_total  = compute_imt(price, STATE['regime'], STATE['territorio'])
    imt_thresh = compute_imt(min(price, STATE['THRESH']), STATE['regime'], STATE['territorio'])
    imt_due = max(0.0, imt_total - STATE['youth_share'] * imt_thresh)
    q = min(price, STATE['THRESH'])
    is_compra = 0.008 * (price - STATE['youth_share'] * q)
    is_credito = imposto_selo_credito(loan, STATE['is_prazo_opt'])
    registos_eff = STATE['upfront_registos'] * (1.0 - (STATE['youth_share'] if price <= STATE['THRESH'] else 0.0))
    upfront_total = registos_eff + STATE['upfront_outros']

    m = max(1, int(round(STATE['horizon'] * 12)))
    cap_month   = (STATE['cap_rate']   * price) / 12.0
    maint_month = (STATE['maint_rate'] * price) / 12.0
    imi_month   = (STATE['imi_rate']   * vpt)   / 12.0
    condo_ins   = float(STATE['condo_month']) + float(STATE['ins_month'])
    upfront_m   = (upfront_total + imt_due + is_compra + is_credito) / m
    sell_m      = (STATE['sell_rate'] * price) / m

    irrec = cap_month + maint_month + imi_month + condo_ins + upfront_m + sell_m
    total = irrec + STATE['buy_utils_month'] + STATE['commute_buy']
    breakdown = {
        "irrec": irrec,
        "imt": imt_due,
        "is_compra": is_compra,
        "is_credito": is_credito,
        "vpt": vpt,
        "upfront_registos_eff": registos_eff,
        "upfront_outros": STATE['upfront_outros'],
        "utils_buy": STATE['buy_utils_month'],
        "commute_buy": STATE['commute_buy'],
    }
    return total, breakdown

# ========= Solver PuLP =========
def build_imt_piece(prob: pulp.LpProblem, var, name_prefix: str, territorio: str, regime: str, p_max: float):
    brackets = IMT_TABLES[(territorio, regime)]
    y  = [pulp.LpVariable(f"{name_prefix}_y{i}", lowBound=0, upBound=1, cat="Binary") for i in range(len(brackets))]
    Vi = [pulp.LpVariable(f"{name_prefix}_Vi{i}", lowBound=0, upBound=p_max, cat="Continuous") for i in range(len(brackets))]
    prob += pulp.lpSum(y) == 1
    for i, b in enumerate(brackets):
        UB = p_max if math.isinf(b.upper) else b.upper
        prob += Vi[i] <= var
        prob += Vi[i] <= p_max * y[i]
        prob += Vi[i] >= var - p_max * (1 - y[i])
        prob += var >= b.lower * y[i]
        prob += var <= UB * y[i] + p_max * (1 - y[i])
    imt_expr = pulp.lpSum([brackets[i].rate * Vi[i] - brackets[i].abat * y[i] for i in range(len(brackets))])
    return imt_expr

def solve_price_pulp(
    target_monthly_cost: float,
    months: int,
    cap_rate: float,
    maint_rate: float,
    imi_rate: float,
    vpt_ratio: float | None,
    vpt_value_fixed: float | None,
    condo_month: float,
    ins_month: float,
    upfront_registos: float,
    upfront_outros: float,
    territorio: str,
    regime: str,
    is_prazo_opt: str,
    ltv: float,
    sell_rate: float,
    buy_utils_month: float,
    commute_buy_month: float,
    youth_share: float,
    THRESH: float,
    p_min: float = 10_000.0,
    p_max: float = 3_000_000.0,
) -> float:
    # Coef linear + constante (sem valorizacao na reta)
    a = cap_rate/12.0 + maint_rate/12.0 + sell_rate/float(months)
    const = 0.0
    if vpt_value_fixed is not None:
        const += (imi_rate * vpt_value_fixed) / 12.0
    else:
        a += (imi_rate * float(vpt_ratio)) / 12.0
    const += float(condo_month) + float(ins_month)
    const += float(buy_utils_month) + float(commute_buy_month)
    const += float(upfront_outros) / float(months)

    if is_prazo_opt == ">=5y":
        credit_is_rate = 0.006
    elif is_prazo_opt == "1-5y":
        credit_is_rate = 0.005
    else:
        credit_is_rate = 0.0004 * 12.0
    a += (credit_is_rate * float(ltv)) / float(months)
    a += 0.008 / float(months)  # IS compra (parte linear em P)

    prob = pulp.LpProblem("preco_justo", pulp.LpMinimize)
    P = pulp.LpVariable("P", lowBound=p_min, upBound=p_max, cat="Continuous")

    Q = pulp.LpVariable("Q", lowBound=0, upBound=THRESH, cat="Continuous")
    w = pulp.LpVariable("w", lowBound=0, upBound=1, cat="Binary")  # w=1 se P<=THRESH
    M = p_max
    prob += Q <= P
    prob += Q <= THRESH
    prob += Q >= P - M * (1 - w)
    prob += Q >= THRESH - M * w

    imt_total = build_imt_piece(prob, P, "imt_total", territorio, regime, p_max)
    imt_Q     = build_imt_piece(prob, Q, "imt_q",     territorio, regime, p_max)
    imt_net   = imt_total - youth_share * imt_Q

    registos_eff = (upfront_registos / float(months)) - (youth_share * upfront_registos / float(months)) * w

    lhs = a * P + (imt_net / float(months)) + const + registos_eff - (0.008 * youth_share / float(months)) * Q
    prob += lhs == target_monthly_cost
    prob += P

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"PuLP nao encontrou solucao otima: {pulp.LpStatus[status]}")
    return float(pulp.value(P))

# ========= Resolver (P*) =========
P_star = solve_price_pulp(
    target_monthly_cost=rent_avg_total,   # media do custo de alugar
    months=months,
    cap_rate=cap_rate,
    maint_rate=maint_rate,
    imi_rate=imi_rate,
    vpt_ratio=(vpt_ratio if vpt_value is None else None),
    vpt_value_fixed=(None if vpt_value is None else vpt_value),
    condo_month=condo_month,
    ins_month=ins_month,
    upfront_registos=upfront_registos,
    upfront_outros=upfront_outros,
    territorio=territorio,
    regime=regime,
    is_prazo_opt=is_prazo_opt,
    ltv=ltv,
    sell_rate=sell_rate,
    buy_utils_month=buy_utils_month,
    commute_buy_month=commute_buy,
    youth_share=youth_share,
    THRESH=THRESH,
)

own_tot_fair, br_fair = owning_monthly_total(P_star)
own_tot_market, br_market = owning_monthly_total(P_market)

irrec_fair_m = br_fair["irrec"] 
irrec_mkt_m  = br_market["irrec"] 
# ========= Mercado vs Justo (futuro vs futuro) =========
g = prop_app_rate_pct / 100.0
F = (1.0 + g) ** horizon


# ========= Sobrepreco vs. valorizacao (rótulos/condições integrados) =========
deltaP = P_market - P_star  # sobrepreço hoje (pode ser <0)
g = prop_app_rate_pct / 100.0
F = (1.0 + g) ** horizon
# Mais-valia do imóvel na base justa (não deixa o resultado subir só por pagar caro)
asset_gain_base = P_star * (F - 1.0)

# Irrecuperáveis adicionais por ter pago acima do justo
extra_irrec_diff = (irrec_mkt_m - irrec_fair_m) * months
# Se preferir ignorar "barganhas", use:
# extra_irrec_diff = max(0.0, irrec_mkt_m - irrec_fair_m) * months

net_user = asset_gain_base - extra_irrec_diff

# Breakeven (taxa mínima para empatar no teu critério)
breakeven_g_text = "—"
breakeven_T_text = "—"
if P_star > 0:
    g_star = (1.0 + (extra_irrec_diff / P_star)) ** (1.0 / horizon) - 1.0
    breakeven_g_text = f"{g_star*100:.2f}%"
    if g > 0:
        try:
            T_star = math.log(1.0 + (extra_irrec_diff / P_star)) / math.log(1.0 + g)
            breakeven_T_text = f"{T_star:.1f} anos"
        except ValueError:
            pass

# ========= UI principal =========
cL, cR = st.columns([1,1])

with cL:
    st.subheader("Cenario ALUGAR")
    st.metric("Renda base (mes 1)", f"{rent_month:,.0f}".replace(","," "))
    st.metric("Renda media (horizonte)", f"{rent_avg_rent_component:,.0f}".replace(","," "))
    st.metric("Commuting (EUR/mes)", f"{commute_rent:,.0f}".replace(","," "))
    st.metric("Utilidades + seguro (EUR/mes)", f"{(rent_utils_month + renter_ins_month):,.0f}".replace(","," "))
    st.metric("Total (mes 1)", f"{rent_total:,.0f}".replace(","," "))
    st.metric("Total medio (horizonte)", f"{rent_avg_total:,.0f}".replace(","," "))

with cR:
    st.subheader("Preco justo para COMPRAR")
    st.metric("Preco maximo justo HOJE (EUR)", f"{P_star:,.0f}".replace(","," "))
    st.metric("Preco de mercado hoje (EUR)", f"{P_market:,.0f}".replace(","," "))
    st.caption("Definicao: custo mensal medio de possuir = custo mensal medio de alugar (no horizonte).")

    st.write("**Decomposicao ao preco justo (mensal)**")
    cap_m   = (STATE['cap_rate']   * P_star) / 12.0
    maint_m = (STATE['maint_rate'] * P_star) / 12.0
    imi_m   = (STATE['imi_rate']   * br_fair['vpt']) / 12.0
    condo_m = float(STATE['condo_month'])
    ins_m   = float(STATE['ins_month'])
    imtm_m  = br_fair["imt"]        / months
    iscomp_m= br_fair["is_compra"]  / months
    iscred_m= br_fair["is_credito"] / months
    sell_m  = (STATE['sell_rate'] * P_star) / months

    rows = [
        {"Item": "Custo de capital (juros + oportunidade)", "EUR_mes": round(cap_m,2),
         "Como": f"LTV*{loan_rate_pct:.2f}% + (1-LTV)*{opp_return_pct:.2f}%"},
        {"Item": "Manutencao",  "EUR_mes": round(maint_m,2), "Como": f"{maint_rate_pct:.2f}% a.a. x preco / 12"},
        {"Item": "IMI",         "EUR_mes": round(imi_m, 2),  "Como": f"{STATE['imi_rate']*100:.2f}% x VPT (EUR {br_fair['vpt']:.0f}) / 12"},
        {"Item": "Condominio",  "EUR_mes": round(condo_m,2), "Como": "mensal informado"},
        {"Item": "Seguros",     "EUR_mes": round(ins_m,  2), "Como": "mensal informado"},
        {"Item": "IMT (amort.)","EUR_mes": round(imtm_m, 2), "Como": f"EUR {br_fair['imt']:.0f} / {months}"},
        {"Item": "IS compra (amort.)","EUR_mes": round(iscomp_m,2), "Como": f"EUR {br_fair['is_compra']:.0f} / {months}"},
        {"Item": "IS credito (amort.)","EUR_mes": round(iscred_m,2), "Como": f"EUR {br_fair['is_credito']:.0f} / {months}"},
        {"Item": "Custo de venda (amort.)","EUR_mes": round(sell_m,2), "Como": f"{STATE['sell_rate']*100:.2f}% x preco / {months}"},
        {"Item": "Emol. registo (amort.)","EUR_mes": round(br_fair['upfront_registos_eff']/months, 2), "Como": "apos isencao jovem (se aplicavel)"},
        {"Item": "Outros upfront (amort.)","EUR_mes": round(br_fair['upfront_outros']/months, 2), "Como": "custos fixos"},
    ]
    irrec_sum = float(sum(r["EUR_mes"] for r in rows))
    rows.append({"Item": "Subtotal irrecuperavel", "EUR_mes": round(irrec_sum,2), "Como": ""})
    rows.append({"Item": "Utilidades (compra)", "EUR_mes": round(br_fair["utils_buy"],2), "Como": ""})
    rows.append({"Item": "Commuting (compra)",  "EUR_mes": round(br_fair["commute_buy"],2), "Como": ""})
    total_own = float(round(irrec_sum + br_fair["utils_buy"] + br_fair["commute_buy"], 2))
    rows.append({"Item": "Total possuir (~ total alugar no preco justo)", "EUR_mes": total_own, "Como": ""})
    df = pd.DataFrame(rows)[["Item","EUR_mes","Como"]]
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.caption(f"Total ALUGAR (media no horizonte) = EUR {rent_avg_total:.2f}/mes | Possuir - Alugar = EUR {(total_own - rent_avg_total):.2f}/mes (≈0 no preco justo).")

st.divider()

# ---- UI clara (um veredito único) ----
st.subheader("Resultado líquido vs alugar (critério final)")
st.metric("Mais-valia do imóvel (base: preço justo)", f"{asset_gain_base:,.0f}".replace(",", " "))
st.metric("Irrecuperáveis adicionais (mkt − justo, no horizonte)", f"{extra_irrec_diff:,.0f}".replace(",", " "))
st.metric("g* para empatar", breakeven_g_text)
st.metric("T* para empatar", breakeven_T_text)

if net_user >= 0:
    st.success(
        f"GANHO líquido esperado: € {net_user:,.0f} em {months} meses "
        f"(≈ € {net_user/months:,.0f}/mês)".replace(",", " ")
    )
else:
    st.error(
        f"PERDA líquida esperada: € {abs(net_user):,.0f} em {months} meses "
        f"(≈ € {abs(net_user)/months:,.0f}/mês)".replace(",", " ")
    )

st.caption(
    "Definição: Resultado = P*·[(1+g)^T−1] − (Irrec_mkt − Irrec_*)·meses. "
    "Pagar acima do justo só afeta a diferença de irrecuperáveis; por isso, comprar mais caro nunca infla "
    "artificialmente o resultado. Se quiseres ignorar 'barganhas', aplica max(0, ·)."
)

with st.expander("Notas"):
    st.markdown(
        """
        • P* nao inclui valorizacao do imovel na reta de custos; a apreciacao e aplicada apos resolver o preco justo.  
        • 'Custo extra de uso' considera apenas o excesso de possuir vs alugar (se for menor, nao conta como credito aqui).  
        • mis_future = mis_today · (1+g)^T, garantindo consistencia temporal.  
        • Evite dupla contagem: custos de venda ja estao na linha de custos irrecuperaveis; nao subtraia/ some novamente no patrimonio.
        """
    )
