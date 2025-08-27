from pma_d import compute_pma_d, PMADParams

def test_scenario_a():
    res = compute_pma_d(
        preco_mercado=200_000,
        area_m2=90,
        ano_construcao=1988,
        classe_energetica="E",
        capex_manutencao_previsto=15_000,
    )
    assert res["pma_d"] < 160_000

def test_scenario_b():
    res = compute_pma_d(
        preco_mercado=200_000,
        area_m2=90,
        ano_construcao=2002,
        classe_energetica="C",
    )
    assert abs(res["pma_d"] - 200_000) / 200_000 < 0.3

def test_scenario_c():
    params = PMADParams()
    res_c = compute_pma_d(200_000, 90, 2002, "C", params=params)
    res_a = compute_pma_d(200_000, 90, 2002, "A", params=params)
    assert res_a["pma_d"] > res_c["pma_d"]
