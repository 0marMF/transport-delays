"""Tests del contrato de datos y del guard anti-leakage (el entregable estrella del proyecto)."""
import numpy as np
import pytest

from src import data, features, validate


def test_contrato_ok(synth_df, cfg):
    assert validate.check_contract(synth_df, cfg) == []
    assert validate.validate(synth_df, cfg) is synth_df   # encadenable


def test_contrato_fuera_de_rango(synth_df, cfg):
    bad = synth_df.copy()
    bad.loc[0, "humidity_percent"] = 150          # fuera de [0, 100]
    assert any("humidity_percent" in i for i in validate.check_contract(bad, cfg))
    with pytest.raises(validate.DataContractError):
        validate.validate(bad, cfg)


def test_contrato_categoria_rara(synth_df, cfg):
    bad = synth_df.copy()
    bad.loc[0, "season"] = "Monsoon"
    assert any("season" in i for i in validate.check_contract(bad, cfg))


def test_contrato_nulo_inesperado(synth_df, cfg):
    bad = synth_df.copy()
    bad.loc[0, "temperature_C"] = np.nan          # temperature_C no es nullable
    assert any("temperature_C" in i for i in validate.check_contract(bad, cfg))


def test_guard_atrapa_leakage(synth_df, cfg):
    X, _ = features.build_design_matrix(data.clean(synth_df), cfg)
    X_leak = X.copy()
    X_leak["delayed"] = 1                          # el target binarizado colado como feature
    with pytest.raises(validate.LeakageError):
        validate.assert_no_leakage(X_leak, cfg)


def test_guard_pasa_sin_leakage(synth_df, cfg):
    X, _ = features.build_design_matrix(data.clean(synth_df), cfg)
    assert validate.assert_no_leakage(X, cfg) is X
