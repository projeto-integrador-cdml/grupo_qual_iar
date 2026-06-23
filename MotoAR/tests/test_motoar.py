"""
╔══════════════════════════════════════════════════════════════╗
║  MotoAR — Suite de Testes Unitários (pytest)                ║
║                                                             ║
║  Cobre:                                                     ║
║    • motoar_pipeline.py  → funções utilitárias              ║
║    • silver_transform.py → limpeza e features               ║
║    • gold_build.py       → agregações e modelo              ║
║    • quality_check.py    → expectativas de qualidade        ║
║                                                             ║
║  Uso:                                                       ║
║    pytest tests/test_motoar.py -v                           ║
║    pytest tests/test_motoar.py -v --tb=short                ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

# Adiciona a raiz do projeto ao sys.path para os imports funcionarem
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quality"))

# ──────────────────────────────────────────────────────────────────────────────
# FIXTURES — dados sintéticos reutilizáveis
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def df_iqair_valido():
    """DataFrame IQAir com dados válidos."""
    return pd.DataFrame({
        "created_at":      pd.date_range("2025-01-01", periods=24, freq="h"),
        "sensor_location": ["Brasília"] * 24,
        "aqi":             [10 + i for i in range(24)],
        "pm25":            [5.0 + i * 0.5 for i in range(24)],
        "aqi_category":    ["Bom"] * 24,
        "temperature":     ["25.0°C"] * 24,
        "humidity":        ["60%"] * 24,
        "wind_speed":      ["10km/h"] * 24,
    })


@pytest.fixture
def df_iqair_invalido():
    """DataFrame IQAir com status inválido e valores fora de range."""
    return pd.DataFrame({
        "created_at":      pd.date_range("2025-01-01", periods=6, freq="h"),
        "sensor_location": ["Brasília"] * 6,
        "aqi":             [10, 600, -5, 50, 100, 200],   # 600 e -5 = fora do range
        "pm25":            [5.0, 600.0, -1.0, 10.0, 20.0, 30.0],
        "aqi_category":    ["Bom", "não encontrado", "Bom", "Bom", "Bom", "Bom"],
        "temperature":     ["25.0°C"] * 6,
        "humidity":        ["60%"] * 6,
        "wind_speed":      ["10km/h"] * 6,
    })


@pytest.fixture
def df_inmet_valido():
    """DataFrame INMET limpo com colunas essenciais."""
    n = 48
    dts = pd.date_range("2025-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "datetime":  dts,
        "station":   ["CRAS Fercal"] * n,
        "pm25":      [10.0 + (i % 5) for i in range(n)],
        "pm10":      [20.0 + (i % 5) for i in range(n)],
        "temp":      [22.0] * n,
        "rh":        [65.0] * n,
        "rain":      [0.0] * n,
        "no2":       [15.0] * n,
        "co":        [0.5] * n,
        "o3":        [40.0] * n,
        "so2":       [5.0] * n,
        "hour":      [d.hour for d in dts],
        "month":     [d.month for d in dts],
        "season":    ["chuva"] * n,
    })


@pytest.fixture
def df_com_duplicatas():
    """DataFrame com linhas duplicadas."""
    base = pd.DataFrame({
        "datetime":  pd.to_datetime(["2025-01-01 00:00", "2025-01-01 01:00",
                                     "2025-01-01 00:00"]),
        "station":   ["A", "A", "A"],
        "pm25":      [10.0, 12.0, 10.0],
    })
    return base


@pytest.fixture
def series_sensor_travado():
    """Série onde o sensor fica travado (mesmo valor repetido)."""
    return pd.Series([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 12.0])


@pytest.fixture
def series_sensor_normal():
    """Série com variação normal do sensor."""
    return pd.Series([10.0, 12.5, 9.0, 14.0, 11.5, 8.5, 13.0, 10.5])


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 1 — motoar_pipeline.py
# ══════════════════════════════════════════════════════════════════════════════

class TestStripUnit:
    """Testa a remoção de sufixos de unidade."""

    def setup_method(self):
        from motoar_pipeline import strip_unit
        self.strip_unit = strip_unit

    def test_remove_celsius(self):
        s = pd.Series(["25.0°C", "30.5°C", "18°C"])
        result = self.strip_unit(s, "°C")
        assert list(result) == [25.0, 30.5, 18.0]

    def test_remove_percent(self):
        s = pd.Series(["60%", "75%", "90%"])
        result = self.strip_unit(s, "%")
        assert list(result) == [60.0, 75.0, 90.0]

    def test_remove_kmh(self):
        s = pd.Series(["10km/h", "25km/h", "0km/h"])
        result = self.strip_unit(s, "km/h")
        assert list(result) == [10.0, 25.0, 0.0]

    def test_valor_invalido_vira_nan(self):
        s = pd.Series(["N/A°C", "ausente°C"])
        result = self.strip_unit(s, "°C")
        assert result.isna().all()

    def test_sem_unidade_converte_direto(self):
        s = pd.Series(["10.5", "20.0", "30"])
        result = self.strip_unit(s, "°C")   # sem correspondência = sem mudança
        assert list(result) == [10.5, 20.0, 30.0]


class TestAddTemporalFeatures:
    """Testa a geração de features temporais."""

    def setup_method(self):
        from motoar_pipeline import add_temporal_features
        self.add_temporal_features = add_temporal_features

    def test_colunas_criadas(self):
        df = pd.DataFrame({"dt": pd.date_range("2025-07-15 08:00", periods=3, freq="h")})
        result = self.add_temporal_features(df.copy(), "dt")
        for col in ["year","month","day","hour","weekday","is_weekend",
                    "season","hour_sin","hour_cos","month_sin","month_cos","is_dry_season"]:
            assert col in result.columns, f"coluna ausente: {col}"

    def test_estacao_seca_julho(self):
        df = pd.DataFrame({"dt": pd.to_datetime(["2025-07-01 12:00"])})
        result = self.add_temporal_features(df.copy(), "dt")
        assert result["season"].iloc[0] == "seca"
        assert result["is_dry_season"].iloc[0] == 1

    def test_estacao_chuva_janeiro(self):
        df = pd.DataFrame({"dt": pd.to_datetime(["2025-01-15 10:00"])})
        result = self.add_temporal_features(df.copy(), "dt")
        assert result["season"].iloc[0] == "chuva"
        assert result["is_dry_season"].iloc[0] == 0

    def test_estacao_transicao_novembro(self):
        df = pd.DataFrame({"dt": pd.to_datetime(["2025-11-10 09:00"])})
        result = self.add_temporal_features(df.copy(), "dt")
        assert result["season"].iloc[0] == "transição"

    def test_is_weekend_sabado(self):
        # 2025-01-04 é sábado
        df = pd.DataFrame({"dt": pd.to_datetime(["2025-01-04 10:00"])})
        result = self.add_temporal_features(df.copy(), "dt")
        assert result["is_weekend"].iloc[0] == 1

    def test_is_weekend_segunda(self):
        # 2025-01-06 é segunda
        df = pd.DataFrame({"dt": pd.to_datetime(["2025-01-06 10:00"])})
        result = self.add_temporal_features(df.copy(), "dt")
        assert result["is_weekend"].iloc[0] == 0

    def test_codificacao_ciclica_hora_range(self):
        df = pd.DataFrame({"dt": pd.date_range("2025-01-01", periods=24, freq="h")})
        result = self.add_temporal_features(df.copy(), "dt")
        assert result["hour_sin"].between(-1, 1).all()
        assert result["hour_cos"].between(-1, 1).all()

    def test_codificacao_ciclica_hora_0_igual_24(self):
        """Hora 0 e hora 24 devem ter mesma codificação cíclica."""
        df = pd.DataFrame({"dt": pd.to_datetime(["2025-01-01 00:00"])})
        result = self.add_temporal_features(df.copy(), "dt")
        sin_0 = result["hour_sin"].iloc[0]
        cos_0 = result["hour_cos"].iloc[0]
        # sin(0) = 0, cos(0) = 1
        assert abs(sin_0 - 0.0) < 1e-9
        assert abs(cos_0 - 1.0) < 1e-9


class TestAddRollingFeatures:
    """Testa a criação de features de janela deslizante."""

    def setup_method(self):
        from motoar_pipeline import add_rolling_features
        self.add_rolling_features = add_rolling_features

    def test_colunas_criadas_default(self):
        df = pd.DataFrame({"pm25": [10.0, 12.0, 14.0, 11.0, 9.0, 13.0]})
        result = self.add_rolling_features(df.copy(), "pm25")
        for col in ["pm25_roll3h", "pm25_roll6h", "pm25_roll24h",
                    "pm25_lag1h", "pm25_lag3h", "pm25_delta1h"]:
            assert col in result.columns

    def test_rolling_media_correta(self):
        df = pd.DataFrame({"pm25": [10.0, 20.0, 30.0, 40.0]})
        result = self.add_rolling_features(df.copy(), "pm25", windows=[3])
        # último valor: média de 20, 30, 40 = 30.0
        assert result["pm25_roll3h"].iloc[3] == pytest.approx(30.0)

    def test_lag1_correto(self):
        df = pd.DataFrame({"pm25": [5.0, 10.0, 15.0]})
        result = self.add_rolling_features(df.copy(), "pm25", windows=[])
        assert result["pm25_lag1h"].iloc[1] == 5.0
        assert result["pm25_lag1h"].iloc[2] == 10.0

    def test_lag3_correto(self):
        df = pd.DataFrame({"pm25": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = self.add_rolling_features(df.copy(), "pm25", windows=[])
        assert result["pm25_lag3h"].iloc[3] == 1.0
        assert result["pm25_lag3h"].iloc[4] == 2.0

    def test_delta_correto(self):
        df = pd.DataFrame({"pm25": [10.0, 15.0, 12.0]})
        result = self.add_rolling_features(df.copy(), "pm25", windows=[])
        assert result["pm25_delta1h"].iloc[1] == pytest.approx(5.0)
        assert result["pm25_delta1h"].iloc[2] == pytest.approx(-3.0)


class TestDetectSensorFrozen:
    """Testa a detecção de sensor travado."""

    def setup_method(self):
        from motoar_pipeline import detect_sensor_frozen
        self.detect = detect_sensor_frozen

    def test_sensor_travado_detectado(self, series_sensor_travado):
        result = self.detect(series_sensor_travado, window=6)
        # A partir do índice 5 (6 iguais) deve ser 1
        assert result.iloc[5] == 1
        assert result.iloc[6] == 1

    def test_sensor_normal_nao_detectado(self, series_sensor_normal):
        result = self.detect(series_sensor_normal, window=6)
        assert result.sum() == 0

    def test_retorna_series(self, series_sensor_travado):
        result = self.detect(series_sensor_travado)
        assert isinstance(result, pd.Series)

    def test_valores_binarios(self, series_sensor_travado):
        result = self.detect(series_sensor_travado)
        assert set(result.dropna().unique()).issubset({0, 1})

    def test_window_personalizada(self):
        s = pd.Series([5.0, 5.0, 5.0, 5.0, 10.0])
        result = self.detect(s, window=4)
        assert result.iloc[3] == 1  # 4 valores iguais
        assert result.iloc[4] == 0  # quebrou a sequência


class TestPipelineReport:
    """Testa o sistema de log de etapas do pipeline."""

    def setup_method(self):
        from motoar_pipeline import PipelineReport
        self.PipelineReport = PipelineReport

    def test_log_registra_etapa(self):
        rpt = self.PipelineReport("teste")
        rpt.log("drop nulos", 1000, 900)
        assert len(rpt.steps) == 1
        assert rpt.steps[0]["removed"] == 100

    def test_log_calcula_pct_correta(self):
        rpt = self.PipelineReport("teste")
        rpt.log("filtro", 200, 150)
        assert rpt.steps[0]["pct_removed"] == pytest.approx(25.0)

    def test_summary_retorna_dict(self):
        rpt = self.PipelineReport("iqair")
        rpt.log("passo1", 100, 90)
        s = rpt.summary()
        assert isinstance(s, dict)
        assert s["dataset"] == "iqair"
        assert s["total_removed"] == 10

    def test_log_sem_remocao(self):
        rpt = self.PipelineReport("teste")
        rpt.log("sem mudança", 500, 500)
        assert rpt.steps[0]["removed"] == 0
        assert rpt.steps[0]["pct_removed"] == 0.0

    def test_log_zero_before_nao_divide_por_zero(self):
        rpt = self.PipelineReport("teste")
        rpt.log("vazio", 0, 0)
        assert rpt.steps[0]["pct_removed"] == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 2 — silver_transform.py
# ══════════════════════════════════════════════════════════════════════════════

class TestSilverDetectFrozen:
    """Testa detect_frozen do silver_transform (equivalente, separado)."""

    def setup_method(self):
        from silver_transform import detect_frozen
        self.detect = detect_frozen

    def test_detecta_congelamento(self):
        s = pd.Series([7.0] * 8)
        result = self.detect(s, window=6)
        assert result.iloc[5] == 1

    def test_nao_detecta_variacao(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = self.detect(s, window=6)
        assert result.sum() == 0

    def test_nan_nao_conta_como_congelado(self):
        s = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        result = self.detect(s, window=6)
        # std de NaN é NaN, não 0
        assert result.iloc[5] == 0


class TestSilverAddTemporalFeatures:
    """Testa add_temporal_features do silver_transform."""

    def setup_method(self):
        from silver_transform import add_temporal_features
        self.fn = add_temporal_features

    def test_todas_as_colunas_presentes(self):
        df = pd.DataFrame({"dt": pd.date_range("2025-03-10 06:00", periods=5, freq="h")})
        result = self.fn(df.copy(), "dt")
        esperadas = ["year","month","day","hour","weekday","is_weekend",
                     "season","hour_sin","hour_cos","month_sin","month_cos","is_dry_season"]
        for col in esperadas:
            assert col in result.columns

    def test_hora_6_sin_positivo(self):
        df = pd.DataFrame({"dt": pd.to_datetime(["2025-01-01 06:00"])})
        result = self.fn(df.copy(), "dt")
        # sin(2π * 6/24) = sin(π/2) = 1
        assert result["hour_sin"].iloc[0] == pytest.approx(1.0, abs=1e-9)


class TestSilverAddRollingFeatures:
    """Testa add_rolling_features do silver_transform."""

    def setup_method(self):
        from silver_transform import add_rolling_features
        self.fn = add_rolling_features

    def test_colunas_criadas(self):
        df = pd.DataFrame({"aqi": [10.0, 12.0, 14.0, 11.0, 9.0, 13.0]})
        result = self.fn(df.copy(), "aqi", windows=[3, 6])
        assert "aqi_roll3h" in result.columns
        assert "aqi_roll6h" in result.columns
        assert "aqi_lag1h" in result.columns
        assert "aqi_lag3h" in result.columns

    def test_sem_windows_cria_lags(self):
        df = pd.DataFrame({"aqi": [5.0, 10.0, 15.0]})
        result = self.fn(df.copy(), "aqi", windows=[])
        assert "aqi_lag1h" in result.columns
        assert "aqi_lag3h" in result.columns


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 3 — gold_build.py  (build_aggregations)
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildAggregations:
    """Testa a geração das agregações Gold."""

    def setup_method(self):
        from gold_build import build_aggregations
        self.fn = build_aggregations

    def _df_inmet(self, n=100):
        """Cria DataFrame INMET mínimo para testes."""
        dts = pd.date_range("2025-01-01", periods=n, freq="h")
        return pd.DataFrame({
            "datetime": dts,
            "station":  ["CRAS Fercal"] * n,
            "pm25":     np.random.uniform(5, 30, n),
            "no2":      np.random.uniform(5, 50, n),
            "co":       np.random.uniform(0.1, 2.0, n),
            "o3":       np.random.uniform(20, 80, n),
            "so2":      np.random.uniform(1, 20, n),
            "rain":     np.zeros(n),
            "hour":     [d.hour for d in dts],
            "month":    [d.month for d in dts],
            "date":     [d.date() for d in dts],
            "season":   ["Chuva"] * n,
        })

    def test_retorna_dict(self):
        result = self.fn(self._df_inmet(), None)
        assert isinstance(result, dict)

    def test_chave_inmet_presente(self):
        result = self.fn(self._df_inmet(), None)
        assert "inmet" in result

    def test_metricas_basicas_inmet(self):
        df = self._df_inmet()
        result = self.fn(df, None)
        inmet = result["inmet"]
        assert "total" in inmet
        assert "pm25_mean" in inmet
        assert "pm25_max" in inmet
        assert inmet["total"] == len(df)

    def test_pm25_mean_coerente(self):
        df = self._df_inmet()
        result = self.fn(df, None)
        assert result["inmet"]["pm25_mean"] == pytest.approx(df["pm25"].mean(), rel=1e-3)

    def test_aggregation_mensais_presentes(self):
        result = self.fn(self._df_inmet(), None)
        assert "monthly" in result["inmet"]
        assert "labels" in result["inmet"]["monthly"]
        assert "mean" in result["inmet"]["monthly"]

    def test_heatmap_presente(self):
        result = self.fn(self._df_inmet(), None)
        assert "heatmap" in result["inmet"]
        assert "z" in result["inmet"]["heatmap"]

    def test_rain_effect_presente(self):
        result = self.fn(self._df_inmet(), None)
        assert "rain_effect" in result["inmet"]

    def test_sem_iqair_nao_quebra(self):
        result = self.fn(self._df_inmet(), None)
        assert "inmet" in result   # não lançou exceção

    def test_com_iqair_adiciona_chave(self):
        df_iq = pd.DataFrame({
            "created_at":      pd.date_range("2025-01-01", periods=10, freq="h"),
            "sensor_location": ["Brasília"] * 10,
            "aqi":             np.random.uniform(5, 50, 10),
            "pm25":            np.random.uniform(2, 25, 10),
            "hour":            list(range(10)),
            "date":            [d.date() for d in pd.date_range("2025-01-01", periods=10, freq="h")],
        })
        result = self.fn(self._df_inmet(), df_iq)
        assert "iqair" in result


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 4 — quality/quality_check.py
# ══════════════════════════════════════════════════════════════════════════════

class TestQualityCheck:
    """Testa o sistema de expectativas de qualidade."""

    def setup_method(self):
        from quality_check import check
        self.check = check

    def _df(self, **kwargs):
        base = {
            "pm25": [10.0, 12.0, 14.0, 11.0, 9.0],
            "aqi":  [20,   25,   30,   22,   18],
        }
        base.update(kwargs)
        return pd.DataFrame(base)

    # ── not_null_pct ─────────────────────────────────────────────────────────

    def test_not_null_passa_100pct(self):
        df = self._df()
        exp = [("teste", "pm25", "not_null_pct", 0.9, True)]
        results = self.check(df, exp)
        assert results[0]["passed"] is True

    def test_not_null_falha_com_nulos(self):
        df = pd.DataFrame({"pm25": [10.0, None, None, None, None]})
        exp = [("teste", "pm25", "not_null_pct", 0.9, True)]
        results = self.check(df, exp)
        assert results[0]["passed"] is False

    def test_not_null_exatamente_no_limite(self):
        df = pd.DataFrame({"pm25": [10.0, None, None, None, None,
                                     12.0, 13.0, 14.0, 15.0, 11.0]})
        # 60% não-nulos, threshold 0.5 → passa
        exp = [("teste", "pm25", "not_null_pct", 0.5, True)]
        results = self.check(df, exp)
        assert results[0]["passed"] is True

    # ── between ──────────────────────────────────────────────────────────────

    def test_between_todos_dentro(self):
        df = self._df()
        exp = [("range_pm25", "pm25", "between", (0, 500), True)]
        results = self.check(df, exp)
        assert results[0]["passed"] is True

    def test_between_valor_fora(self):
        df = pd.DataFrame({"pm25": [10.0, 10.0, 10.0, 10.0, 600.0]})
        exp = [("range_pm25", "pm25", "between", (0, 500), True)]
        results = self.check(df, exp)
        assert results[0]["passed"] is False

    def test_between_coluna_ausente_nao_falha_critico(self):
        df = self._df()
        exp = [("col_inexistente", "nao_existe", "between", (0, 100), True)]
        results = self.check(df, exp)
        assert results[0]["passed"] is True  # coluna ausente = ignorado

    # ── no_duplicates ─────────────────────────────────────────────────────────

    def test_no_duplicates_passa_sem_dups(self):
        df = pd.DataFrame({
            "datetime": pd.date_range("2025-01-01", periods=5, freq="h"),
            "station":  ["A"] * 5,
        })
        exp = [("dups", None, "no_duplicates", ["datetime","station"], True)]
        results = self.check(df, exp)
        assert results[0]["passed"] is True

    def test_no_duplicates_falha_com_dups(self, df_com_duplicatas):
        exp = [("dups", None, "no_duplicates", ["datetime","station"], True)]
        results = self.check(df_com_duplicatas, exp)
        assert results[0]["passed"] is False

    # ── max_mean ─────────────────────────────────────────────────────────────

    def test_max_mean_passa(self):
        df = pd.DataFrame({"sensor_frozen": [0, 0, 0, 1, 0]})
        exp = [("frozen", "sensor_frozen", "max_mean", 0.5, False)]
        results = self.check(df, exp)
        assert results[0]["passed"] is True  # mean=0.2 ≤ 0.5

    def test_max_mean_falha(self):
        df = pd.DataFrame({"sensor_frozen": [1, 1, 1, 1, 0]})
        exp = [("frozen", "sensor_frozen", "max_mean", 0.5, False)]
        results = self.check(df, exp)
        assert results[0]["passed"] is False  # mean=0.8 > 0.5

    # ── mean_between ─────────────────────────────────────────────────────────

    def test_mean_between_passa(self):
        df = pd.DataFrame({"pm25": [10.0, 12.0, 11.0, 10.5, 11.5]})
        exp = [("media_ok", "pm25", "mean_between", (0, 100), True)]
        results = self.check(df, exp)
        assert results[0]["passed"] is True

    def test_mean_between_falha_acima(self):
        df = pd.DataFrame({"pm25": [110.0, 120.0, 115.0]})
        exp = [("media_alta", "pm25", "mean_between", (0, 100), True)]
        results = self.check(df, exp)
        assert results[0]["passed"] is False

    # ── min_rows ──────────────────────────────────────────────────────────────

    def test_min_rows_passa(self):
        df = pd.DataFrame({"x": range(1000)})
        exp = [("rows", None, "min_rows", 100, True)]
        results = self.check(df, exp)
        assert results[0]["passed"] is True

    def test_min_rows_falha(self):
        df = pd.DataFrame({"x": range(5)})
        exp = [("rows", None, "min_rows", 100, True)]
        results = self.check(df, exp)
        assert results[0]["passed"] is False

    # ── severidade ───────────────────────────────────────────────────────────

    def test_falha_critica_tem_severity_critical(self):
        df = pd.DataFrame({"x": range(5)})
        exp = [("rows", None, "min_rows", 100, True)]
        results = self.check(df, exp)
        assert results[0]["severity"] == "critical"

    def test_falha_nao_critica_tem_severity_warning(self):
        df = pd.DataFrame({"x": range(5)})
        exp = [("rows", None, "min_rows", 100, False)]
        results = self.check(df, exp)
        assert results[0]["severity"] == "warning"

    def test_sucesso_tem_severity_ok(self):
        df = pd.DataFrame({"pm25": [10.0, 12.0]})
        exp = [("ok", "pm25", "not_null_pct", 0.9, True)]
        results = self.check(df, exp)
        assert results[0]["severity"] == "ok"

    # ── múltiplas expectativas ───────────────────────────────────────────────

    def test_multiplas_expectativas_retorna_todas(self):
        df = pd.DataFrame({"pm25": [10.0, 12.0, 14.0]})
        exps = [
            ("e1", "pm25", "not_null_pct", 0.9, True),
            ("e2", "pm25", "between", (0, 500), True),
            ("e3", None, "min_rows", 1, True),
        ]
        results = self.check(df, exps)
        assert len(results) == 3
        assert all(r["passed"] for r in results)

    def test_resultado_tem_campos_obrigatorios(self):
        df = self._df()
        exp = [("teste", "pm25", "not_null_pct", 0.9, True)]
        results = self.check(df, exp)
        for campo in ["expectation", "column", "passed", "severity", "detail"]:
            assert campo in results[0]

    def test_passed_sempre_bool(self):
        df = self._df()
        exps = [
            ("e1", "pm25", "not_null_pct", 0.9, True),
            ("e2", "pm25", "between", (0, 5), True),   # vai falhar
        ]
        results = self.check(df, exps)
        for r in results:
            assert isinstance(r["passed"], bool)


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 5 — Testes de integração leve (sem I/O de arquivo)
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegracaoPipelineCompleto:
    """Testa o fluxo completo das funções encadeadas."""

    def test_temporal_seguido_de_rolling(self):
        from motoar_pipeline import add_temporal_features, add_rolling_features
        df = pd.DataFrame({
            "dt":  pd.date_range("2025-07-01 08:00", periods=24, freq="h"),
            "pm25": np.random.uniform(10, 40, 24),
        })
        df = add_temporal_features(df.copy(), "dt")
        df = add_rolling_features(df, "pm25", windows=[3, 6])
        assert "hour_sin" in df.columns
        assert "pm25_roll3h" in df.columns
        assert len(df) == 24

    def test_frozen_detection_apos_temporal(self):
        from motoar_pipeline import add_temporal_features, detect_sensor_frozen
        # Sensor trava nas primeiras 8 horas
        vals = [15.0]*8 + [float(i) for i in range(16)]
        df = pd.DataFrame({
            "dt":  pd.date_range("2025-01-01", periods=24, freq="h"),
            "pm25": vals,
        })
        df = add_temporal_features(df.copy(), "dt")
        df["frozen"] = detect_sensor_frozen(pd.Series(vals), window=6)
        # Índices 5,6,7 devem ser frozen=1
        assert df["frozen"].iloc[5] == 1
        assert df["frozen"].iloc[6] == 1
        assert df["frozen"].iloc[7] == 1
        # Depois de variar, não deve mais ser frozen
        assert df["frozen"].iloc[23] == 0

    def test_quality_check_no_dataframe_silver(self):
        from quality_check import check
        n = 500
        df = pd.DataFrame({
            "pm25":           np.random.uniform(5, 40, n),
            "datetime":       pd.date_range("2025-01-01", periods=n, freq="h"),
            "station":        ["CRAS Fercal"] * n,
            "sensor_frozen":  [0] * n,
            "no2":            np.random.uniform(5, 80, n),
            "co":             np.random.uniform(0.1, 5.0, n),
        })
        expectations = [
            ("pm25_nao_nulo",    "pm25",   "not_null_pct",   0.9,  True),
            ("pm25_range",       "pm25",   "between",        (0, 500), True),
            ("no_dups",          None,     "no_duplicates",  ["datetime","station"], True),
            ("frozen_ok",        "sensor_frozen", "max_mean", 0.4, False),
            ("pm25_media_ok",    "pm25",   "mean_between",   (0, 100), True),
            ("min_rows",         None,     "min_rows",       100, True),
        ]
        results = check(df, expectations)
        criticos_falhando = [r for r in results
                             if not r["passed"] and r["severity"] == "critical"]
        assert len(criticos_falhando) == 0, \
            f"Expectativas críticas falharam: {[r['expectation'] for r in criticos_falhando]}"


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 6 — Casos extremos e dados degenerados
# ══════════════════════════════════════════════════════════════════════════════

class TestCasosExtremos:
    """Testa comportamento com dados incomuns."""

    def test_strip_unit_string_vazia(self):
        from motoar_pipeline import strip_unit
        s = pd.Series([""])
        result = strip_unit(s, "°C")
        assert result.isna().iloc[0]

    def test_add_temporal_single_row(self):
        from motoar_pipeline import add_temporal_features
        df = pd.DataFrame({"dt": pd.to_datetime(["2025-12-31 23:59"])})
        result = add_temporal_features(df.copy(), "dt")
        assert result["month"].iloc[0] == 12
        assert result["hour"].iloc[0] == 23

    def test_rolling_coluna_toda_nan(self):
        from motoar_pipeline import add_rolling_features
        df = pd.DataFrame({"pm25": [np.nan, np.nan, np.nan]})
        result = add_rolling_features(df.copy(), "pm25", windows=[3])
        # rolling de NaN deve retornar NaN
        assert result["pm25_roll3h"].isna().all()

    def test_frozen_series_de_1_elemento(self):
        from motoar_pipeline import detect_sensor_frozen
        s = pd.Series([10.0])
        result = detect_sensor_frozen(s, window=6)
        assert len(result) == 1
        assert result.iloc[0] == 0  # window=6 > tamanho da série

    def test_pipeline_report_multiplos_logs(self):
        from motoar_pipeline import PipelineReport
        rpt = PipelineReport("teste")
        rpt.log("passo1", 1000, 900, "detalhe A")
        rpt.log("passo2", 900, 850, "detalhe B")
        rpt.log("passo3", 850, 850)
        assert len(rpt.steps) == 3
        assert rpt.summary()["total_removed"] == 150

    def test_quality_check_dataframe_vazio(self):
        from quality_check import check
        df = pd.DataFrame({"pm25": []})
        exp = [("rows", None, "min_rows", 1, True)]
        results = check(df, exp)
        assert results[0]["passed"] is False

    def test_quality_check_coluna_toda_nula(self):
        from quality_check import check
        df = pd.DataFrame({"pm25": [None, None, None]})
        exp = [("nulos", "pm25", "not_null_pct", 0.9, True)]
        results = check(df, exp)
        assert results[0]["passed"] is False

    def test_season_map_todos_meses_cobertos(self):
        from motoar_pipeline import SEASON_MAP
        assert set(SEASON_MAP.keys()) == set(range(1, 13))

    def test_season_map_valores_validos(self):
        from motoar_pipeline import SEASON_MAP
        valores_validos = {"chuva", "seca", "transição"}
        for v in SEASON_MAP.values():
            assert v in valores_validos
