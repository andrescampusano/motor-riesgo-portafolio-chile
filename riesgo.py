"""
riesgo.py
Funciones del proyecto "Motor de riesgo de un portafolio de acciones chilenas".

Convenciones que se usan en todo el proyecto (no se mezclan):
- Precios: cierre ajustado por dividendos y splits (auto_adjust=True), así que el retorno
  de las acciones es retorno total.
- Referencia: el ^IPSA no descarga en Yahoo, así que uso el ETF CFMITNIPSA.SN, que replica
  el IPSA. Es un ETF, no el índice en sí.
- Retornos: logarítmicos diarios, ln(P_t / P_t-1).
- Anualización: 250 días (la del curso). Retorno anual = media diaria * 250,
  covarianza anual = covarianza diaria * 250.
- Retorno diario del portafolio: suma ponderada de los retornos log de cada acción, con
  pesos constantes (aproximación estándar a escala diaria).
- VaR y CVaR: a 1 día, no se anualizan. Se entregan como pérdida positiva, en unidades
  de retorno log (casi lo mismo que un porcentaje).
- Última fecha de datos: último cierre completo. Se saca el día en curso, porque durante
  el horario de mercado es un precio parcial.
"""
import glob
import os
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

# ----------------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------------
INICIO = "2018-01-01"
DIAS_ANIO = 250
ACCIONES = ["FALABELLA.SN", "COPEC.SN", "SQM-B.SN", "BCI.SN", "CENCOSUD.SN", "CMPC.SN"]
INDICE = "CFMITNIPSA.SN"   # ETF que replica el IPSA (^IPSA no descarga en Yahoo)
DATA_DIR = "data"
NIVELES = (0.95, 0.99)


# ----------------------------------------------------------------------------
# Datos: descarga, verificación, limpieza y guardado
# ----------------------------------------------------------------------------
def descargar_precios(tickers, inicio=INICIO):
    """Baja el cierre ajustado desde `inicio` hasta el último cierre completo."""
    crudo = yf.download(tickers, start=inicio, auto_adjust=True, progress=False)
    px = crudo["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    px = px.reindex(columns=tickers)            # un ticker que falla queda como columna vacía
    if getattr(px.index, "tz", None) is not None:
        px.index = px.index.tz_localize(None)
    px.index = pd.to_datetime(px.index).normalize()
    px.index.name = "Date"
    hoy = pd.Timestamp.today().normalize()
    px = px[px.index < hoy]                     # fuera el día de hoy (puede ser precio parcial)
    return px.dropna(how="all").sort_index()


def verificar_datos(px, inicio=INICIO, tol_inicio_dias=15, tol_final_dias=5, max_nan=0.02):
    """
    Revisa que cada serie tenga datos completos desde `inicio`.
    Devuelve el reporte y la lista de problemas (lista vacía = todo bien).
    """
    inicio = pd.Timestamp(inicio)
    ultima_global = px.index.max()
    filas, problemas = {}, []
    for t in px.columns:
        s = px[t]
        v = s.dropna()
        if v.empty:
            filas[t] = dict(primera=None, ultima=None, n_obs=0, pct_NaN=1.0, pmin=np.nan, pmax=np.nan)
            problemas.append(f"{t}: no descargó ningún dato")
            continue
        primera, ultima, pct_nan = v.index.min(), v.index.max(), s.isna().mean()
        filas[t] = dict(primera=primera.date(), ultima=ultima.date(), n_obs=len(v),
                        pct_NaN=round(pct_nan, 4), pmin=v.min(), pmax=v.max())
        if primera > inicio + pd.Timedelta(days=tol_inicio_dias):
            problemas.append(f"{t}: los datos parten el {primera.date()} (no cubre {inicio.date()})")
        if ultima < ultima_global - pd.Timedelta(days=tol_final_dias):
            problemas.append(f"{t}: los datos terminan el {ultima.date()} (dejó de cotizar o falló)")
        if pct_nan > max_nan:
            problemas.append(f"{t}: {pct_nan:.1%} de días sin dato (máximo permitido {max_nan:.0%})")
        if (v <= 0).any():
            problemas.append(f"{t}: tiene precios <= 0")
    return pd.DataFrame(filas).T, problemas


def limpiar_precios(px, limite_ffill=3):
    """
    Rellena huecos de hasta `limite_ffill` días con el último precio y elimina las filas
    que sigan incompletas. Devuelve los precios limpios y cuántas filas se descartaron.
    """
    n0 = len(px)
    px = px.ffill(limit=limite_ffill).dropna(how="any")
    return px, n0 - len(px)


def guardar_precios(px, carpeta=DATA_DIR, inicio=INICIO):
    """Guarda el CSV con la fecha de descarga en el nombre y devuelve la ruta."""
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, f"precios_ajustados_{inicio}_descarga_{date.today().isoformat()}.csv")
    px.to_csv(ruta)
    return ruta


def cargar_precios_guardados(carpeta=DATA_DIR):
    """Carga el CSV más reciente de la carpeta. Si no hay ninguno, devuelve (None, None)."""
    rutas = sorted(glob.glob(os.path.join(carpeta, "precios_ajustados_*.csv")))
    if not rutas:
        return None, None
    px = pd.read_csv(rutas[-1], index_col=0, parse_dates=True)
    return px, rutas[-1]


def obtener_precios(forzar_descarga=False, carpeta=DATA_DIR, tickers=None):
    """
    Usa el CSV guardado si existe y calza con los tickers pedidos (así los resultados
    son reproducibles). Si no, descarga de Yahoo. Siempre verifica los datos y, si hay
    problemas, se detiene sin guardar. Devuelve precios crudos, ruta del CSV y reporte.
    """
    tickers = list(tickers) if tickers else ACCIONES + [INDICE]
    px, ruta = (None, None) if forzar_descarga else cargar_precios_guardados(carpeta)
    if px is not None and list(px.columns) != tickers:
        print("El CSV guardado no calza con los tickers pedidos, se descarga de nuevo.")
        px = None
    descargado = px is None
    if descargado:
        px = descargar_precios(tickers)
    reporte, problemas = verificar_datos(px)
    print(reporte.to_string())
    if problemas:
        raise ValueError("Datos incompletos, no se guardó el CSV:\n - " + "\n - ".join(problemas))
    if descargado:
        ruta = guardar_precios(px, carpeta)
    return px, ruta, reporte


# ----------------------------------------------------------------------------
# Retornos
# ----------------------------------------------------------------------------
def retornos_log(px):
    """Retornos logarítmicos diarios: ln(P_t / P_t-1)."""
    return np.log(px / px.shift(1)).dropna()


def detectar_saltos(px, umbral=0.25):
    """
    Retornos diarios con |r| > umbral (25% por defecto). En acciones líquidas casi siempre
    es un error de datos (split sin ajustar, precio malo), así que se revisan a mano.
    """
    r = retornos_log(px).stack()
    return r[r.abs() > umbral]


# ----------------------------------------------------------------------------
# Portafolios y normalidad
# ----------------------------------------------------------------------------
def estadisticos_anuales(r_acc):
    """Retorno anual (vector) y covarianza anual (matriz), con DIAS_ANIO = 250."""
    mu = r_acc.mean().values * DIAS_ANIO
    cov = r_acc.cov().values * DIAS_ANIO
    return mu, cov


def validar_cov(cov):
    """La covarianza tiene que ser simétrica y definida positiva."""
    assert np.allclose(cov, cov.T), "La covarianza no es simétrica"
    assert np.all(np.linalg.eigvalsh(cov) > 0), "La covarianza no es definida positiva"


def metricas_portafolio(w, mu, cov, rf=0.0):
    """Retorno anual, volatilidad anual y Sharpe de un portafolio con pesos w."""
    assert np.isclose(np.sum(w), 1.0), "Los pesos no suman 1"
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    return ret, vol, (ret - rf) / vol


def portafolios_aleatorios(mu, cov, n=50_000, rf=0.0, seed=42):
    """Pesos al azar (positivos, suman 1), como en el curso. Devuelve pesos, retornos, vols y Sharpes."""
    rng = np.random.default_rng(seed)
    W = rng.random((n, len(mu)))
    W = W / W.sum(axis=1, keepdims=True)
    rets = W @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
    return W, rets, vols, (rets - rf) / vols


def test_normalidad(x):
    """Asimetría, curtosis en exceso (la normal vale 0) y Jarque-Bera de una serie de retornos."""
    jb = stats.jarque_bera(x)
    return {"media_diaria": x.mean(), "vol_diaria": x.std(), "asimetria": stats.skew(x),
            "curtosis_exceso": stats.kurtosis(x), "JB": jb.statistic, "p_valor": jb.pvalue}


# ----------------------------------------------------------------------------
# VaR y CVaR a 1 día
# ----------------------------------------------------------------------------
def var_cvar_historico(x, nivel):
    """VaR = -(percentil 1-nivel). CVaR = pérdida promedio de los días que cayeron más que el VaR."""
    x = np.asarray(x)
    q = np.quantile(x, 1 - nivel)
    return float(-q), float(-x[x <= q].mean())


def var_cvar_parametrico(x, nivel):
    """Supone retornos normales con la media y desviación diaria de x (usa norm.ppf)."""
    x = np.asarray(x)
    mu, sd = x.mean(), x.std(ddof=1)
    z = stats.norm.ppf(1 - nivel)
    var = -(mu + sd * z)
    cvar = -mu + sd * stats.norm.pdf(z) / (1 - nivel)
    return float(var), float(cvar)


def simular_portafolio(r_acc, w, n_sim=100_000, seed=42):
    """Monte Carlo: simula retornos diarios de las 6 acciones juntas (normal multivariada) y los combina con w."""
    rng = np.random.default_rng(seed)
    mu_d = r_acc.mean().values
    cov_d = r_acc.cov().values
    sims = rng.multivariate_normal(mu_d, cov_d, size=n_sim)
    return sims @ w


def tabla_var(x, sims, niveles=NIVELES):
    """Tabla comparativa de los tres métodos. Filas = método, columnas = VaR y CVaR por nivel."""
    filas = {}
    for metodo, serie, f in [("Histórico", x, var_cvar_historico),
                             ("Paramétrico (normal)", x, var_cvar_parametrico),
                             ("Monte Carlo", sims, var_cvar_historico)]:
        fila = {}
        for a in niveles:
            v, c = f(serie, a)
            fila[f"VaR {a:.0%}"] = v
            fila[f"CVaR {a:.0%}"] = c
        filas[metodo] = fila
    tabla = pd.DataFrame(filas).T
    niv = sorted(niveles)
    for m in tabla.index:
        for a in niv:
            assert tabla.loc[m, f"CVaR {a:.0%}"] >= tabla.loc[m, f"VaR {a:.0%}"], f"CVaR < VaR en {m}"
        for a1, a2 in zip(niv[:-1], niv[1:]):
            assert tabla.loc[m, f"VaR {a2:.0%}"] >= tabla.loc[m, f"VaR {a1:.0%}"], f"VaR {a2:.0%} < VaR {a1:.0%} en {m}"
    assert (tabla > 0).all().all(), "Hay VaR/CVaR no positivos"
    return tabla


def tasa_excepciones(x, tabla, niveles=NIVELES):
    """Fracción de días de la muestra en que la pérdida real superó el VaR de cada método."""
    x = np.asarray(x)
    out = {m: {f"{a:.0%}": float((x < -tabla.loc[m, f"VaR {a:.0%}"]).mean()) for a in niveles}
           for m in tabla.index}
    return pd.DataFrame(out).T
