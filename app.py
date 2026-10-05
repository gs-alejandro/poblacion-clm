"""
Cuadro de mando de población de Castilla-La Mancha.

Ficheros de entrada:
    data/pob_clm_calculos.csv       <- 2_calculos (panel municipio/provincia x año)
    data/pob_clm_predicciones.csv   <- 4_prediccion (Holt amortiguado, 10 años)
    data/pob_clm_clusters.csv       <- 3_clustering (columna cluster, opcional)
    data/region_*_clm.parquet       <- 0_preparacion_recintos (GeoParquet en ETRS89, sin simplificar)

Requiere Streamlit >= 1.39 (selección de objetos en st.pydeck_chart).
"""
import json
import logging
import re
from pathlib import Path

import altair as alt
import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
import shapely
import streamlit as st


####################################################################
### CONFIGURACIÓN
####################################################################

DATA_PATH = "data/pob_clm_calculos.csv"
PRED_PATH = "data/pob_clm_predicciones.csv"
CLUSTERS_PATH = "data/pob_clm_clusters.csv"

NIVEL_MUN, NIVEL_PROV, NIVEL_CA = "Municipio", "Provincia", "Comunidad autónoma"
GEO_PATHS = {
    NIVEL_MUN: "data/region_municipios_clm.parquet",
    NIVEL_PROV: "data/region_provincias_clm.parquet",
    NIVEL_CA: "data/region_clm.parquet",
}
COLUMNA_CODIGO = {NIVEL_MUN: "cod_mun", NIVEL_PROV: "cod_prov"}  # columnas de 0_preparacion_recintos
TOLERANCIA_M = 150  # simplificación de las siluetas, en metros. Solo afecta al dibujo
PLURAL = {NIVEL_MUN: "Municipios", NIVEL_PROV: "Provincias", NIVEL_CA: "Comunidades"}

PROVINCIAS = {"02": "Albacete", "13": "Ciudad Real", "16": "Cuenca", "19": "Guadalajara", "45": "Toledo"}
COD_CA, NOMBRE_CA = "CLM", "Castilla-La Mancha"

VISTA_INICIAL = dict(latitude=39.55, longitude=-3.0, zoom=6.2)
VISTA_LOCALIZADOR = dict(latitude=39.65, longitude=-3.15, zoom=5.3)  # la comunidad entera en ~280 px
ID_CAPA = "territorios"
ESTILO_TOOLTIP = {"backgroundColor": "#0c2c84", "color": "white", "fontSize": "13px"}

# coolwarm de matplotlib, muestreado en 17 paradas (azul = bajo, rojo = alto)
COOLWARM = ["#3b4cc0", "#4e68d8", "#6282ea", "#779af7", "#8db0fe", "#a3c2fe", "#b9d0f9", "#ccd9ed",
            "#dddcdc", "#ecd3c5", "#f5c4ac", "#f7b093", "#f4987a", "#eb7d62", "#dd5f4b", "#ca3b37",
            "#b40426"]
SIN_DATO = [90, 90, 90, 170]  # gris oscuro: el centro de coolwarm ya es gris claro
COLORES_TERR = ["#1f4e79", "#e0a52f", "#8c8c8c"]  # territorio, su provincia, la comunidad
# Okabe-Ito, la misma paleta que las figuras de 3_clustering
# • Negro: #000000
# • Naranja: #E69F00
# • Celeste / Azul claro: #56B4E9
# • Verde azulado: #009E73
# • Amarillo: #F0E442
# • Azul: #0072B2
# • Rojo anaranjado: #D55E00
# • Rosado / Magenta: #CC79A7
COLORES_CLUSTER = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442", "#999999"]
# Nombres y descripciones de los grupos, del capítulo de agrupación de la memoria (tipo 0 = grupo A…).
# Si se rehace el clustering, hay que revisarlos: las cifras de la app se recalculan, estos textos no.
NOMBRES_GRUPO = {"A": "rural receptor", "B": "rural en declive", "C": "intermedio arraigado",
                 "D": "en expansión"}
DESCRIPCIONES_GRUPO = {
    "A": "Municipios muy pequeños que han ganado población en la última década, aunque no recuperan "
         "lo perdido en veinte años. La llegada de población del extranjero y de otros municipios "
         "compensa su crecimiento vegetativo, muy negativo. Tienen el saldo exterior más alto de los "
         "cuatro grupos. Tras el grupo D, son los que menos población nacida en el municipio tienen.",
    "B": "El grupo más numeroso. Municipios tan pequeños como los del grupo A, pero los más "
         "envejecidos, con muy pocos menores de 16 años. Pierden población sin pausa, porque tienen el "
         "crecimiento vegetativo más negativo y se van más vecinos a otros municipios de los que llegan.",
    "C": "Municipios de tamaño medio que pierden población poco a poco. Son los que más población "
         "nacida en el propio municipio tienen. Incluyen las capitales de Albacete, Ciudad Real y "
         "Cuenca, y Talavera de la Reina.",
    "D": "Los municipios de mayor tamaño mediano, más densos y más jóvenes. Crecen con fuerza, sobre "
         "todo por la llegada de población de otros municipios, y su crecimiento vegetativo es casi "
         "nulo. Son los que menos población nacida en el municipio tienen. Están sobre todo cerca de "
         "Madrid y alrededor de algunas capitales, e incluyen Guadalajara y Toledo.",
}

# Formatos: "pct" (%), "pct2" (% con 2 decimales), "pm" (‰), "dens" (hab./km²), "int"
FACTOR = {"pct": 100, "pct2": 100, "pm": 1000, "dens": 1, "int": 1}
UNIDAD = {"pct": "%", "pct2": "%", "pm": "‰", "dens": "hab./km²", "int": "hab."}

# Acreditación de las fuentes: una línea al pie de cada página y el detalle en la barra lateral
PIE_FUENTES = ("Elaboración propia con datos del [INE](https://www.ine.es) y del "
               "[IGN y el O.A. CNIG](https://centrodedescargas.cnig.es), con licencia CC BY 4.0 "
               "[ine.es](https://www.ine.es) y CC BY 4.0 [ign.es](https://www.ign.es). "
               "La atribución completa y el aviso legal están en la barra lateral.")
PIE_AVISO = ("No es una publicación oficial. Los datos se ofrecen sin garantías y su uso es responsabilidad "
             "de quien los utiliza. Datos de población a 1 de enero de {anio}.")
FUENTES = """\
**Elaboración propia** con datos del [Instituto Nacional de Estadística (INE)](https://www.ine.es),
con licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.es) [ine.es](https://www.ine.es),
y del [Centro de Descargas del IGN y el O.A. CNIG](https://centrodedescargas.cnig.es).

**Padrón (INE)**
- [Cifras oficiales de población de los municipios españoles: Revisión del Padrón Municipal](https://ine.es/uc/dA8Sxl1o)
- [Estadística del Padrón continuo](https://ine.es/uc/UB0wLyl0)
- [Estadística de variaciones residenciales](https://ine.es/uc/wQpyKtQp)

**Censos (INE)**
- [Censo anual de población](https://ine.es/uc/jbXo8yEB): población (2021–2025), educación y relación
  con la actividad (2021–2024), y ocupación y actividad (2021–2023)

**Fenómenos demográficos (INE)**
- Movimiento natural de la población: estadísticas de [matrimonios](https://ine.es/uc/HpkZkF3p),
  [nacimientos](https://ine.es/uc/l9p5PNDe) y [defunciones](https://ine.es/uc/hdGaQCi6)
- [Estadística de migraciones y cambios de residencia](https://ine.es/uc/F8p4iuJ5)

**Información geográfica (IGN y O.A. CNIG)**
- [Nomenclátor Geográfico de Municipios y Entidades de Población](https://centrodedescargas.cnig.es/CentroDescargas/detalleArchivo?sec=9000004):
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.es) [ign.es](https://www.ign.es)
- [Límites y Unidades Administrativas Actuales](https://centrodedescargas.cnig.es/CentroDescargas/detalleArchivo?sec=9000029):
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.es) [ign.es](https://www.ign.es)
"""
AVISO_LEGAL = """\
Este cuadro de mando es un trabajo académico de elaboración propia, con fines exclusivamente
informativos. No es una publicación oficial del INE ni del IGN, que no lo respaldan. La exactitud de
la información derivada de sus datos no es atribuible a ninguno de ellos.

Los datos, los indicadores, las proyecciones y la clasificación de municipios se ofrecen tal cual,
sin garantías de ningún tipo, tampoco de exactitud, exhaustividad ni actualización. Las proyecciones
son estimaciones de un modelo estadístico, no previsiones oficiales.

El autor no se hace responsable de los errores que puedan contener ni de las decisiones o los daños
que se deriven de su uso. Quien los utilice lo hace bajo su propia responsabilidad. Para cualquier
uso oficial, consulte las fuentes originales.
"""


####################################################################
### CATÁLOGO DE INDICADORES (solo p_ y ratios, comparables entre niveles)
####################################################################

INDICADORES = {
    "Estructura": [
        ("densidad_pob", "Densidad de población", "dens"),
        ("tasa_dep", "Tasa de dependencia", "pct"),
        ("ind_env", "Índice de envejecimiento", "pct"),
        ("p_mujeres", "Mujeres", "pct"),
    ],
    "Edad": [
        ("p_ed_0_15", "Menores de 16 años", "pct"),
        ("p_ed_16_64", "De 16 a 64 años", "pct"),
        ("p_ed_65+", "65 años o más", "pct"),
    ],
    "Lugar de nacimiento": [
        ("p_n_en_mun", "En el mismo municipio", "pct"),
        ("p_n_dif_mun", "En otro municipio de la provincia", "pct"),
        ("p_n_dif_prov", "En otra provincia de la comunidad", "pct"),
        ("p_n_dif_ca", "En otra comunidad autónoma", "pct"),
        ("p_n_ext", "En el extranjero", "pct"),
    ],
    "Migraciones (tasa anual)": [
        ("p_altas_int", "Altas interiores", "pm"),
        ("p_altas_ext", "Altas exteriores", "pm"),
        ("p_bajas_int", "Bajas interiores", "pm"),
        ("p_bajas_ext", "Bajas exteriores", "pm"),
        ("p_saldo_int", "Saldo migratorio interior", "pm"),
        ("p_saldo_ext", "Saldo migratorio exterior", "pm"),
    ],
    "Movimiento natural (tasa anual)": [
        ("p_mnp_nacimientos", "Natalidad", "pm"),
        ("p_mnp_fallecidos", "Mortalidad", "pm"),
        ("p_mnp_crecimiento_veg", "Crecimiento vegetativo", "pm"),
        ("p_mnp_matrimonios", "Nupcialidad", "pm"),
    ],
    "Educación (15 y más años)": [
        ("p_edu_primaria_inf", "Primaria o inferior", "pct"),
        ("p_edu_secundaria_1", "Secundaria, 1.ª etapa", "pct"),
        ("p_edu_secundaria_2", "Secundaria, 2.ª etapa", "pct"),
        ("p_edu_superior", "Superior", "pct"),
    ],
    "Actividad (16 y más años)": [
        ("tasa_actividad", "Tasa de actividad", "pct"),
        ("tasa_paro", "Tasa de paro", "pct"),
        ("p_act_ocupado", "Ocupados", "pct"),
        ("p_act_parado", "Parados", "pct"),
        ("p_act_inactivo", "Inactivos", "pct"),
    ],
    "Situación profesional": [
        ("p_sp_cuenta_propia", "Por cuenta propia", "pct"),
        ("p_sp_cuenta_ajena_y_otros", "Por cuenta ajena y otras", "pct"),
    ],
    "Rama de actividad": [
        ("p_ae_agricultura_ganaderia_pesca", "Agricultura, ganadería y pesca", "pct"),
        ("p_ae_industria", "Industria", "pct"),
        ("p_ae_construccion", "Construcción", "pct"),
        ("p_ae_servicios", "Servicios", "pct"),
        ("p_ae_no_consta", "No consta", "pct"),
    ],
}
for _n in (5, 10):
    INDICADORES[f"Flujos acumulados ({_n} años)"] = [
        (f"p_saldo_int_{_n}", "Saldo migratorio interior", "pct"),
        (f"p_saldo_ext_{_n}", "Saldo migratorio exterior", "pct"),
        (f"p_suma_altas_int_{_n}", "Altas interiores", "pct"),
        (f"p_suma_altas_ext_{_n}", "Altas exteriores", "pct"),
        (f"p_suma_bajas_int_{_n}", "Bajas interiores", "pct"),
        (f"p_suma_bajas_ext_{_n}", "Bajas exteriores", "pct"),
        (f"p_suma_mnp_nacimientos_{_n}", "Nacimientos", "pct"),
        (f"p_suma_mnp_fallecidos_{_n}", "Defunciones", "pct"),
        (f"p_suma_mnp_crecimiento_veg_{_n}", "Crecimiento vegetativo", "pct"),
    ]

TEMA_CAMBIO = "Cambio de población"
CAMBIO = [
    ("p_cambio_total_{h}_anual_constante", "Tasa anual compuesta", "pct2"),
    ("p_cambio_total_{h}", "Cambio acumulado", "pct"),
    ("p_cambio_total_{h}_anual_medio", "Tasa anual media (punto medio)", "pct2"),
]
HORIZONTES = (5, 10, 20)
# Columnas *_prev de 2_calculos: sufijo -> años de desfase respecto a la fila
PREV = {5: {"": 0, "_prev_1": 5, "_prev_2": 10, "_prev_3": 15}, 10: {"": 0, "_prev": 10}, 20: {"": 0}}

TEMA_CLUSTERING = "Variables usadas para formar los grupos"
VARIABLES_CLUSTERING = [  # entradas de 3_clustering (col_sel), con los saldos ya calculados
    ("total", "Población", "int"),
    ("densidad_pob", "Densidad de población", "dens"),
    ("p_n_en_mun", "Nacidos en el mismo municipio", "pct"),
    ("p_ed_0_15", "Menores de 16 años", "pct"),
    ("p_ed_65+", "65 años o más", "pct"),
    ("p_cambio_total_10", "Cambio de población en 10 años", "pct"),
    ("p_cambio_total_20", "Cambio de población en 20 años", "pct"),
    ("p_saldo_int_10", "Saldo migratorio interior (10 años)", "pct"),
    ("p_saldo_ext_10", "Saldo migratorio exterior (10 años)", "pct"),
    ("p_suma_mnp_crecimiento_veg_10", "Crecimiento vegetativo (10 años)", "pct"),
]
ETIQUETAS_RADIAL = {  # versiones cortas para los ejes del gráfico radial
    "total": "Población", "densidad_pob": "Densidad", "p_n_en_mun": "Nacidos en el municipio",
    "p_ed_0_15": "Menores de 16", "p_ed_65+": "65 y más", "p_cambio_total_10": "Cambio 10 años",
    "p_cambio_total_20": "Cambio 20 años", "p_saldo_int_10": "Saldo interior",
    "p_saldo_ext_10": "Saldo exterior", "p_suma_mnp_crecimiento_veg_10": "Crec. vegetativo",
}
ESTADISTICOS = {"Mediana": "median", "Media": "mean", "Agregado": "agregado"}

TEMAS_DETALLE = ["Edad", "Lugar de nacimiento", "Educación (15 y más años)",
                 "Actividad (16 y más años)", "Situación profesional", "Rama de actividad"]


def es_divergente(col):
    return any(s in col for s in ("saldo", "crecimiento_veg", "cambio"))


def fmt(v, tipo="int", decimales=None, unidad=True):
    """Formato español: 1.234,5."""
    if v is None or pd.isna(v):
        return "sin dato"
    if np.isinf(v):  # p. ej. índice de envejecimiento sin menores de 16 años
        return "∞" if v > 0 else "−∞"
    if decimales is None:
        decimales = {"pct": 1, "pct2": 2, "pm": 1, "dens": 1, "int": 0}[tipo]
    num = f"{v * FACTOR[tipo]:,.{decimales}f}".translate(str.maketrans(",.", ".,"))
    return num if tipo == "int" or not unidad else f"{num} {UNIDAD[tipo]}"


def pct_txt(v):
    """Porcentaje con signo: +12,3 %."""
    return f"{v:+.1f} %".replace(".", ",")


def fmt_delta(x):
    return None if pd.isna(x) else pct_txt(100 * x)


####################################################################
### DERIVACIÓN DE INDICADORES
####################################################################

FLUJOS = ["altas_int", "altas_ext", "bajas_int", "bajas_ext",
          "mnp_nacimientos", "mnp_fallecidos", "mnp_crecimiento_veg"]
SOBRE_TOTAL = ["hombres", "mujeres", "n_ext", "n_dif_ca", "n_dif_prov", "n_dif_mun", "n_en_mun",
               "ed_65+", "ed_16_64", "ed_0_15", "altas_ext", "altas_int", "bajas_ext", "bajas_int",
               "mnp_crecimiento_veg", "mnp_nacimientos", "mnp_fallecidos", "mnp_matrimonios"]
GRUPOS_DEN = {  # proporciones sobre la suma de categorías, como en 2_calculos
    "edu": ["edu_primaria_inf", "edu_secundaria_1", "edu_secundaria_2", "edu_superior"],
    "act": ["act_inactivo", "act_ocupado", "act_parado"],
    "sp": ["sp_cuenta_propia", "sp_cuenta_ajena_y_otros"],
    "ae": ["ae_agricultura_ganaderia_pesca", "ae_construccion", "ae_industria",
           "ae_servicios", "ae_no_consta"],
}
CONTEOS = SOBRE_TOTAL + sum(GRUPOS_DEN.values(), []) + ["total", "superficie"]


def columnas_cambio(h):
    return [f"cambio_total_{h}", f"p_cambio_total_{h}", f"cambio_total_{h}_anual",
            f"p_cambio_total_{h}_anual_constante", f"p_cambio_total_{h}_anual_medio"]


# Réplica de 2_calculos, para el agregado autonómico y los grupos

def calcular_prev(df):
    """Columnas *_prev como en 2_calculos. Solo hacen falta para el agregado autonómico."""
    g = df.groupby("cod_mun")
    nuevas = {base + suf: g[base].shift(desfase)
              for h, sufijos in PREV.items() for base in columnas_cambio(h)
              for suf, desfase in sufijos.items() if suf}
    return pd.concat([df, pd.DataFrame(nuevas)], axis=1)


def add_cambio_n(df, col, n):
    df = df.sort_values(["cod_mun", "periodo"])
    g = df.groupby("cod_mun")[col]
    ini = g.shift(n)
    c = f"cambio_{col}_{n}"
    df[c] = g.diff(n)
    df[f"p_{c}"] = df[c] / ini
    df[f"{c}_anual"] = df[c] / n
    df[f"p_{c}_anual_constante"] = (df[col] / ini) ** (1 / n) - 1
    df[f"p_{c}_anual_medio"] = df[c] / ((df[col] + ini) / 2) / n
    return df


def add_suma_n(df, col, n):
    df = df.sort_values(["cod_mun", "periodo"])
    s = f"suma_{col}_{n}"
    df[s] = df.groupby("cod_mun")[col].transform(lambda x: x.rolling(n).sum().shift(1))
    df[f"p_{s}"] = df[s] / df.groupby("cod_mun")["total"].shift(n)
    return df


def proporciones(df):
    nuevas = {f"p_{c}": df[c] / df["total"] for c in SOBRE_TOTAL}
    for cols in GRUPOS_DEN.values():
        den = df[cols].sum(axis=1, min_count=1)
        nuevas.update({f"p_{c}": df[c] / den for c in cols})
    return pd.concat([df, pd.DataFrame(nuevas)], axis=1)


def agregar(df, claves, min_count=1):
    """Suma los conteos por grupo (serie `claves`) y año y recalcula los indicadores. El grupo
    queda en cod_mun, para reutilizar las funciones de 2_calculos."""
    ag = (df.groupby([claves.rename("cod_mun"), "periodo"])[CONTEOS].sum(min_count=min_count)
          .reset_index())
    ag["densidad_pob"] = ag["total"] / ag["superficie"]
    ag["tasa_dep"] = (ag["ed_0_15"] + ag["ed_65+"]) / ag["ed_16_64"]
    ag["ind_env"] = ag["ed_65+"] / ag["ed_0_15"]
    for n in (5, 10, 20):
        ag = add_cambio_n(ag, "total", n)
    for n in (5, 10):
        for c in FLUJOS:
            ag = add_suma_n(ag, c, n)
    ag = proporciones(calcular_prev(ag))
    # Saldos y tasas de actividad y de paro, como en las secciones 3.6, 3.7 y 4 de 2_calculos
    activos = ag["act_ocupado"] + ag["act_parado"]
    nuevas = {"p_saldo_int": ag["p_altas_int"] - ag["p_bajas_int"],
              "p_saldo_ext": ag["p_altas_ext"] - ag["p_bajas_ext"],
              "tasa_actividad": activos / (activos + ag["act_inactivo"]),
              "tasa_paro": ag["act_parado"] / activos}
    for n in (5, 10):
        for t in ("int", "ext"):
            nuevas[f"p_saldo_{t}_{n}"] = ag[f"p_suma_altas_{t}_{n}"] - ag[f"p_suma_bajas_{t}_{n}"]
    return pd.concat([ag, pd.DataFrame(nuevas)], axis=1).copy()


def agregado_autonomico(df):
    """Castilla-La Mancha como suma de sus provincias. Si falta una provincia, queda sin dato."""
    prov = df[df["cod_mun"].isin(list(PROVINCIAS))]
    return (agregar(prov, pd.Series(COD_CA, index=prov.index), min_count=len(PROVINCIAS))
            .assign(cod_prov=COD_CA, nombre=NOMBRE_CA))

# Fin de la réplica de 2_calculos


####################################################################
### CARGA DE DATOS
####################################################################
# cache_resource evita copiar ~30 MB en cada interacción: los datos no se modifican nunca.

def leer_clusters(path):
    """cod_mun -> clúster, de la salida de 3_clustering (vacío si no existe)."""
    if not Path(path).exists():
        return pd.Series(dtype=int)
    c = pd.read_csv(path, usecols=["cod_mun", "cluster"], dtype={"cod_mun": str}).dropna()
    return c.drop_duplicates("cod_mun").set_index("cod_mun")["cluster"].astype(int)


@st.cache_resource
def cargar_panel(path, path_clusters):
    df = pd.read_csv(path, dtype={"cod_mun": str, "cod_prov": str})
    df = df[df["cod_mun"].str.len().isin([2, 5])]
    df = pd.concat([df, agregado_autonomico(df)], ignore_index=True).copy()  # copy(): desfragmenta
    largo = df["cod_mun"].str.len()
    df["nivel"] = np.select([largo == 5, largo == 2], [NIVEL_MUN, NIVEL_PROV], NIVEL_CA)
    df["cluster"] = df["cod_mun"].map(leer_clusters(path_clusters)).astype("Int64")
    con_cluster = df[df["cluster"].notna()]
    ag_cl = (agregar(con_cluster, con_cluster["cluster"]).rename(columns={"cod_mun": "cluster"})
             if len(con_cluster) else None)
    return df, registro_territorios(df), ag_cl


def registro_territorios(df):
    """Nombre más reciente, nivel y etiquetas de cada territorio."""
    t = df.sort_values("periodo").groupby("cod_mun")[["nombre", "cod_prov", "nivel", "cluster"]].last()
    prov = t["cod_prov"].map(PROVINCIAS)
    es_mun, es_prov = t["nivel"] == NIVEL_MUN, t["nivel"] == NIVEL_PROV
    codigo = t.index.to_series()
    t["provincia"] = prov
    t["corta"] = np.select([es_mun, es_prov], [t["nombre"], "Provincia de " + prov], NOMBRE_CA)
    t["busqueda"] = np.select(
        [es_mun, es_prov],
        [codigo + " " + t["nombre"] + " (" + prov + ")", codigo + " Provincia de " + prov],
        NOMBRE_CA,
    )
    municipios = t[es_mun].sort_values("corta", key=lambda s: s.str.normalize("NFKD")).index
    orden = [COD_CA, *sorted(t[es_prov].index), *municipios]
    return t.loc[orden]


@st.cache_resource
def cargar_proyecciones(path):
    """Series largas de proyección (valor central). Provincias y comunidad: suma municipal."""
    pred = pd.read_csv(path, dtype={"cod_mun": str, "cod_prov": str})
    cols = [c for c in pred.columns if re.fullmatch(r"pob_\d{4}", c)]
    largo = pred.melt(id_vars=["cod_mun", "cod_prov"], value_vars=cols,
                      var_name="periodo", value_name="total")
    largo["periodo"] = largo["periodo"].str[4:].astype(int)
    prov = (largo.groupby(["cod_prov", "periodo"], as_index=False)["total"].sum()
            .rename(columns={"cod_prov": "cod_mun"}))
    ca = largo.groupby("periodo", as_index=False)["total"].sum().assign(cod_mun=COD_CA)
    return pd.concat([largo[["cod_mun", "periodo", "total"]], prov, ca], ignore_index=True)


def simplificar(geoms):
    """Simplifica en metros manteniendo encajadas las fronteras compartidas (Shapely >= 2.1)."""
    arr = np.asarray(geoms.to_crs(25830))
    try:
        arr = shapely.coverage_simplify(arr, TOLERANCIA_M)
    except Exception:
        arr = shapely.simplify(arr, TOLERANCIA_M, preserve_topology=True)
    wgs84 = gpd.GeoSeries(arr, index=geoms.index, crs=25830).to_crs(4326)
    return gpd.GeoSeries(shapely.transform(np.asarray(wgs84), lambda c: np.round(c, 5)),
                         index=geoms.index, crs=4326)


@st.cache_resource
def cargar_geo(path, nivel, codigos):
    """GeoJSON simplificado con una propiedad «cod» por silueta, listo para pydeck."""
    gdf = gpd.read_parquet(path)
    if nivel == NIVEL_CA:
        gdf = gdf.assign(cod=COD_CA)
    else:
        gdf = gdf.rename(columns={COLUMNA_CODIGO[nivel]: "cod"})
        gdf = gdf[gdf["cod"].isin(codigos)]
        if faltan := sorted(set(codigos) - set(gdf["cod"])):
            logging.warning("%s sin silueta: %s", nivel, faltan)
    gdf["geometry"] = simplificar(gdf.geometry)
    return json.loads(gdf[["cod", "geometry"]].to_json())


####################################################################
### NAVEGACIÓN
####################################################################

def abrir(cod):
    st.session_state.cod = cod
    st.session_state.vista = "detalle"


def volver():
    st.session_state.vista = "mapa"
    st.session_state.buscador = None
    st.session_state.nonce += 1  # clave nueva para el mapa: se descarta la selección anterior


def al_buscar():
    if st.session_state.buscador:
        abrir(st.session_state.buscador)


####################################################################
### MAPA
####################################################################

def paso_redondo(rango, n):
    """Paso de 1, 2, 2,5 o 5 x 10^k que divide el rango en unas n partes."""
    bruto = rango / n
    mag = 10 ** np.floor(np.log10(bruto))
    return next((m * mag for m in (1, 2, 2.5, 5, 10) if bruto <= m * mag * (1 + 1e-9)), 10 * mag)


def escala(valores, divergente):
    """Dominio de color (percentiles 2-98, extremos redondeados) y paso de la leyenda.
    Los indicadores divergentes quedan centrados en 0."""
    v = pd.Series(valores, dtype=float)
    v = v[np.isfinite(v)]  # los infinitos (división entre 0) se pintan con el color del extremo
    if v.empty:
        return 0.0, 1.0, 0.25
    lo, hi = v.quantile([0.02, 0.98]) if len(v) > 20 else (v.min(), v.max())
    if divergente:
        m = max(abs(lo), abs(hi)) or 1e-6
        paso = paso_redondo(m, 4)
        m = np.ceil(m / paso - 1e-9) * paso
        return -m, m, paso
    if hi - lo < 1e-9:
        lo, hi = lo - max(abs(lo) * 0.05, 1e-6), hi + max(abs(hi) * 0.05, 1e-6)
    paso = paso_redondo(hi - lo, 8)
    return np.floor(lo / paso + 1e-9) * paso, np.ceil(hi / paso - 1e-9) * paso, paso


def decimales_paso(paso, tipo):
    """Decimales justos para que las marcas de la leyenda se lean sin redondeos engañosos."""
    p = paso * FACTOR[tipo]
    exp = int(np.floor(np.log10(p) + 1e-9))
    return max(0, -exp + (1 if round(p / 10 ** exp, 3) == 2.5 else 0))


def _rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)])


def colorear(v, lo, hi, paleta, alfa=215):
    if v is None or pd.isna(v):
        return SIN_DATO
    t = min(max((v - lo) / (hi - lo), 0.0), 1.0) * (len(paleta) - 1)
    i = min(int(t), len(paleta) - 2)
    a, b = _rgb(paleta[i]), _rgb(paleta[i + 1])
    return [*(a + (b - a) * (t - i)).round().astype(int).tolist(), alfa]


def features(geo, propiedades):
    """Las siluetas de `geo` con las propiedades que devuelve propiedades(cod)."""
    return [{"type": "Feature", "geometry": f["geometry"], "properties": propiedades(str(f["properties"]["cod"]))}
            for f in geo["features"]]


def capa_geo(data, borde, ancho, relleno=None, **k):
    """Capa GeoJSON de pydeck. Sin `relleno`, dibuja solo el contorno."""
    if relleno is not None:
        k["get_fill_color"] = relleno
    return pdk.Layer("GeoJsonLayer", data=data, stroked=True, filled=relleno is not None,
                     get_line_color=borde, line_width_min_pixels=ancho, **k)


def mapa_clicable(geos, feats, ancho, tooltip, limites, alto, clave):
    """Dibuja los territorios coloreados (y los límites provinciales) y devuelve el código pulsado."""
    capas = [capa_geo({"type": "FeatureCollection", "features": feats}, [255, 255, 255, 150], ancho,
                      "properties.color", id=ID_CAPA, pickable=True, auto_highlight=True,
                      highlight_color=[0, 0, 0, 70])]
    if limites:
        capas.append(capa_geo(geos[NIVEL_PROV], [60, 60, 60, 220], 1.2, id="limites_provinciales"))
    deck = pdk.Deck(layers=capas, initial_view_state=pdk.ViewState(**VISTA_INICIAL), map_style="light",
                    tooltip={"html": tooltip, "style": ESTILO_TOOLTIP})
    evento = st.pydeck_chart(deck, height=alto, on_select="rerun", selection_mode="single-object",
                             key=f"{clave}_{st.session_state.nonce}")
    try:
        objetos = evento.selection["objects"].get(ID_CAPA, [])
    except (AttributeError, KeyError, TypeError):
        return None
    return str(objetos[0].get("properties", objetos[0]).get("cod")) if objetos else None


def ir_a_ficha(cod, terr):
    if cod in terr.index:
        abrir(cod)
        st.rerun()


def leyenda(lo, hi, paso, tipo, valores, alto=460):
    """Barra de color vertical con marcas numéricas. ≤ y ≥ indican que hay valores
    fuera de la escala, pintados con el color del extremo."""
    v = pd.Series(valores, dtype=float)
    d = decimales_paso(paso, tipo)
    marcas = []
    for x in np.arange(lo, hi + paso / 2, paso):
        texto = fmt(0.0 if abs(x) < paso * 1e-6 else x, tipo, d, unidad=False)
        if np.isclose(x, hi) and (v > hi).any():
            texto = "≥ " + texto
        elif np.isclose(x, lo) and (v < lo).any():
            texto = "≤ " + texto
        marcas.append(
            f'<div style="position:absolute;left:16px;bottom:{100 * (x - lo) / (hi - lo):.2f}%;'
            f'transform:translateY(50%);white-space:nowrap;">'
            f'<span style="display:inline-block;width:5px;margin-right:4px;vertical-align:middle;'
            f'border-top:1px solid currentColor;"></span>{texto}</div>')
    hueco = ('<div style="margin-top:14px;white-space:nowrap;"><span style="display:inline-block;'
             'width:14px;height:10px;margin-right:6px;vertical-align:middle;'
             'background:rgba(90,90,90,.67);"></span>sin dato</div>') if v.isna().any() else ""
    return (f'<div style="font-size:0.78rem;line-height:1;padding-top:6px;">'
            f'<div style="margin-bottom:12px;opacity:.75;">{UNIDAD[tipo]}</div>'
            f'<div style="position:relative;height:{alto}px;">'
            f'<div style="position:absolute;left:0;top:0;bottom:0;width:14px;border-radius:2px;'
            f'background:linear-gradient(to top,{",".join(COOLWARM)});"></div>'
            f'{"".join(marcas)}</div>{hueco}</div>')


def ventanas(df_nivel, plantilla, h, ref):
    """Periodos disponibles en la fila del año de referencia, del más antiguo al más reciente:
    {"2005–2010": columna, …}."""
    fila = df_nivel[df_nivel["periodo"] == ref]
    cols = {ref - d: plantilla.format(h=h) + suf for suf, d in PREV[h].items()}
    return {f"{fin - h}–{fin}": c for fin, c in sorted(cols.items()) if c in fila and fila[c].notna().any()}


def deslizador(etiqueta, opciones, **k):
    """select_slider que admite una sola opción (Streamlit no la dibuja)."""
    return st.select_slider(etiqueta, opciones, value=opciones[-1], **k) if len(opciones) > 1 else opciones[0]


def marco(titulo, icono, nota=None):
    """Recuadro con título e icono para agrupar controles relacionados de la barra lateral."""
    caja = st.container(border=True)
    caja.markdown(f":material/{icono}: **{titulo}**")
    if nota:
        caja.caption(nota)
    return caja


def vista_mapa(df, terr, geos):
    ref = int(df["periodo"].max())
    aviso = None

    with st.sidebar:
        st.header("Mapa")

        with marco("Nivel territorial", "location_on", "Qué territorios se colorean en el mapa."):
            nivel = st.radio("Nivel territorial", list(GEO_PATHS), label_visibility="collapsed")
        df_nivel = df[df["nivel"] == nivel]

        with marco("Variable", "bar_chart", "Elige un tema y, dentro de él, el indicador."):
            tema = st.selectbox("Tema", [TEMA_CAMBIO, *INDICADORES])
            if tema == TEMA_CAMBIO:
                plantilla, etiqueta, tipo = st.selectbox("Indicador", CAMBIO, format_func=lambda x: x[1])
                centrado_natural = True
            else:
                col, etiqueta, tipo = st.selectbox("Indicador", INDICADORES[tema], format_func=lambda x: x[1])
                centrado_natural = es_divergente(col)

        nota_tiempo = "Cuántos años abarca el cambio y qué periodo se muestra." if tema == TEMA_CAMBIO else None
        with marco("Tiempo", "calendar_month", nota_tiempo):
            if tema == TEMA_CAMBIO:
                h = st.radio("Horizonte", HORIZONTES, index=1, horizontal=True,
                             format_func=lambda n: f"{n} años")
                opciones = ventanas(df_nivel, plantilla, h, ref)
                if not opciones:
                    aviso = f"No hay datos de «{etiqueta}» a {h} años para el nivel {nivel.lower()}."
                else:
                    periodo_txt = deslizador(
                        "Periodo", list(opciones),
                        help="Ventanas seguidas de la misma duración. Compáralas para ver si el "
                             "cambio se acelera o se frena.")
                    anio, col = ref, opciones[periodo_txt]
            else:
                anios = sorted(df_nivel.loc[df_nivel[col].notna(), "periodo"].unique().tolist())
                if not anios:
                    aviso = f"No hay datos de «{etiqueta}» para el nivel {nivel.lower()}."
                else:
                    anio = deslizador("Año", anios)
                    periodo_txt = str(anio)
            if aviso:
                st.caption("Sin datos para esta combinación.")
            elif (len(opciones) if tema == TEMA_CAMBIO else len(anios)) == 1:
                st.caption(f"Solo hay datos de {periodo_txt}.")

        with marco("Escala de color", "palette"):
            escala_comun = st.toggle(
                "Mantener la escala municipal", value=True,
                help="Usa la misma escala que el mapa de municipios, así un color significa el mismo "
                     "valor en los dos mapas. Desactívalo para ver mejor las diferencias entre provincias.",
            ) if nivel == NIVEL_PROV else True
            centrar = st.toggle(
                "Centrar la escala en 0", value=centrado_natural,
                help="Pone el 0 en el color central, con azul por debajo y rojo por encima. Viene "
                     "activado en los indicadores que pueden ser negativos.",
            )

    st.title("Población de Castilla-La Mancha")
    if aviso:
        st.info(aviso + " Elige otro nivel territorial o indicador.")
        return

    datos = df_nivel[df_nivel["periodo"] == anio].set_index("cod_mun")
    mun = df[(df["nivel"] == NIVEL_MUN) & (df["periodo"] == anio)]
    ca = df[(df["cod_mun"] == COD_CA) & (df["periodo"] == anio)]
    base_escala = mun[col] if (escala_comun or nivel == NIVEL_CA) else datos[col]
    lo, hi, paso = escala(base_escala, centrar)

    st.subheader(f"{etiqueta}, {periodo_txt}")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Valor en {NOMBRE_CA}", fmt(ca[col].iloc[0] if len(ca) else np.nan, tipo))
    c2.metric("Mediana de los municipios", fmt(mun[col].median(), tipo))
    c3.metric(f"{PLURAL[nivel]} sin dato", f"{int(datos[col].isna().sum())} de {len(datos)}")

    valores, poblacion = datos[col].to_dict(), datos["total"].to_dict()
    feats = features(geos[nivel], lambda cod: {
        "cod": cod, "nombre": terr["corta"].get(cod, cod), "valor": fmt(valores.get(cod), tipo),
        "pob": fmt(poblacion.get(cod)), "color": colorear(valores.get(cod), lo, hi, COOLWARM)})
    tooltip = (f"<b>{{nombre}}</b><br/>{etiqueta}: <b>{{valor}}</b><br/>Población {anio}: {{pob}}<br/>"
               "<i>Clic para abrir la ficha</i>")
    col_mapa, col_leyenda = st.columns([10, 1])
    with col_mapa:
        cod = mapa_clicable(geos, feats, 0.4 if nivel == NIVEL_MUN else 1, tooltip, nivel == NIVEL_MUN, 620, "mapa")
    col_leyenda.markdown(leyenda(lo, hi, paso, tipo, datos[col]), unsafe_allow_html=True)
    ir_a_ficha(cod, terr)

    rango = ("del valor mínimo al máximo de las provincias" if nivel == NIVEL_PROV and not escala_comun
             else "del percentil 2 al 98 de los municipios")
    st.caption("Haz clic en un territorio o búscalo en la barra lateral para abrir su ficha. "
               f"La escala va {rango}. Los valores que quedan fuera toman el color del extremo, "
               "y la leyenda lo indica con ≤ o ≥.")

    with st.expander("Ver los datos en tabla"):
        valor = f"{etiqueta} ({UNIDAD[tipo]})"
        tabla = pd.DataFrame({
            "Código": datos.index,
            "Territorio": terr.loc[datos.index, "corta"].to_numpy(),
            "Provincia": terr.loc[datos.index, "provincia"].to_numpy(),
            f"Población {anio}": datos["total"].to_numpy(),
            valor: datos[col].to_numpy() * FACTOR[tipo],
        })
        if nivel != NIVEL_MUN:
            tabla = tabla.drop(columns="Provincia")
        st.dataframe(tabla.sort_values(valor, ascending=False), hide_index=True,
                     column_config={valor: st.column_config.NumberColumn(format="%.2f")})


####################################################################
### GRÁFICOS COMUNES
####################################################################

EJE_ANIOS = alt.X("periodo:Q", title=None, axis=alt.Axis(format="d"))
TRAZO_PROYECCION = alt.StrokeDash("tipo:N", title=None, legend=None,  # se explica en el pie del gráfico
                                  scale=alt.Scale(domain=["Observada", "Proyección"], range=[[1, 0], [5, 3]]))


def lineas_anuales(d, campo, titulo, color, tipo, punto, formato):
    """Columna `valor` por año, una línea por `campo`."""
    return alt.Chart(d).mark_line(point=alt.OverlayMarkDef(size=punto)).encode(
        x=EJE_ANIOS, y=alt.Y("valor:Q", title=UNIDAD[tipo], scale=alt.Scale(zero=False)), color=color,
        tooltip=[alt.Tooltip(f"{campo}:N", title=titulo), alt.Tooltip("periodo:Q", title="Año"),
                 alt.Tooltip("valor:Q", title=UNIDAD[tipo], format=formato)])


def anclar(pr, obs, clave):
    """Escala cada proyección para que coincida con el valor observado del año base
    (corrige los huecos de las sumas de proyecciones municipales)."""
    base = int(pr["periodo"].min())
    factor = (obs[obs["periodo"] == base].set_index(clave)["total"]
              / pr[pr["periodo"] == base].set_index(clave)["total"])
    return pr.assign(total=pr["total"] * pr[clave].map(factor))


def grafico_cambio(d, obs, clave, campo, titulo, color, alto):
    """Cuánto ha ganado o perdido cada serie desde el primer año en que todas tienen dato, con la
    proyección en discontinua. Devuelve el gráfico y ese año base.

    El año base es común a todas las series para que sean comparables: suele ser 2000, pero en un
    municipio creado después (Pozo Cañada no existía a 1 de enero de 2000) es su primer año, y su
    provincia y la comunidad se miden desde ese mismo año."""
    anio0 = int(obs.groupby(clave)["periodo"].min().max())
    base = obs[obs["periodo"] == anio0].set_index(clave)["total"]
    d = d[d["periodo"] >= anio0]
    d = d.assign(cambio=100 * (d["total"] / d[clave].map(base) - 1))
    d["cambio_txt"] = d["cambio"].map(pct_txt)
    lineas = alt.Chart(d.dropna(subset=["cambio"])).mark_line().encode(
        x=EJE_ANIOS,
        y=alt.Y("cambio:Q", title=f"Ganancia o pérdida de población desde {anio0}",
                axis=alt.Axis(labelExpr="(datum.value > 0 ? '+' : '') + datum.value + ' %'")),
        color=color, strokeDash=TRAZO_PROYECCION,
        tooltip=[alt.Tooltip(f"{campo}:N", title=titulo), alt.Tooltip("periodo:Q", title="Año"),
                 alt.Tooltip("cambio_txt:N", title=f"Cambio desde {anio0}")],
    )
    return (lineas + regla_cero()).properties(height=alto), anio0


####################################################################
### FICHA DE TERRITORIO
####################################################################

def ancestros(cod, terr):
    nivel = terr.at[cod, "nivel"]
    if nivel == NIVEL_MUN:
        return [terr.at[cod, "cod_prov"], COD_CA]
    return [COD_CA] if nivel == NIVEL_PROV else []


def color_terr(etiquetas):
    return alt.Color("territorio:N", title=None, sort=etiquetas,
                     scale=alt.Scale(domain=etiquetas, range=COLORES_TERR[:len(etiquetas)]),
                     legend=alt.Legend(orient="right"))


def serie_proyectada(proy, obs, comparar):
    if proy is None:
        return pd.DataFrame(columns=["cod_mun", "periodo", "total"])
    p = proy[proy["cod_mun"].isin(comparar)]
    return anclar(p, obs, "cod_mun") if len(p) else p.copy()


def regla_cero(discontinua=False):
    return alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        color="#888", strokeDash=[2, 2] if discontinua else [1, 0]).encode(y=alt.datum(0))


def grafico_evolucion(obs, proy, cod, etiquetas_cod):
    datos = pd.concat([obs.assign(tipo="Observada"), proy.assign(tipo="Proyección")], ignore_index=True)
    datos["territorio"] = datos["cod_mun"].map(etiquetas_cod)
    absoluta = alt.Chart(datos[datos["cod_mun"] == cod]).mark_line(point=alt.OverlayMarkDef(size=14)).encode(
        x=EJE_ANIOS, y=alt.Y("total:Q", title="Habitantes", scale=alt.Scale(zero=False)),
        strokeDash=TRAZO_PROYECCION, color=alt.value(COLORES_TERR[0]),
        tooltip=[alt.Tooltip("periodo:Q", title="Año"), alt.Tooltip("tipo:N", title="Serie"),
                 alt.Tooltip("total:Q", title="Habitantes", format=",.0f")],
    ).properties(height=300)
    relativa, anio0 = grafico_cambio(datos, obs, "cod_mun", "territorio", "Territorio",
                                     color_terr(list(etiquetas_cod.values())), 300)
    return absoluta, relativa, anio0


def grafico_ventanas(serie, comparar, etiquetas_cod, h, ref):
    filas = []
    for c in comparar:
        fila = serie[(serie["cod_mun"] == c) & (serie["periodo"] == ref)]
        if fila.empty:
            continue
        for suf, desfase in PREV[h].items():
            col = f"p_cambio_total_{h}_anual_constante{suf}"
            if col in fila:
                filas.append({"territorio": etiquetas_cod[c], "fin": ref - desfase,
                              "periodo": f"{ref - desfase - h}–{ref - desfase}",
                              "valor": 100 * fila[col].iloc[0]})
    d = pd.DataFrame(filas).dropna(subset=["valor"])
    if d.empty:
        return None
    orden = d.sort_values("fin")["periodo"].unique().tolist()
    etiquetas = list(etiquetas_cod.values())
    barras = alt.Chart(d).mark_bar().encode(
        x=alt.X("periodo:N", sort=orden, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("territorio:N", sort=etiquetas),
        y=alt.Y("valor:Q", title="Tasa anual compuesta (%)"),
        color=color_terr(etiquetas),
        tooltip=[alt.Tooltip("territorio:N", title="Territorio"), alt.Tooltip("periodo:N", title="Periodo"),
                 alt.Tooltip("valor:Q", title="% anual", format=".2f")],
    )
    return (barras + regla_cero()).properties(height=280)


def grafico_lineas(serie, col, tipo, etiquetas):
    d = serie[["territorio", "periodo", col]].dropna().rename(columns={col: "valor"})
    d["valor"] = d["valor"] * FACTOR[tipo]
    lineas = lineas_anuales(d, "territorio", "Territorio", color_terr(etiquetas), tipo, 14, ".2f").properties(height=260)
    return lineas + regla_cero(discontinua=True) if es_divergente(col) else lineas


def bloque_tema(serie, cod, tema, etiquetas):
    inds = INDICADORES[tema]
    cols = [c for c, _, _ in inds]
    propia = serie[serie["cod_mun"] == cod].dropna(subset=cols, how="all")
    if propia.empty:
        st.info("No hay datos de este tema para este territorio. Desde 2021, el censo no publica la "
                "educación ni la actividad de los municipios de menos de 50 habitantes.")
        return
    anio = int(propia["periodo"].max())
    nombres = {c: e for c, e, _ in inds}
    d = (serie[serie["periodo"] == anio]
         .melt(id_vars="territorio", value_vars=cols, var_name="col", value_name="valor")
         .dropna(subset=["valor"]))
    d["indicador"] = d["col"].map(nombres)
    d["valor"] = 100 * d["valor"]  # todos los temas de la ficha son porcentajes
    d["valor_txt"] = d["valor"].map(lambda v: f"{v:.1f}".replace(".", ","))

    base = alt.Chart(d).encode(
        x=alt.X("indicador:N", sort=list(nombres.values()), title=None,
                axis=alt.Axis(labelAngle=0, labelLimit=260, labelFontSize=12)),
        xOffset=alt.XOffset("territorio:N", sort=etiquetas),
        y=alt.Y("valor:Q", title=f"% en {anio}", scale=alt.Scale(nice=True)),
        tooltip=[alt.Tooltip("territorio:N", title="Territorio"), alt.Tooltip("indicador:N", title="Indicador"),
                 alt.Tooltip("valor_txt:N", title="%")],
    )
    barras = base.mark_bar().encode(color=color_terr(etiquetas))
    valores = base.mark_text(dy=-7, fontSize=11, color="#9aa0a6").encode(text="valor_txt:N")
    st.altair_chart((barras + valores).properties(height=360))

    sel, _ = st.columns([1, 2])
    c, e, t = sel.selectbox("Evolución de", inds, format_func=lambda x: x[1], key=f"evo_{tema}")
    st.altair_chart(grafico_lineas(serie, c, t, etiquetas))


def mapa_situacion(geos, cod, nivel, lonlat):
    """La comunidad en gris con el territorio de la ficha resaltado. Sin interacción."""
    rgb = _rgb(COLORES_TERR[0]).tolist()
    capas = [capa_geo(geos[NIVEL_CA], [100, 100, 100, 220], 1, [*rgb, 235] if nivel == NIVEL_CA else [128, 128, 128, 45]),
             capa_geo(geos[NIVEL_PROV], [100, 100, 100, 150], 0.8)]
    if nivel != NIVEL_CA:
        propio = [f for f in geos[nivel]["features"] if str(f["properties"]["cod"]) == cod]
        capas.append(capa_geo({"type": "FeatureCollection", "features": propio}, [255, 255, 255, 230], 1, [*rgb, 235]))
    if nivel == NIVEL_MUN and lonlat is not None:  # anillo: los municipios pequeños se ven a esta escala
        capas.append(pdk.Layer(
            "ScatterplotLayer", data=[{"pos": lonlat}], get_position="pos", get_radius=0,
            radius_min_pixels=11, filled=False, stroked=True,
            get_line_color=[*rgb, 255], line_width_min_pixels=2,
        ))
    return pdk.Deck(layers=capas, initial_view_state=pdk.ViewState(**VISTA_LOCALIZADOR),
                    views=[pdk.View(type="MapView", controller=False)], map_style="light")


def vista_detalle(df, terr, proy, cod, geos):
    comparar = [cod] + ancestros(cod, terr)
    etiquetas_cod = {c: terr.at[c, "corta"] for c in comparar}
    etiquetas = list(etiquetas_cod.values())
    serie = df[df["cod_mun"].isin(comparar)].copy()
    serie["territorio"] = serie["cod_mun"].map(etiquetas_cod)
    propia = serie[serie["cod_mun"] == cod].set_index("periodo").sort_index()
    ref = int(propia.index.max())
    nivel = terr.at[cod, "nivel"]

    st.button("Volver al mapa", on_click=volver)
    cabecera, situacion = st.columns([3, 2])
    with situacion:
        lonlat = propia[["lon", "lat"]].dropna() if {"lon", "lat"} <= set(propia.columns) else pd.DataFrame()
        st.pydeck_chart(mapa_situacion(geos, cod, nivel, lonlat.iloc[-1].tolist() if len(lonlat) else None),
                        height=280)
    with cabecera:
        st.title(etiquetas_cod[cod])
        if nivel == NIVEL_MUN:
            grupo = terr.at[cod, "cluster"]
            st.caption(f"Municipio de la provincia de {terr.at[cod, 'provincia']}. Código INE {cod}."
                       + (f" {nombre_cluster(int(grupo))}." if pd.notna(grupo) else ""))
        elif nivel == NIVEL_PROV:
            st.caption(f"Provincia de Castilla-La Mancha. Código INE {cod}.")
        else:
            st.caption("Comunidad autónoma, calculada como suma de sus cinco provincias.")
        (c1, c2), (c3, c4) = st.columns(2), st.columns(2)

    obs = serie[["cod_mun", "periodo", "total"]].dropna()
    pr = serie_proyectada(proy, obs, comparar)
    pob, pob_ant = propia.at[ref, "total"], propia["total"].get(ref - 1, np.nan)
    c1.metric(f"Población {ref}", fmt(pob), fmt_delta(pob / pob_ant - 1),
              help=f"Variación respecto a {ref - 1}.")
    c2.metric("Densidad", fmt(propia.at[ref, "densidad_pob"], "dens"))
    c3.metric(f"Tasa anual {ref - 10}–{ref}", fmt(propia.at[ref, "p_cambio_total_10_anual_constante"], "pct2"))
    propia_pr = pr[pr["cod_mun"] == cod]
    if not propia_pr.empty:
        fin = int(propia_pr["periodo"].max())
        pob_fin = propia_pr.loc[propia_pr["periodo"] == fin, "total"].iloc[0]
        c4.metric(f"Proyección {fin}", fmt(pob_fin), fmt_delta(pob_fin / pob - 1),
                  help=f"Valor central del modelo de Holt. Variación respecto a {ref}.")

    st.subheader("Evolución de la población")
    absoluta, relativa, anio0 = grafico_evolucion(obs, pr, cod, etiquetas_cod)
    izq, der = st.columns(2)
    izq.markdown("**Habitantes**")
    izq.altair_chart(absoluta)
    der.markdown(f"**Cuánto ha crecido o menguado desde {anio0}, frente a su entorno**")
    der.altair_chart(relativa)
    if proy is not None:
        st.caption("La línea continua son los datos observados y la discontinua, la proyección con el "
                   "método de Holt. En provincias y comunidad, la proyección es la suma de la de sus "
                   "municipios.")

    st.subheader("Ritmo de cambio por periodos")
    h = st.radio("Periodos de", HORIZONTES, horizontal=True, format_func=lambda n: f"{n} años",
                 key="det_h")
    ventanas_graf = grafico_ventanas(serie, comparar, etiquetas_cod, h, ref)
    if ventanas_graf is None:
        st.info("No hay periodos completos para este horizonte.")
    else:
        st.altair_chart(ventanas_graf)

    st.subheader("Características de la población")
    for pestana, tema in zip(st.tabs(TEMAS_DETALLE), TEMAS_DETALLE):
        with pestana:
            bloque_tema(serie, cod, tema, etiquetas)

    st.subheader("Migraciones y movimiento natural")
    tasas = INDICADORES["Migraciones (tasa anual)"] + INDICADORES["Movimiento natural (tasa anual)"]
    izq, der = st.columns([3, 2])
    with izq:
        c, e, t = st.selectbox("Tasa anual", tasas, format_func=lambda x: x[1], key="det_flujo")
        st.altair_chart(grafico_lineas(serie, c, t, etiquetas))
    with der:
        acum = INDICADORES["Flujos acumulados (10 años)"]
        fila = serie[serie["periodo"] == ref].set_index("cod_mun")
        tabla = pd.DataFrame({etiquetas_cod[k]: [fmt(fila.at[k, c], t) if k in fila.index else "sin dato"
                                                 for c, _, t in acum] for k in comparar},
                             index=[e for _, e, _ in acum])
        st.markdown(f"**Acumulado {ref - 10}–{ref - 1}**, sobre la población de {ref - 10}")
        st.dataframe(tabla)

    if not propia_pr.empty:
        with st.expander("Tabla de proyecciones"):
            t = propia_pr.sort_values("periodo")
            st.dataframe(pd.DataFrame({
                "Año": t["periodo"].to_numpy(),
                "Población proyectada": t["total"].round().astype(int).to_numpy(),
                "Tasa anual desde la base": [fmt((v / pob) ** (1 / (a - ref)) - 1, "pct2") if a > ref else ""
                                             for a, v in zip(t["periodo"], t["total"])],
            }), hide_index=True)


####################################################################
### CLÚSTERES
####################################################################

def nombre_cluster(k):
    """Etiqueta visible del clúster k de 3_clustering: 0 -> «Grupo A», 1 -> «Grupo B»…"""
    letra = chr(ord("A") + int(k))
    return f"Grupo {letra}" + (f": {NOMBRES_GRUPO[letra]}" if letra in NOMBRES_GRUPO else "")


def color_grupo(k):
    return COLORES_CLUSTER[int(k) % len(COLORES_CLUSTER)]


def indicadores_cluster(tema, ref):
    if tema == TEMA_CLUSTERING:
        return VARIABLES_CLUSTERING
    if tema == TEMA_CAMBIO:  # todas las ventanas *_prev, de la más antigua a la más reciente
        return [(f"p_cambio_total_{h}_anual_constante{suf}", f"Tasa anual {ref - d - h}–{ref - d}", "pct2")
                for h in HORIZONTES for suf, d in sorted(PREV[h].items(), key=lambda x: -x[1])]
    return INDICADORES[tema]


def foto_reciente(mun, cols):
    """Último valor disponible de cada municipio, como el ffill de 3_clustering."""
    return mun.sort_values("periodo").groupby("cod_mun")[cols].last()


def percentil(serie, v):
    s = serie[serie.notna()]
    return np.nan if pd.isna(v) or s.empty else 100 * ((s < v).mean() + 0.5 * (s == v).mean())


def valores_por_cluster(df, ag_cl, cols, estadistico):
    """Valor de cada clúster y de la comunidad, con el estadístico elegido."""
    foto = foto_reciente(df[df["nivel"] == NIVEL_MUN], cols + ["cluster"])
    if estadistico == "agregado":
        grupos = ag_cl.sort_values("periodo").groupby("cluster")[cols].last()
        ca = df[df["cod_mun"] == COD_CA].sort_values("periodo")[cols].ffill().iloc[-1]
    else:
        grupos = foto.groupby("cluster")[cols].agg(estadistico)
        ca = foto[cols].agg(estadistico)
    tabla = grupos.rename(index=nombre_cluster)
    tabla.loc[NOMBRE_CA] = ca
    return tabla, foto


def serie_por_cluster(df, ag_cl, col, estadistico):
    """Valor de cada clúster y de la comunidad, año a año."""
    mun = df[df["nivel"] == NIVEL_MUN]
    if estadistico == "agregado":
        grupos = ag_cl[["cluster", "periodo", col]].rename(columns={col: "valor"})
        ca = df.loc[df["cod_mun"] == COD_CA, ["periodo", col]].rename(columns={col: "valor"})
    else:
        grupos = (mun[mun["cluster"].notna()].groupby(["cluster", "periodo"])[col]
                  .agg(estadistico).reset_index(name="valor"))
        ca = mun.groupby("periodo")[col].agg(estadistico).reset_index(name="valor")
    grupos["grupo"] = grupos["cluster"].map(nombre_cluster)
    return pd.concat([grupos[["grupo", "periodo", "valor"]], ca.assign(grupo=NOMBRE_CA)], ignore_index=True)


ETIQUETA_EN_DOS_LINEAS = "split(datum.label, ': ')"  # «Grupo C» / «intermedio arraigado»


def color_grupos(etiquetas, colores):
    return alt.Color("grupo:N", title=None, legend=alt.Legend(orient="right", labelLimit=260),
                     scale=alt.Scale(domain=etiquetas + [NOMBRE_CA], range=colores + [COLORES_TERR[2]]))


def tamano_grupos(df, ids, ref):
    """Municipios, población y mediana de habitantes de cada grupo (y de los que no tienen grupo)."""
    m = df[(df["nivel"] == NIVEL_MUN) & (df["periodo"] == ref)]
    filas = []
    for k in [*ids, None]:
        sub = m[m["cluster"] == k] if k is not None else m[m["cluster"].isna()]
        if len(sub):
            filas.append({"k": k, "municipios": len(sub), "poblacion": sub["total"].sum(),
                          "pct": sub["total"].sum() / m["total"].sum(), "mediana": sub["total"].median()})
    return pd.DataFrame(filas)


def muestra_color(color, ancho=12):
    return (f'<span style="display:inline-block;width:{ancho}px;height:12px;margin-right:8px;'
            f'vertical-align:middle;background:{color};"></span>')


def resumen_clusters(df, ids, ref):
    """Leyenda del mapa con el tamaño de cada grupo, en HTML."""
    filas = []
    for t in tamano_grupos(df, ids, ref).itertuples():
        k = None if pd.isna(t.k) else int(t.k)
        color = color_grupo(k) if k is not None else "rgba(90,90,90,.67)"
        nombre = nombre_cluster(k) if k is not None else "Sin grupo"
        celdas = [t.municipios, fmt(t.poblacion), fmt(t.pct, "pct"), fmt(t.mediana)]
        filas.append(
            f'<tr><td style="padding:4px 8px;">{muestra_color(color)}{nombre}</td>'
            + "".join(f'<td style="padding:4px 8px;text-align:right;">{c}</td>' for c in celdas) + "</tr>")
    cabecera = "".join(f'<th style="padding:4px 8px;text-align:{a};font-weight:600;">{t}</th>' for t, a in [
        ("Grupo", "left"), ("Municipios", "right"), (f"Población {ref}", "right"),
        ("% población", "right"), ("Mediana hab.", "right")])
    return (f'<table style="border-collapse:collapse;font-size:0.85rem;width:100%;">'
            f'<thead><tr>{cabecera}</tr></thead><tbody>{"".join(filas)}</tbody></table>')


def grafico_perfil(tabla, foto, inds, colorear=True):
    """Mapa de calor: valor de cada clúster, coloreado por su percentil entre los municipios.
    Con colorear=False (el agregado) las celdas son neutras y solo muestran el valor."""
    filas = []
    for col, etq, tipo in inds:
        for grupo, v in tabla[col].items():
            p = percentil(foto[col], v) if colorear else np.nan
            if pd.isna(v) or (colorear and pd.isna(p)):
                tinta = "#888"
            elif colorear and (p < 15 or p > 85):
                tinta = "white"
            else:
                tinta = "#1a1a1a"
            filas.append({"indicador": etq, "grupo": grupo, "valor_txt": fmt(v, tipo), "percentil": p,
                          "tinta": tinta})
    d = pd.DataFrame(filas)
    ayuda = [alt.Tooltip("grupo:N", title="Grupo"), alt.Tooltip("indicador:N", title="Indicador"),
             alt.Tooltip("valor_txt:N", title="Valor")]
    if colorear:
        ayuda.append(alt.Tooltip("percentil:Q", title="Percentil entre municipios", format=".0f"))
    base = alt.Chart(d).encode(
        x=alt.X("grupo:N", sort=list(tabla.index), title=None,
                axis=alt.Axis(orient="top", labelAngle=0, labelLimit=220, labelExpr=ETIQUETA_EN_DOS_LINEAS)),
        y=alt.Y("indicador:N", sort=[e for _, e, _ in inds], title=None, axis=alt.Axis(labelLimit=320)),
        tooltip=ayuda,
    )
    if colorear:
        celdas = base.mark_rect().encode(color=alt.Color(
            "percentil:Q", title="Percentil", scale=alt.Scale(domain=[0, 100], range=COOLWARM)))
    else:
        celdas = base.mark_rect(color="#eef1f5", stroke="white", strokeWidth=2)
    # Una capa de texto por color de tinta, con el color fijo en la marca. Codificarlo como campo con
    # scale=None hace fallar a Vega-Lite al combinar las capas («Cannot read properties of null»).
    textos = [base.transform_filter(alt.datum.tinta == tinta)
              .mark_text(fontSize=12, color=tinta).encode(text="valor_txt:N")
              for tinta in sorted(set(d["tinta"]))]
    # Altura en píxeles: 32 por fila más el eje superior de dos líneas.
    return alt.layer(celdas, *textos).properties(height=32 * len(inds) + 50)


def grafico_evolucion_clusters(df, ag_cl, col, tipo, etq, estadistico, etiquetas, colores):
    base = re.sub(r"_prev(_\d)?$", "", col)  # en las ventanas *_prev, la serie de la ventana actual
    h = re.search(r"cambio_total_(\d+)", base)
    titulo = f"Tasa anual de los {h.group(1)} años anteriores" if h else etq
    d = serie_por_cluster(df, ag_cl, base, estadistico)
    d["valor"] = d["valor"] * FACTOR[tipo]
    lineas = lineas_anuales(d[np.isfinite(d["valor"].astype(float))], "grupo", "Grupo",
                            color_grupos(etiquetas, colores), tipo, 12, ",.2f")
    if es_divergente(base):
        lineas = lineas + regla_cero(discontinua=True)
    return lineas.properties(height=320), f"{titulo}, por año"


def grafico_caja(foto, col, tipo, etiquetas, colores):
    d = foto[["cluster", col]].dropna()
    d = d[np.isfinite(d[col].astype(float))]
    d = d.assign(grupo=d["cluster"].map(nombre_cluster), valor=d[col] * FACTOR[tipo])
    log = tipo in ("int", "dens")  # muy asimétricas: escala logarítmica
    return alt.Chart(d).mark_boxplot(extent=1.5, size=38).encode(
        x=alt.X("grupo:N", sort=etiquetas, title=None, axis=alt.Axis(labelAngle=0, labelExpr=ETIQUETA_EN_DOS_LINEAS)),
        y=alt.Y("valor:Q", title=UNIDAD[tipo] + (" (escala logarítmica)" if log else ""),
                scale=alt.Scale(type="log") if log else alt.Scale(zero=False)),
        color=alt.Color("grupo:N", scale=alt.Scale(domain=etiquetas, range=colores), legend=None),
    ).properties(height=320)


def grafico_poblacion_clusters(df, terr, ag_cl, proy, etiquetas, colores):
    """Cambio de población de cada grupo desde el primer año, con la proyección de 4_prediccion."""
    obs = pd.concat([
        ag_cl.assign(grupo=ag_cl["cluster"].map(nombre_cluster))[["grupo", "periodo", "total"]],
        df.loc[df["cod_mun"] == COD_CA, ["periodo", "total"]].assign(grupo=NOMBRE_CA),
    ], ignore_index=True).dropna()
    partes = [obs.assign(tipo="Observada")]
    if proy is not None:
        asignacion = terr["cluster"].dropna().map(nombre_cluster)
        pm = proy[proy["cod_mun"].isin(asignacion.index)]
        pr = pd.concat([
            pm.assign(grupo=pm["cod_mun"].map(asignacion)).groupby(["grupo", "periodo"], as_index=False)["total"].sum(),
            proy.loc[proy["cod_mun"] == COD_CA, ["periodo", "total"]].assign(grupo=NOMBRE_CA),
        ], ignore_index=True)
        partes.append(anclar(pr, obs, "grupo").assign(tipo="Proyección"))
    grafico, _ = grafico_cambio(pd.concat(partes, ignore_index=True), obs, "grupo", "grupo", "Grupo",
                                color_grupos(etiquetas, colores), 340)
    return grafico


def perfil_mediano(df, ag_cl, inds):
    """Percentil de la mediana de cada grupo entre los municipios, como el radial de 3_clustering."""
    tabla, foto = valores_por_cluster(df, ag_cl, [c for c, _, _ in inds], "median")
    filas = [{"grupo": grupo, "indicador": etq, "orden": i, "valor_txt": fmt(v, tipo),
              "percentil": percentil(foto[col], v)}
             for i, (col, etq, tipo) in enumerate(inds)
             for grupo, v in tabla[col].drop(NOMBRE_CA).items()]
    return pd.DataFrame(filas)


def grafico_radial(perfil, inds, etiquetas, colores):
    """Gráfico radial en Altair: los ángulos se convierten a coordenadas x, y."""
    n = len(inds)
    angulo = lambda i: 2 * np.pi * np.asarray(i) / n  # empieza arriba y gira en sentido horario
    d = perfil.dropna(subset=["percentil"]).assign(r=lambda t: t["percentil"] / 100)
    d = pd.concat([d, d[d["orden"] == 0].assign(orden=n)])  # repite el primer eje para cerrar el polígono
    d["x"], d["y"] = d["r"] * np.sin(angulo(d["orden"])), d["r"] * np.cos(angulo(d["orden"]))

    ejes = pd.DataFrame({"indicador": [ETIQUETAS_RADIAL.get(c, e) for c, e, _ in inds], "orden": range(n)})
    ejes["x"], ejes["y"] = 1.1 * np.sin(angulo(ejes["orden"])), 1.1 * np.cos(angulo(ejes["orden"]))
    radios = pd.concat([ejes.assign(x=0.0, y=0.0), ejes.assign(x=ejes["x"] / 1.1, y=ejes["y"] / 1.1)])
    anillos = pd.DataFrame([{"anillo": r, "orden": i, "x": r * np.sin(angulo(i)), "y": r * np.cos(angulo(i))}
                            for r in (0.25, 0.5, 0.75, 1) for i in range(n + 1)])
    marcas = pd.DataFrame({"x": 0.0, "r": [0.25, 0.5, 0.75, 1], "texto": ["25", "50", "75", "100"]})

    # Misma escala en x e y (4,6 x 2,6 unidades en 500 x 283 px) para que no se deforme.
    # El margen horizontal deja sitio a las etiquetas de los ejes laterales.
    x = lambda c: alt.X(f"{c}:Q", scale=alt.Scale(domain=[-2.3, 2.3]), axis=None)
    y = lambda c: alt.Y(f"{c}:Q", scale=alt.Scale(domain=[-1.3, 1.3]), axis=None)
    gris = dict(color="#999", strokeWidth=0.7, opacity=0.6)
    capas = (
        alt.Chart(anillos).mark_line(**gris).encode(x=x("x"), y=y("y"), detail="anillo:N", order="orden:Q")
        + alt.Chart(radios).mark_line(**gris).encode(x=x("x"), y=y("y"), detail="indicador:N")
        + alt.Chart(marcas).mark_text(align="left", dx=4, dy=-6, fontSize=10, color="#9aa0a6")
        .encode(x=x("x"), y=y("r"), text="texto:N")
    )
    for filtro, alineacion in [(ejes["x"] > 0.05, "left"), (ejes["x"] < -0.05, "right"),
                               (ejes["x"].abs() <= 0.05, "center")]:
        capas += (alt.Chart(ejes[filtro]).mark_text(align=alineacion, fontSize=10, color="#9aa0a6")
                  .encode(x=x("x"), y=y("y"), text="indicador:N"))
    capas += alt.Chart(d).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=20, filled=True)).encode(
        x=x("x"), y=y("y"), order="orden:Q",
        color=alt.Color("grupo:N", title=None, scale=alt.Scale(domain=etiquetas, range=colores), legend=None),
        tooltip=[alt.Tooltip("grupo:N", title="Grupo"), alt.Tooltip("indicador:N", title="Indicador"),
                 alt.Tooltip("valor_txt:N", title="Mediana"),
                 alt.Tooltip("percentil:Q", title="Percentil entre municipios", format=".0f")],
    )
    return capas.properties(width=500, height=283).configure_view(strokeWidth=0)


def tarjetas_grupos(df, ag_cl, ids, ref):
    """Una tarjeta por grupo: descripción de la memoria y cifras clave."""
    tam = tamano_grupos(df, ids, ref).dropna(subset=["k"]).set_index("k")
    claves, _ = valores_por_cluster(df, ag_cl, ["total", "p_cambio_total_10", "p_cambio_total_20"], "median")
    for columna, k in zip(st.columns(len(ids)), ids):
        nombre, letra = nombre_cluster(k), chr(ord("A") + k)
        with columna.container(border=True):
            st.markdown(muestra_color(color_grupo(k)) + f"**{nombre}**",
                        unsafe_allow_html=True)
            st.caption(f"{tam.at[k, 'municipios']} municipios, {fmt(tam.at[k, 'pct'], 'pct')} de la población")
            if letra in DESCRIPCIONES_GRUPO:
                st.markdown(DESCRIPCIONES_GRUPO[letra])
            st.caption(f"Mediana de {fmt(claves.at[nombre, 'total'])} habitantes. Cambio de población: "
                       f"{fmt_delta(claves.at[nombre, 'p_cambio_total_10'])} en 10 años y "
                       f"{fmt_delta(claves.at[nombre, 'p_cambio_total_20'])} en 20.")


def vista_clusters(df, terr, geos, ag_cl, proy):
    st.title("Grupos de municipios")
    if ag_cl is None:
        logging.warning("No se encuentra %s. Se genera con la libreta 3_clustering.", CLUSTERS_PATH)
        st.info("La vista de grupos no está disponible porque faltan los datos de los grupos.")
        return
    ref = int(df["periodo"].max())
    ids = sorted(int(k) for k in ag_cl["cluster"].unique())
    etiquetas = [nombre_cluster(k) for k in ids]
    colores = [color_grupo(k) for k in ids]
    st.caption(f"{len(ids)} grupos de municipios con perfiles demográficos parecidos, formados con un "
               "algoritmo de agrupación a partir de los datos más recientes de cada municipio. Las "
               "letras van de menor a mayor población mediana.")

    st.subheader("Pertenencia y perfil")
    poblacion = df[(df["nivel"] == NIVEL_MUN) & (df["periodo"] == ref)].set_index("cod_mun")["total"].to_dict()
    izq, der = st.columns([3, 2])
    def propiedades(cod):
        k = terr["cluster"].get(cod)
        tiene = k is not None and pd.notna(k)
        return {"cod": cod, "nombre": terr["corta"].get(cod, cod), "pob": fmt(poblacion.get(cod)),
                "grupo": nombre_cluster(k) if tiene else "Sin grupo",
                "color": [*_rgb(color_grupo(k)).tolist(), 220] if tiene else SIN_DATO}
    with izq:
        cod = mapa_clicable(geos, features(geos[NIVEL_MUN], propiedades), 0.4,
                            "<b>{nombre}</b><br/>{grupo}<br/>Población: {pob}<br/><i>Clic para abrir la ficha</i>",
                            True, 560, "mapa_cl")
    perfil = perfil_mediano(df, ag_cl, VARIABLES_CLUSTERING)
    with der:
        st.markdown(resumen_clusters(df, ids, ref), unsafe_allow_html=True)
        st.caption("Haz clic en un municipio del mapa para abrir su ficha.")
        st.altair_chart(grafico_radial(perfil, VARIABLES_CLUSTERING, etiquetas, colores))
        st.caption("Percentil de la mediana de cada grupo entre todos los municipios: 0 en el centro y 100 "
                   "en el borde. Los colores son los de la tabla.")
    ir_a_ficha(cod, terr)

    st.markdown("**Qué define a cada grupo**")
    tarjetas_grupos(df, ag_cl, ids, ref)
    st.caption("Las cifras son medianas de los municipios de cada grupo.")

    st.subheader("Perfil detallado")
    c1, c2 = st.columns(2)
    tema = c1.selectbox("Indicadores", [TEMA_CLUSTERING, TEMA_CAMBIO, *INDICADORES], key="cl_tema")
    estadistico = ESTADISTICOS[c2.radio(
        "Valor de cada grupo", list(ESTADISTICOS), horizontal=True, key="cl_estad",
        help="La mediana y la media cuentan igual todos los municipios del grupo. El agregado trata "
             "el grupo como un único territorio, así que pesan más los municipios grandes.")]
    inds = indicadores_cluster(tema, ref)
    tabla, foto = valores_por_cluster(df, ag_cl, [c for c, _, _ in inds], estadistico)
    agregado = estadistico == "agregado"
    st.altair_chart(grafico_perfil(tabla, foto, inds, colorear=not agregado))
    if agregado:
        st.caption("Cada celda muestra el valor del grupo como si fuera un único territorio, con el último "
                   "dato disponible. Las celdas no se colorean, porque un valor que suma muchos municipios "
                   "no se puede comparar con el de un municipio.")
    else:
        st.caption("Cada celda muestra el valor del grupo. El color indica dónde queda ese valor entre los "
                   "municipios de la comunidad: azul si está por debajo de lo habitual y rojo si está por "
                   "encima. Se usa el último dato disponible de cada municipio.")

    st.subheader("Comparación de un indicador")
    sel, _ = st.columns([1, 2])
    col, etq, tipo = sel.selectbox("Indicador", inds, format_func=lambda x: x[1], key=f"cl_ind_{tema}")
    evolucion, titulo = grafico_evolucion_clusters(df, ag_cl, col, tipo, etq, estadistico, etiquetas, colores)
    izq, der = st.columns(2)
    izq.markdown(f"**{titulo}**")
    izq.altair_chart(evolucion)
    der.markdown("**Dispersión entre los municipios de cada grupo**")
    der.altair_chart(grafico_caja(foto, col, tipo, etiquetas, colores))

    st.subheader("Evolución de la población de cada grupo")
    st.altair_chart(grafico_poblacion_clusters(df, terr, ag_cl, proy, etiquetas, colores))
    if proy is not None:
        st.caption("Población total de los municipios de cada grupo. La línea continua son los datos "
                   "observados y la discontinua, la suma de las proyecciones de sus municipios.")


####################################################################
### PRINCIPAL
####################################################################

def fuentes(anio):
    """Acreditación de las fuentes y aviso legal, comunes a todas las vistas."""
    st.divider()
    st.caption(PIE_FUENTES)
    st.caption(PIE_AVISO.format(anio=anio))
    with st.sidebar.expander("Atribución de fuentes de datos"):
        st.markdown(FUENTES)
    with st.sidebar.expander("Aviso legal"):
        st.markdown(AVISO_LEGAL)


def main():
    st.set_page_config(page_title="Población de Castilla-La Mancha", layout="wide")

    faltan = [p for p in [DATA_PATH, *GEO_PATHS.values()] if not Path(p).exists()]
    if faltan:
        logging.error("Faltan ficheros de datos: %s", ", ".join(faltan))
        st.error("No se han encontrado los datos de la aplicación.")
        st.stop()

    df, terr, ag_cl = cargar_panel(DATA_PATH, CLUSTERS_PATH)
    proy = cargar_proyecciones(PRED_PATH) if Path(PRED_PATH).exists() else None
    codigos = {NIVEL_MUN: tuple(terr.index[terr["nivel"] == NIVEL_MUN]),
               NIVEL_PROV: tuple(PROVINCIAS), NIVEL_CA: (COD_CA,)}
    geos = {nivel: cargar_geo(p, nivel, codigos[nivel]) for nivel, p in GEO_PATHS.items()}

    for clave, inicial in [("vista", "mapa"), ("cod", None), ("nonce", 0)]:
        st.session_state.setdefault(clave, inicial)

    with st.sidebar:
        st.radio("Vista", ["Indicadores", "Grupos"], key="modo", horizontal=True, on_change=volver)
        st.selectbox("Buscar territorio", terr.index, index=None, key="buscador", on_change=al_buscar,
                     format_func=lambda c: terr.at[c, "busqueda"], placeholder="Nombre o código INE")

    if st.session_state.vista == "detalle" and st.session_state.cod in terr.index:
        vista_detalle(df, terr, proy, st.session_state.cod, geos)
    elif st.session_state.modo == "Grupos":
        vista_clusters(df, terr, geos, ag_cl, proy)
    else:
        vista_mapa(df, terr, geos)
    fuentes(int(df["periodo"].max()))


if __name__ == "__main__":
    main()