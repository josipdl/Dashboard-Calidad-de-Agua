import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from mapa_panama_component import get_layout as get_mapa_layout


RUTA_CSV = "BKB_WaterQualityData_2020084.csv"

COLS_CORRELACION = [
    "Salinity (ppt)",
    "Dissolved Oxygen (mg/L)",
    "pH (standard units)",
    "Secchi Depth (m)",
    "Water Depth (m)",
    "Water Temp (?C)",
    "AirTemp (C)",
]

ETIQUETAS_CORR = [
    "Salinidad",
    "O2 Disuelto",
    "pH",
    "Prof. Secchi",
    "Prof. Agua",
    "Temp Agua",
    "Temp Aire"
]

FEATURES_MODELO = [
    "Salinity (ppt)",
    "pH (standard units)",
    "Water Temp (?C)",
    "AirTemp (C)",
    "Secchi Depth (m)",
    "Water Depth (m)",
    "Year",
]

VARIABLE_OBJETIVO = "Dissolved Oxygen (mg/L)"


def cargar_dataset(ruta: str) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    return df


def limpiar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Site_Id"] = df["Site_Id"].replace({"d": "D"})

    df = df[df["Year"] >= 1989].copy()

    cols_candidatas_eliminar = df.isnull().sum()[df.isnull().sum() > 250].index.tolist()

    col_conservar = "Dissolved Oxygen (mg/L)"
    if col_conservar in cols_candidatas_eliminar:
        cols_candidatas_eliminar.remove(col_conservar)

    df_clean = df.drop(columns=cols_candidatas_eliminar)

    cols_numericas = df_clean.select_dtypes(include=[np.number]).columns

    for col in cols_numericas:
        if df_clean[col].isnull().sum() > 0:
            mediana = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(mediana)

    return df_clean


df_raw = cargar_dataset(RUTA_CSV)
df_clean = limpiar_dataset(df_raw)

SITIOS_DISPONIBLES = sorted(df_clean["Site_Id"].dropna().unique().tolist())
YEAR_MIN = int(df_clean["Year"].min())
YEAR_MAX = int(df_clean["Year"].max())


def entrenar_modelo(df_clean: pd.DataFrame):

    features_reg = [c for c in FEATURES_MODELO if c in df_clean.columns]

    X_r = df_clean[features_reg]
    y_r = df_clean[VARIABLE_OBJETIVO]

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_r,
        y_r,
        test_size=0.2,
        random_state=42
    )

    rf_reg = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    rf_reg.fit(Xr_train, yr_train)

    r2_train = r2_score(
        yr_train,
        rf_reg.predict(Xr_train)
    )

    r2_test = r2_score(
        yr_test,
        rf_reg.predict(Xr_test)
    )

    return rf_reg, features_reg, r2_train, r2_test


modelo_rf, FEATURES_MODELO_FINAL, R2_TRAIN, R2_TEST = entrenar_modelo(df_clean)


def clasificar_oxigeno(valor: float) -> str:
    if valor < 2:
        return "Bajo"
    elif valor < 5:
        return "Medio"
    else:
        return "Alto"


TEMPLATE = "plotly_dark"
PALETA = px.colors.qualitative.Set2


def filtrar_por_anio(df: pd.DataFrame, rango_anios):
    return df[
        (df["Year"] >= rango_anios[0]) &
        (df["Year"] <= rango_anios[1])
    ]

def grafica_heatmap(df: pd.DataFrame) -> go.Figure:
    matriz_corr = df[COLS_CORRELACION].corr()

    matriz_corr.index = ETIQUETAS_CORR
    matriz_corr.columns = ETIQUETAS_CORR

    fig = px.imshow(
        matriz_corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )

    fig.update_layout(
        template=TEMPLATE,
        title="Correlación entre Variables Físico-Químicas",
        margin=dict(l=10, r=10, t=50, b=10),
        coloraxis_colorbar=dict(title="r"),
    )

    return fig


def grafica_scatter(df: pd.DataFrame, sitio: str) -> go.Figure:
    datos = df[
        [
            "Water Temp (?C)",
            VARIABLE_OBJETIVO,
            "Site_Id"
        ]
    ].dropna()

    if sitio and sitio != "TODOS":
        datos = datos[datos["Site_Id"] == sitio]

    if len(datos) < 2:
        fig = go.Figure()
        fig.update_layout(
            template=TEMPLATE,
            title="Sin datos suficientes para este filtro"
        )
        return fig

    corr = datos["Water Temp (?C)"].corr(
        datos[VARIABLE_OBJETIVO]
    )

    z = np.polyfit(
        datos["Water Temp (?C)"],
        datos[VARIABLE_OBJETIVO],
        1
    )

    p = np.poly1d(z)

    x_line = np.linspace(
        datos["Water Temp (?C)"].min(),
        datos["Water Temp (?C)"].max(),
        100
    )

    fig = px.scatter(
        datos,
        x="Water Temp (?C)",
        y=VARIABLE_OBJETIVO,
        opacity=0.45,
        color_discrete_sequence=["#4FD1C5"],
    )

    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=p(x_line),
            mode="lines",
            name="Tendencia",
            line=dict(
                color="#F56565",
                width=3,
                dash="dash"
            ),
        )
    )

    fig.update_layout(
        template=TEMPLATE,
        title=f"Temp. del Agua vs Oxígeno Disuelto  (r = {corr:.3f})",
        xaxis_title="Temperatura del Agua (°C)",
        yaxis_title="Oxígeno Disuelto (mg/L)",
        margin=dict(l=10, r=10, t=50, b=10),
    )

    return fig


def grafica_barras_sitio(df: pd.DataFrame) -> go.Figure:

    promedios_sitio = df.groupby("Site_Id")[
        [
            "Salinity (ppt)",
            VARIABLE_OBJETIVO,
            "pH (standard units)",
            "Water Temp (?C)"
        ]
    ].mean().reset_index()

    promedios_sitio = promedios_sitio.rename(
        columns={
            "Salinity (ppt)": "Salinidad (ppt)",
            VARIABLE_OBJETIVO: "O2 Disuelto (mg/L)",
            "pH (standard units)": "pH",
            "Water Temp (?C)": "Temp. Agua (°C)",
        }
    )

    df_melt = promedios_sitio.melt(
        id_vars="Site_Id",
        var_name="Variable",
        value_name="Promedio"
    )

    fig = px.bar(
        df_melt,
        x="Site_Id",
        y="Promedio",
        color="Variable",
        barmode="group",
        color_discrete_sequence=PALETA,
    )

    fig.update_layout(
        template=TEMPLATE,
        title="Promedio de Variables por Sitio de Muestreo",
        xaxis_title="Sitio (Site_Id)",
        yaxis_title="Valor promedio",
        legend_title="Variable",
        margin=dict(l=10, r=10, t=50, b=10),
    )

    return fig


def grafica_evolucion_anual(df: pd.DataFrame, sitio: str) -> go.Figure:

    datos = df.copy()

    if sitio and sitio != "TODOS":
        datos = datos[datos["Site_Id"] == sitio]

    por_anio = datos.groupby("Year")[
        VARIABLE_OBJETIVO
    ].mean().reset_index()

    fig = px.line(
        por_anio,
        x="Year",
        y=VARIABLE_OBJETIVO,
        markers=True,
        color_discrete_sequence=["#68D391"],
    )

    fig.update_layout(
        template=TEMPLATE,
        title="Evolución Anual del Oxígeno Disuelto"
        + (
            f" — Sitio {sitio}"
            if sitio and sitio != "TODOS"
            else " — Todos los sitios"
        ),
        xaxis_title="Año",
        yaxis_title="Oxígeno Disuelto promedio (mg/L)",
        margin=dict(l=10, r=10, t=50, b=10),
    )

    return fig


app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        dbc.icons.FONT_AWESOME
    ],
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1"
        }
    ],
    suppress_callback_exceptions=True,
)

app.title = "Dashboard Calidad del Agua"

server = app.server


CARD_STYLE = {
    "borderRadius": "1rem",
    "boxShadow": "0 4px 18px rgba(0,0,0,0.35)",
    "border": "1px solid rgba(255,255,255,0.06)",
}


def tarjeta_kpi(titulo, valor, icono):
    return dbc.Card(
        dbc.CardBody(
            [
                html.I(
                    className=f"fa-solid {icono} fa-lg",
                    style={"color": "#4FD1C5"}
                ),
                html.H3(
                    valor,
                    className="mt-2 mb-0",
                    style={"fontWeight": "700"}
                ),
                html.P(
                    titulo,
                    className="text-muted mb-0",
                    style={"fontSize": "0.85rem"}
                ),
            ]
        ),
        style=CARD_STYLE,
        className="text-center h-100",
    )

header = dbc.Container(
    [
        html.H1(
            "Dashboard de Calidad del Agua (1994-2019)",
            className="mt-4 mb-1",
            style={"fontWeight": "800"}
        ),
        html.P(
            "Análisis Exploratorio y Predicción de Oxígeno Disuelto.",
            className="text-muted mb-4",
            style={"fontSize": "1.05rem"},
        ),
        dbc.Row(
            [
                dbc.Col(
                    tarjeta_kpi(
                        "Total de registros",
                        f"{len(df_raw):,}",
                        "fa-database"
                    ),
                    md=4,
                    className="mb-3"
                ),
                dbc.Col(
                    tarjeta_kpi(
                        "Variables",
                        f"{df_raw.shape[1]}",
                        "fa-table-columns"
                    ),
                    md=4,
                    className="mb-3"
                ),
                dbc.Col(
                    tarjeta_kpi(
                        "Periodo cubierto",
                        f"{YEAR_MIN} - {YEAR_MAX}",
                        "fa-calendar-days"
                    ),
                    md=4,
                    className="mb-3"
                ),
            ]
        ),
    ],
    fluid=True,
)


controles = dbc.Container(
    dbc.Card(
        dbc.CardBody(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label(
                                "Sitio de muestreo (Site_Id)",
                                className="fw-bold mb-1"
                            ),
                            dcc.Dropdown(
                                id="dd-sitio",
                                options=[
                                    {
                                        "label": "Todos los sitios",
                                        "value": "TODOS"
                                    }
                                ]
                                + [
                                    {
                                        "label": s,
                                        "value": s
                                    }
                                    for s in SITIOS_DISPONIBLES
                                ],
                                value="TODOS",
                                clearable=False,
                                style={"color": "#111"},
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.Label(
                                "Rango de años",
                                className="fw-bold mb-1"
                            ),
                            dcc.RangeSlider(
                                id="rs-anios",
                                min=YEAR_MIN,
                                max=YEAR_MAX,
                                value=[
                                    YEAR_MIN,
                                    YEAR_MAX
                                ],
                                marks={
                                    y: str(y)
                                    for y in range(
                                        YEAR_MIN,
                                        YEAR_MAX + 1,
                                        5
                                    )
                                },
                                tooltip={
                                    "placement": "bottom",
                                    "always_visible": False
                                },
                            ),
                        ],
                        md=8,
                    ),
                ],
                align="center",
            )
        ),
        style=CARD_STYLE,
    ),
    fluid=True,
    className="mb-4",
)


graficas = dbc.Container(
    [
        html.H2(
            "Análisis Exploratorio",
            className="mb-3",
            style={"fontWeight": "700"}
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dcc.Graph(id="g-heatmap"),
                        style=CARD_STYLE
                    ),
                    md=6,
                    className="mb-4"
                ),
                dbc.Col(
                    dbc.Card(
                        dcc.Graph(id="g-scatter"),
                        style=CARD_STYLE
                    ),
                    md=6,
                    className="mb-4"
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dcc.Graph(id="g-barras-sitio"),
                        style=CARD_STYLE
                    ),
                    md=6,
                    className="mb-4"
                ),
                dbc.Col(
                    dbc.Card(
                        dcc.Graph(id="g-evolucion"),
                        style=CARD_STYLE
                    ),
                    md=6,
                    className="mb-4"
                ),
            ]
        ),
    ],
    fluid=True,
)


def input_predictor(id_, label, valor, step=0.1):
    return dbc.Col(
        [
            html.Label(
                label,
                className="mb-1",
                style={"fontSize": "0.85rem"}
            ),
            dbc.Input(
                id=id_,
                type="number",
                value=valor,
                step=step
            ),
        ],
        md=3,
        className="mb-3",
    )


prediccion = dbc.Container(
    [
        html.H2(
            "Predicción de Oxígeno Disuelto",
            className="mb-2",
            style={"fontWeight": "700"}
        ),
        html.P(
            f"Modelo: Random Forest Regressor (n_estimators=200) — "
            f"R² en test: {R2_TEST:.3f} "
            f"(R² en entrenamiento: {R2_TRAIN:.3f}).",
            className="text-muted",
        ),
        dbc.Card(
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            input_predictor(
                                "in-salinidad",
                                "Salinidad (ppt)",
                                round(
                                    df_clean["Salinity (ppt)"].median(),
                                    2
                                )
                            ),
                            input_predictor(
                                "in-ph",
                                "pH (standard units)",
                                round(
                                    df_clean["pH (standard units)"].median(),
                                    2
                                )
                            ),
                            input_predictor(
                                "in-temp-agua",
                                "Water Temp (°C)",
                                round(
                                    df_clean["Water Temp (?C)"].median(),
                                    2
                                )
                            ),
                            input_predictor(
                                "in-temp-aire",
                                "AirTemp (°C)",
                                round(
                                    df_clean["AirTemp (C)"].median(),
                                    2
                                )
                            ),
                        ]
                    ),
                    dbc.Row(
                        [
                            input_predictor(
                                "in-secchi",
                                "Secchi Depth (m)",
                                round(
                                    df_clean["Secchi Depth (m)"].median(),
                                    2
                                )
                            ),
                            input_predictor(
                                "in-prof-agua",
                                "Water Depth (m)",
                                round(
                                    df_clean["Water Depth (m)"].median(),
                                    2
                                )
                            ),
                            input_predictor(
                                "in-year",
                                "Año (Year)",
                                YEAR_MAX,
                                step=1
                            ),
                        ]
                    ),
                    dbc.Button(
                        "Predecir Oxígeno Disuelto",
                        id="btn-predecir",
                        color="info",
                        className="mt-2 mb-3",
                        n_clicks=0,
                    ),
                    html.Div(
                        id="resultado-prediccion"
                    ),
                ]
            ),
            style=CARD_STYLE,
        ),
    ],
    fluid=True,
    className="mb-5",
)


mapa_panama = dbc.Container(
    [
        html.H2(
            "Mapa Interactivo de Panamá"
        ),
        get_mapa_layout(),
    ],
    fluid=True,
    className="mb-5",
)


app.layout = html.Div(
    [
        header,
        controles,
        graficas,
        prediccion,
        mapa_panama
    ],
    style={
        "paddingBottom": "3rem"
    },
)


@app.callback(
    Output("g-heatmap", "figure"),
    Output("g-barras-sitio", "figure"),
    Input("rs-anios", "value"),
)
def actualizar_heatmap_y_barras(rango_anios):

    df_filtrado = filtrar_por_anio(
        df_clean,
        rango_anios
    )

    return (
        grafica_heatmap(df_filtrado),
        grafica_barras_sitio(df_filtrado)
    )


@app.callback(
    Output("g-scatter", "figure"),
    Output("g-evolucion", "figure"),
    Input("dd-sitio", "value"),
    Input("rs-anios", "value"),
)
def actualizar_scatter_y_evolucion(sitio, rango_anios):

    df_filtrado = filtrar_por_anio(
        df_clean,
        rango_anios
    )

    return (
        grafica_scatter(df_filtrado, sitio),
        grafica_evolucion_anual(df_filtrado, sitio)
    )


@app.callback(
    Output("resultado-prediccion", "children"),
    Input("btn-predecir", "n_clicks"),
    State("in-salinidad", "value"),
    State("in-ph", "value"),
    State("in-temp-agua", "value"),
    State("in-temp-aire", "value"),
    State("in-secchi", "value"),
    State("in-prof-agua", "value"),
    State("in-year", "value"),
    prevent_initial_call=True,
)
def predecir_oxigeno(
    n_clicks,
    salinidad,
    ph,
    temp_agua,
    temp_aire,
    secchi,
    prof_agua,
    year
):

    valores = {
        "Salinity (ppt)": salinidad,
        "pH (standard units)": ph,
        "Water Temp (?C)": temp_agua,
        "AirTemp (C)": temp_aire,
        "Secchi Depth (m)": secchi,
        "Water Depth (m)": prof_agua,
        "Year": year,
    }

    if any(v is None for v in valores.values()):
        return dbc.Alert(
            "Por favor completa todos los campos numéricos.",
            color="warning"
        )

    X_nuevo = pd.DataFrame(
        [
            [
                valores[f]
                for f in FEATURES_MODELO_FINAL
            ]
        ],
        columns=FEATURES_MODELO_FINAL
    )

    prediccion_valor = float(
        modelo_rf.predict(X_nuevo)[0]
    )

    categoria = clasificar_oxigeno(
        prediccion_valor
    )

    colores = {
        "Bajo": "danger",
        "Medio": "warning",
        "Alto": "success"
    }

    descripciones = {
        "Bajo": "Condición de hipoxia (< 2 mg/L): riesgo para la vida acuática.",
        "Medio": "Nivel aceptable (2 - 5 mg/L).",
        "Alto": "Buen nivel de oxigenación (> 5 mg/L).",
    }

    return dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.P(
                                "Predicción estimada:",
                                className="mb-1 text-muted"
                            ),
                            html.H3(
                                f"{prediccion_valor:.2f} mg/L",
                                style={
                                    "fontWeight": "800"
                                }
                            ),
                        ]
                    ),
                    style=CARD_STYLE,
                ),
                md=6,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.P(
                                "Categoría de calidad:",
                                className="mb-1 text-muted"
                            ),
                            dbc.Badge(
                                categoria,
                                color=colores[categoria],
                                className="fs-5 px-3 py-2 mb-2"
                            ),
                            html.P(
                                descripciones[categoria],
                                className="mb-0",
                                style={
                                    "fontSize": "0.85rem"
                                }
                            ),
                        ]
                    ),
                    style=CARD_STYLE,
                ),
                md=6,
            ),
        ],
        className="mt-1",
    )


if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=8050
    )