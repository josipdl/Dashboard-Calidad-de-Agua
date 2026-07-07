"""
Componente de Dash: Mapa Interactivo de Población por Distrito - Panamá
-------------------------------------------------------------------------
Este módulo expone:
  - get_figuras()  -> (fig_mapa, fig_barras)   objetos Plotly listos para usar
  - get_layout()   -> layout de Dash (html.Div) para insertar en la app

Uso dentro de la app principal del equipo (app.py del Ingeniero de ML):

    from mapa_panama_component import get_layout

    app.layout = html.Div([
        ...                     # resto de gráficas del dashboard
        get_layout(),           # <- aquí se inserta el mapa + resumen
        ...
    ])

Requiere en requirements.txt:
    dash
    plotly
    geopandas
    pandas
"""

import geopandas as gpd
import plotly.express as px
from dash import dcc, html

from preparar_datos import construir_gdf_final


def get_figuras():
    gdf = construir_gdf_final("geoBoundaries-PAN-ADM2.geojson", "datos_poblacion.csv")
    gdf = gdf.to_crs(epsg=4326)
    gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.0015, preserve_topology=True)
    geojson = gdf.__geo_interface__

    fig_mapa = px.choropleth_map(
        gdf,
        geojson=geojson,
        locations=gdf.index,
        color="poblacion_total",
        hover_name="distrito",
        hover_data={
            "provincia": True,
            "poblacion_total": ":,.0f",
            "densidad_hab_km2": ":,.1f",
            "area_km2": ":,.1f",
        },
        color_continuous_scale="YlOrRd",
        map_style="carto-positron",
        zoom=6.4,
        center={"lat": 8.6, "lon": -80.1},
        opacity=0.85,
        labels={
            "poblacion_total": "Población total",
            "densidad_hab_km2": "Densidad (hab/km²)",
            "area_km2": "Área (km²)",
            "provincia": "Provincia",
        },
        title="Población total por distrito - Panamá (Censo 2023, INEC)",
    )
    fig_mapa.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0}, height=600)

    top15 = gdf.sort_values("poblacion_total", ascending=False).head(15).iloc[::-1]
    fig_barras = px.bar(
        top15,
        x="poblacion_total",
        y="distrito",
        color="provincia",
        orientation="h",
        text="poblacion_total",
        labels={"poblacion_total": "Población total", "distrito": "Distrito",
                "provincia": "Provincia"},
        title="Los 15 distritos más poblados de Panamá (Censo 2023)",
    )
    fig_barras.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_barras.update_layout(height=500, margin={"r": 20, "t": 60, "l": 120, "b": 40})

    return fig_mapa, fig_barras


def get_layout():
    """Bloque listo para insertar en app.layout de Dash."""
    fig_mapa, fig_barras = get_figuras()
    return html.Div([
        html.H3("Mapa Interactivo: Población por Distrito (Panamá, Censo 2023 - INEC)"),
        dcc.Graph(id="mapa-distritos-panama", figure=fig_mapa),
        dcc.Graph(id="resumen-top15-distritos", figure=fig_barras),
    ])


# Para probar este componente de forma aislada:
#   python mapa_panama_component.py
if __name__ == "__main__":
    from dash import Dash
    app = Dash(__name__)
    app.layout = get_layout()
    app.run(debug=True)
