"""
Preparación de datos para el Mapa Interactivo de Panamá (nivel distrito)
--------------------------------------------------------------------------
Fuentes:
  - Geometría: geoBoundaries (geoBoundaries-PAN-ADM2.geojson) - 76 distritos
  - Variable:  INEC Mapi - Población total por distrito, Censo 2023
               (datos_poblacion.csv)

Por qué se necesita un "merge" especial:
  geoBoundaries todavía refleja la división político-administrativa ANTERIOR
  a varias segregaciones de distritos aprobadas entre 2015 y 2018. El censo
  2023 (INEC) ya usa la división NUEVA (82 distritos). Por lo tanto, antes de
  unir los datos a la geometría, se deben re-agrupar los distritos nuevos
  dentro de su distrito "padre" histórico:

    Distrito nuevo (CSV)                     -> Distrito padre (geoJSON)
    ------------------------------------------------------------------
    Almirante (Bocas del Toro)               -> Changuinola
    Omar Torrijos Herrera (Colón)            -> Donoso
    Tierras Altas (Chiriquí)                 -> Bugaba
    Jirondai (Comarca Ngäbe-Buglé)           -> Kankintú
    Santa Catalina o Calovébora (Ngäbe-Buglé)-> Kusapín
    Santa Fe (Darién)                        -> Chepigana

  (Fuentes: Wikipedia/Asamblea Nacional de Panamá - leyes de creación de
  cada distrito). Esto permite that las 82 filas del censo se agreguen en
  76 geometrías, sin perder ni un habitante.
"""

import unicodedata
import geopandas as gpd
import pandas as pd


def normalizar(texto: str) -> str:
    """Mayúsculas, sin tildes/diéresis, sin espacios extra. Para poder cruzar
    nombres que vienen con distinta ortografía en cada fuente."""
    texto = str(texto).strip().upper()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return " ".join(texto.split())


# Distritos nuevos -> distrito padre histórico (ya normalizados)
MERGE_A_DISTRITO_PADRE = {
    "ALMIRANTE": "CHANGUINOLA",
    "OMAR TORRIJOS HERRERA": "DONOSO",
    "TIERRAS ALTAS": "BUGABA",
    "JIRONDAI": "KANKINTU",
    "SANTA CATALINA O CALOVEBORA (BLEDESHIA)": "KUSAPIN",
}

# Caso especial: "SANTA FE" aparece 2 veces en el censo (Darién y Veraguas).
# La de Darién es un distrito nuevo (2018) segregado de Chepigana; la de
# Veraguas es un distrito histórico que sí existe en geoBoundaries y no debe
# tocarse. Se distingue por la provincia.
SANTA_FE_DARIEN_PADRE = "CHEPIGANA"


def cargar_geometrias(path_geojson: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path_geojson)
    gdf["distrito_norm"] = gdf["shapeName"].apply(normalizar)
    return gdf


def cargar_poblacion(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    # Quitar la fila de totales nacionales (columnas corridas / sin geometría)
    df = df[df["ID Provincia"] != "TOTAL"].copy()
    df["Valor"] = pd.to_numeric(df["Valor"], errors="coerce")

    df["distrito_norm"] = df["Nombre Distrito"].apply(normalizar)
    df["provincia_norm"] = df["Nombre Provincia"].apply(normalizar)

    # Resolver el caso especial de "SANTA FE" (Darién) antes del merge general
    es_santa_fe_darien = (df["distrito_norm"] == "SANTA FE") & (
        df["provincia_norm"] == "DARIEN"
    )
    df.loc[es_santa_fe_darien, "distrito_norm"] = SANTA_FE_DARIEN_PADRE

    # Reasignar los demás distritos nuevos a su distrito padre
    df["distrito_norm"] = df["distrito_norm"].replace(MERGE_A_DISTRITO_PADRE)

    # Conservar la provincia "representativa" de cada distrito padre
    provincia_por_distrito = (
        df.sort_values("Valor", ascending=False)
        .groupby("distrito_norm")["Nombre Provincia"]
        .first()
    )

    poblacion = df.groupby("distrito_norm", as_index=False)["Valor"].sum()
    poblacion = poblacion.rename(columns={"Valor": "poblacion_total"})
    poblacion["provincia"] = poblacion["distrito_norm"].map(provincia_por_distrito)
    return poblacion


def construir_gdf_final(path_geojson: str, path_csv: str) -> gpd.GeoDataFrame:
    gdf = cargar_geometrias(path_geojson)
    poblacion = cargar_poblacion(path_csv)

    gdf_final = gdf.merge(poblacion, on="distrito_norm", how="left")

    faltantes = gdf_final[gdf_final["poblacion_total"].isna()]
    if len(faltantes):
        print("ATENCIÓN, distritos sin dato de población:")
        print(faltantes[["shapeName"]])

    # Área y densidad poblacional (variable derivada extra)
    gdf_metric = gdf_final.to_crs(epsg=32617)  # UTM 17N, adecuado para Panamá
    gdf_final["area_km2"] = gdf_metric.geometry.area / 1_000_000
    gdf_final["densidad_hab_km2"] = (
        gdf_final["poblacion_total"] / gdf_final["area_km2"]
    ).round(1)

    gdf_final = gdf_final.rename(columns={"shapeName": "distrito"})
    return gdf_final[
        ["distrito", "provincia", "poblacion_total", "area_km2",
         "densidad_hab_km2", "geometry"]
    ]


if __name__ == "__main__":
    gdf_final = construir_gdf_final(
        "geoBoundaries-PAN-ADM2.geojson", "datos_poblacion.csv"
    )
    print(f"Distritos totales: {len(gdf_final)}")
    print(f"Población total Panamá: {gdf_final['poblacion_total'].sum():,.0f}")
    print(gdf_final.sort_values("poblacion_total", ascending=False).head(10)
          [["distrito", "provincia", "poblacion_total", "densidad_hab_km2"]])

    gdf_final.to_file("panama_distritos_poblacion.geojson", driver="GeoJSON")
    gdf_final.drop(columns="geometry").to_csv(
        "panama_distritos_poblacion.csv", index=False
    )
    print("\nArchivos generados: panama_distritos_poblacion.geojson / .csv")
