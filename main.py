import streamlit as st
import pandas as pd
import os
import numpy as np
import requests

# Configuración de directorios
output_dir_colima = "datos_estaciones_colima"
output_dir_cerca = "datos_estaciones_cerca_colima"
os.makedirs(output_dir_colima, exist_ok=True)
os.makedirs(output_dir_cerca, exist_ok=True)

# Función para descargar archivo desde Google Drive
def download_file_from_google_drive(file_id, destination):
    """Descargar un archivo desde Google Drive dado su ID y destino."""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    response = session.get(url, stream=True)

    # Manejar confirmación para archivos grandes
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            url = f"https://drive.google.com/uc?export=download&confirm={value}&id={file_id}"
            response = session.get(url, stream=True)
            break

    # Descargar el archivo
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:  # Evitar escribir chunks vacíos
                f.write(chunk)

# Configuración del archivo ACE2
file_id = "1Y9b9gLF0xb0DVc8enniOnyxPXv8KZUPA"  # ID del archivo en Google Drive
file_path = "Colima_ACE2.ace2"  # Ruta destino para guardar el archivo
tile_size = (6000, 6000)  # Tamaño esperado de la matriz

# Descargar archivo ACE2 si no existe
if not os.path.exists(file_path):
    st.write("Descargando el archivo ACE2...")
    try:
        download_file_from_google_drive(file_id, file_path)
        st.success("Archivo descargado correctamente.")
    except Exception as e:
        st.error(f"Error al descargar el archivo ACE2: {e}")
        st.stop()

# Función para leer el archivo ACE2
def read_ace2(file_path, tile_size):
    """Leer archivo ACE2 y convertirlo en una matriz NumPy."""
    try:
        data = np.fromfile(file_path, dtype=np.float32)
        return data.reshape(tile_size)
    except ValueError:
        raise RuntimeError("El archivo no coincide con las dimensiones esperadas. Verifica el archivo descargado.")

# Cargar datos de elevación
try:
    elevation_data = read_ace2(file_path, tile_size)
    st.write("Datos de elevación cargados correctamente.")
except Exception as e:
    st.error(f"Error al cargar el archivo ACE2: {e}")
    st.stop()



# Listas de claves
claves_colima = [
    "C06001", "C06049", "C06076", "C06074", "C06006", "C06040", "C06010", "C06015",
    "C06024", "C06043", "C06071", "C06062", "C06014", "ARMCM", "C06008", "C06075",
    "C06056", "C06020", "C06002", "C06009", "C06041", "C06021", "C06073", "C06012",
    "C06042", "C06016", "CHNCM", "CSTCM", "LPSCM", "ASLCM", "ORTCM", "RDRCM", 
    "CMLCM", "PNTCM", "SCHCM", "CQMCM", "LAECM", "CMTCM", "C06063", "C06004",
    "C06060", "C06064", "C06068", "C06036", "C06054", "C06018", "C06051", "C06069",
    "C06070", "C06025", "BVSCM", "CUACM", "TRPCM", "C06066", "C06039", "C06030",
    "C06048", "C06061", "C06003", "C06005", "C06053", "C06011", "C06013", "C06059",
    "C06067", "C06017", "C06022", "C06023", "C06058", "C06007", "C06052", "C06065",
    "IXHCM", "ACMCM", "CMDCM", "MNZCM", "SNTCM", "MINCM", "CLRCM", "CLLCM", 
    "CDOCM", "LDACM", "DJLCM", "TCMCM"]

claves_colima_cerca = [
    "C14008", "C14018", "MRZJL", "C14019", "C14046", "C14390", "ELCJL", "TMLJL", "C14027", "CHFJL",
    "C14148", "C14112", "C14029", "C14094", "C14043", "C14343", "C14050", "BBAJL", "C14051", "C14315",
    "VIHJL", "C14348", "C14011", "C14042", "C14086", "C14099", "C14336", "C14109", "TRJCM", "C14031",
    "C14368", "C14034", "ECAJL", "C14141", "C14095", "C14052", "NOGJL", "C14142", "C14184", "TAPJL",
    "C14005", "SLTJL", "C14322", "C14311", "C14151", "C14190", "C14024", "CPEJL", "CP4JL", "CP3JL",
    "CP1JL", "C14067", "HIGJL", "RTOJL", "C14387", "C14350", "C14155", "C14022", "C14118", "C14342",
    "ALCJL", "C14395", "IVAJL", "C14197", "C14158", "C14007", "C14079", "C14117", "C14166", "C14170",
    "C14120", "C14352", "C14030", "CGZJL"
]

claves_jalisco = {'BBAJL', 'C14008', 'C14018', 'C14019', 'C14027', 'C14029', 'C14043',
                  'C14046', 'C14050', 'C14051', 'C14094', 'C14112', 'C14148', 'C14343',
                  'C14390', 'CHFJL', 'ELCJL', 'MRZJL', 'TMLJL'}
claves_michoacan = {'ALCJL', 'C14005', 'C14007', 'C14011', 'C14022', 'C14024', 'C14030',
                    'C14031', 'C14034', 'C14042', 'C14052', 'C14067', 'C14079', 'C14086',
                    'C14095', 'C14099', 'C14109', 'C14117', 'C14118', 'C14120', 'C14141',
                    'C14142', 'C14151', 'C14155', 'C14158', 'C14166', 'C14170', 'C14184',
                    'C14190', 'C14197', 'C14311', 'C14315', 'C14322', 'C14336', 'C14342',
                    'C14348', 'C14350', 'C14352', 'C14368', 'C14387', 'C14395', 'CGZJL',
                    'CP1JL', 'CP3JL', 'CP4JL', 'CPEJL', 'ECAJL', 'HIGJL', 'IVAJL', 'NOGJL',
                    'RTOJL', 'SLTJL', 'TAPJL', 'TRJCM', 'VIHJL'}


# Combinar todas las claves
claves = claves_colima + claves_colima_cerca

# Columnas numéricas disponibles
columnas_numericas = [
    ' Precipitación(mm)', ' Temperatura Media(ºC)', 
    ' Temperatura Máxima(ºC)', ' Temperatura Mínima(ºC)', ' Evaporación(mm)'
]

# Función para obtener años disponibles
def obtener_anos_disponibles(claves, output_dirs):
    anos_disponibles = set()
    for output_dir in output_dirs:
        for clave in claves:
            archivo = os.path.join(output_dir, f"{clave}_df.csv")
            if os.path.exists(archivo):
                df = pd.read_csv(archivo)
                df['Fecha'] = pd.to_datetime(df['Fecha'], format='%Y/%m/%d', errors='coerce')
                anos = df['Fecha'].dt.year.dropna().unique()
                anos_disponibles.update(anos)
    return sorted(anos_disponibles)

# Función para obtener la elevación desde el archivo ACE2
def obtener_elevacion(lat, lon, tile_size, elevation_data):
    """
    Obtiene la elevación en kilómetros desde el archivo ACE2 usando latitud y longitud.
    """
    # Calcular índices en la matriz ACE2 basados en la latitud y longitud
    lat_idx = int(max(0, min((30 - lat) * tile_size[0] / 15, tile_size[0] - 1)))  # Ajusta para el rango ACE2
    lon_idx = int(max(0, min((lon + 105) * tile_size[1] / 15, tile_size[1] - 1)))  # Ajusta para el rango ACE2
    elevacion = elevation_data[lat_idx, lon_idx] / 1000  # Convertir de metros a kilómetros
    return max(0, elevacion)  # Evitar valores negativos


# Función para procesar datos
#def procesar_datos(ano, mes, claves, output_dirs):
#    datos_procesados = []

#    for output_dir in output_dirs:
#        for clave in claves:
#            archivo = os.path.join(output_dir, f"{clave}_df.csv")
#            if os.path.exists(archivo):
#                df = pd.read_csv(archivo)
#                df['Fecha'] = pd.to_datetime(df['Fecha'], format='%Y/%m/%d', errors='coerce')
#                df['ano'] = df['Fecha'].dt.year
#                df['mes'] = df['Fecha'].dt.month

#                # Filtrar por año y mes
#                df_filtrado = df[df['ano'] == ano]
#                if mes:
#                    df_filtrado = df_filtrado[df_filtrado['mes'] == mes]

#                # Limpiar columnas numéricas y calcular promedios
#                promedios = {}
#                for col in columnas_numericas:
#                    if col in df_filtrado.columns:
#                        df_filtrado[col] = pd.to_numeric(df_filtrado[col].astype(str).str.replace('[^0-9.]', '', regex=True), errors='coerce')
#                        promedios[col] = df_filtrado[col].mean()

#                # Obtener latitud y longitud
#                if 'Latitud' in df.columns and 'Longitud' in df.columns:
#                    latitud = df['Latitud'].iloc[0]
#                    longitud = df['Longitud'].iloc[0]
#                    elevacion = obtener_elevacion(latitud, longitud, tile_size, elevation_data)
#                else:
#                    latitud = np.nan
#                    longitud = np.nan
#                    elevacion = np.nan

#                # Determinar el estado de la estación
#                if clave in claves_colima:
#                    estado = "Colima"
#                elif clave in claves_jalisco:
#                    estado = "Jalisco"
#                elif clave in claves_michoacan:
#                    estado = "Michoacán"
#                else:
#                    estado = "Desconocido"

#                # Agregar datos al resultado
#                estacion_data = {
#                    'Clave': clave,
#                    'Estado': estado,
#                    'Latitud': latitud,
#                    'Longitud': longitud,
#                    'Elevación (km)': elevacion  # Agregar elevación
#                }
#                estacion_data.update(promedios)
#                datos_procesados.append(estacion_data)


#    return pd.DataFrame(datos_procesados)


def procesar_datos(ano, mes, claves, output_dirs):
    datos_procesados = []

    for output_dir in output_dirs:
        for clave in claves:
            archivo = os.path.join(output_dir, f"{clave}_df.csv")
            if os.path.exists(archivo):
                df = pd.read_csv(archivo)
                df['Fecha'] = pd.to_datetime(df['Fecha'], format='%Y/%m/%d', errors='coerce')
                df['ano'] = df['Fecha'].dt.year
                df['mes'] = df['Fecha'].dt.month

                # Filtrar por año y mes
                df_filtrado = df[df['ano'] == ano]
                if mes and mes != 0:  # Si se selecciona un mes específico
                    df_filtrado = df_filtrado[df_filtrado['mes'] == mes]

                # Si no hay datos para el año (o mes) seleccionado, omitir esta estación
                if df_filtrado.empty:
                    continue

                # Limpiar columnas numéricas y calcular promedios
                promedios = {}
                for col in columnas_numericas:
                    if col in df_filtrado.columns:
                        df_filtrado[col] = pd.to_numeric(df_filtrado[col].astype(str).str.replace('[^0-9.]', '', regex=True), errors='coerce')
                        promedios[col] = df_filtrado[col].mean()

                # Obtener latitud y longitud
                if 'Latitud' in df.columns and 'Longitud' in df.columns:
                    latitud = df['Latitud'].iloc[0]
                    longitud = df['Longitud'].iloc[0]
                    elevacion = obtener_elevacion(latitud, longitud, tile_size, elevation_data)
                else:
                    latitud = np.nan
                    longitud = np.nan
                    elevacion = np.nan

                # Determinar el estado de la estación
                if clave in claves_colima:
                    estado = "Colima"
                elif clave in claves_jalisco:
                    estado = "Jalisco"
                elif clave in claves_michoacan:
                    estado = "Michoacán"
                else:
                    estado = "Desconocido"

                # Agregar datos al resultado
                estacion_data = {
                    'Clave': clave,
                    'Estado': estado,
                    'Latitud': latitud,
                    'Longitud': longitud,
                    'Elevación (km)': elevacion  # Agregar elevación
                }
                estacion_data.update(promedios)
                datos_procesados.append(estacion_data)

    return pd.DataFrame(datos_procesados)



# Configuración de Streamlit
st.title("Análisis de Datos Meteorológicos")

import streamlit as st

import streamlit as st

st.markdown("""
<div style="text-align: justify;">
<h3>Mapas Meteorológicos del Estado de Colima</h3>

<p>En esta sección se muestran los mapas meteorológicos para el estado de Colima. Dichos mapas fueron creados combinando datos de estaciones meteorológicas disponibles con modelos para el cálculo de la radiación solar y la corrección por la altura. A continuación se detallan las fuentes:</p>

<ul>
  <li><b>Datos Meteorológicos:</b> Los datos de precipitación (medida en milímetros), temperatura (medida en grados Celsius) y evaporación (medida en milímetros) se obtuvieron de la base de datos de estaciones meteorológicas de la 
  <a href="https://sih.conagua.gob.mx/climas.html" target="_blank">Comisión Nacional del Agua (CONAGUA)</a>.</li>
  <li><b>Elevación sobre el Nivel del Mar:</b> La elevación sobre el nivel del mar (medida en kilómetros) se obtuvo del 
  <a href="https://sedac.ciesin.columbia.edu/mapping/ace2/?_ga=2.64821862.877322575.1732493587-819847203.1710446044" target="_blank">Modelo Digital de Elevación Global de la NASA</a>.</li>
  <li><b>Radiación Solar:</b> Los valores para la radiación en la atmósfera superior se calcularon a partir de las ecuaciones para la radiación solar descritas por el 
  <a href="https://www.redalyc.org/pdf/4455/445543787010.pdf" target="_blank">modelo de Bristow y Campbell</a>. Para la corrección por la atmósfera se ha utilizado el valor de referencia de un aumento del 12% por cada kilómetro sobre el nivel del mar.</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: justify;">
<p>En el siguiente menú puede seleccionar el mes y año para visualizar los valores promedio de los parámetros climáticos y la radiación solar (si quiere ver el promedio anual, seleccione la opción <b>"Todo el año"</b>). Si desea descargar la base de datos, coloque el cursor en la esquina superior derecha y seleccione la opción de <b>"Descargar como CSV"</b> o presione el botón <b>"Descargar"</b> ubicado debajo de la base de datos.</p>
</div>
""", unsafe_allow_html=True)

# Directorios de entrada
output_dirs = [output_dir_colima, output_dir_cerca]

# Obtener años disponibles
anos_disponibles = obtener_anos_disponibles(claves, output_dirs)
if not anos_disponibles:
    st.error("No se encontraron datos disponibles.")
    st.stop()

# Menú desplegable para seleccionar año y mes
ano = st.selectbox("Selecciona el año", options=anos_disponibles)
meses = {0: "Todo el año", 1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
         7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"}
mes = st.selectbox("Selecciona el mes", options=list(meses.keys()), format_func=lambda x: meses[x])

# Procesar datos seleccionados
df_resultado = procesar_datos(ano, mes if mes != 0 else None, claves, output_dirs)

# Parámetros para radiación solar
S0 = 1361  # Constante solar (W/m²)
Ta = 0.75  # Transmisión atmosférica promedio
k = 0.12   # Incremento de radiación por km de altitud

def calculate_annual_radiation(latitude, altitude):
    """Calcular radiación solar promedio anual considerando declinación solar y ángulo horario."""
    total_radiation = 0
    for day in range(1, 366):
        # Calcular declinación solar
        declination = 23.45 * np.sin(np.radians((360 / 365) * (day - 81)))
        declination_rad = np.radians(declination)
        
        # Convertir latitud a radianes
        latitude_rad = np.radians(latitude)
        
        # Calcular ángulo horario del amanecer/atardecer
        h_s = np.arccos(-np.tan(latitude_rad) * np.tan(declination_rad))
        
        # Calcular radiación diaria
        daily_radiation = (
            S0 * Ta * (1 + k * altitude) * 
            (np.cos(latitude_rad) * np.cos(declination_rad) * np.sin(h_s) +
             h_s * np.sin(latitude_rad) * np.sin(declination_rad))
        )
        
        total_radiation += max(0, daily_radiation)  # Evitar valores negativos

    return total_radiation / 365  # Promedio anual


def calculate_monthly_radiation(latitude, altitude, days_in_month):
    """Calcular radiación solar promedio mensual."""
    total_radiation = 0
    for day in range(1, days_in_month + 1):
        declination = 23.45 * np.sin(np.radians((360 / 365) * (day - 81)))
        declination_rad = np.radians(declination)
        latitude_rad = np.radians(latitude)
        h_s = np.arccos(-np.tan(latitude_rad) * np.tan(declination_rad))
        daily_radiation = (
            S0 * Ta * (1 + k * altitude) *
            (np.cos(latitude_rad) * np.cos(declination_rad) * np.sin(h_s) +
             h_s * np.sin(latitude_rad) * np.sin(declination_rad))
        )
        total_radiation += max(0, daily_radiation)  # Evitar valores negativos
    return total_radiation / days_in_month

# Actualizar el DataFrame con la radiación solar
if not df_resultado.empty:
    radiaciones = []
    for _, row in df_resultado.iterrows():
        latitud = row['Latitud']
        elevacion = row['Elevación (km)']
        if mes == 0:  # Todo el año
            radiacion = calculate_annual_radiation(latitud, elevacion)
        else:  # Mes específico
            # Días en cada mes (no considera años bisiestos)
            dias_por_mes = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            dias_mes = dias_por_mes[mes - 1]
            radiacion = calculate_monthly_radiation(latitud, elevacion, dias_mes)
        radiaciones.append(radiacion)

    # Agregar la columna de radiación al DataFrame
    df_resultado['Radiación Solar Promedio (W/m²)'] = radiaciones


# Parámetros para corrección
gradiente_temperatura = -6.5  # °C/km, gradiente ambiental típico

# Corregir valores de radiación y temperatura en función de la elevación
if not df_resultado.empty:
    temperaturas_corregidas = []
    radiaciones_corregidas = []

    for _, row in df_resultado.iterrows():
        elevacion = row['Elevación (km)']

        # Corregir temperatura media
        temp_media_original = row[' Temperatura Media(ºC)'] if ' Temperatura Media(ºC)' in row else np.nan
        temp_media_corregida = temp_media_original + (elevacion * gradiente_temperatura) if not pd.isna(temp_media_original) else np.nan
        temperaturas_corregidas.append(temp_media_corregida)

        # Corregir radiación solar
        radiacion_original = row['Radiación Solar Promedio (W/m²)'] if 'Radiación Solar Promedio (W/m²)' in row else np.nan
        radiacion_corregida = radiacion_original * (1 + k * elevacion) if not pd.isna(radiacion_original) else np.nan
        radiaciones_corregidas.append(radiacion_corregida)

    df_resultado['Radiación Solar Corregida (W/m²)'] = radiaciones_corregidas
    df_2=df_resultado.copy()


# Mostrar resultados
if not df_resultado.empty:
    st.write(f"Datos procesados para {meses[mes]} del año {ano}:")
    st.dataframe(df_resultado)
else:
    st.write("No se encontraron datos para el período seleccionado.")


# Determinar el nombre del archivo
if mes:  # Si hay un mes seleccionado
    filename = f"datos_climaticos_para_{mes}_{ano}.csv"
else:  # Si es todo el año
    filename = f"datos_climaticos_para_{ano}.csv"

# Botón de descarga
csv = df_resultado.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Descargar",
    data=csv,
    file_name=filename,
    mime="text/csv"
)

import plotly.express as px
import plotly.graph_objects as go
import json
import matplotlib as plt


# Definir una escala coolwarm personalizada
coolwarm_scale = [
    [0.0, 'rgb(59,76,192)'],  # Azul oscuro
    [0.35, 'rgb(116,173,209)'],  # Azul claro
    [0.5, 'rgb(221,221,221)'],  # Blanco/neutral
    [0.65, 'rgb(244,109,67)'],  # Naranja claro
    [1.0, 'rgb(180,4,38)']  # Rojo oscuro
]

# Crear el esquema de colores 'coolwarm'
coolwarm_colorscale = plt.cm.coolwarm(np.linspace(0, 1, 256))
coolwarm_colorscale = [
    [i / 255.0, f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"]
    for i, (r, g, b, _) in enumerate(coolwarm_colorscale)
]

# Cargar el archivo GeoJSON (Colima.JSON) para referencia del mapa
try:
    with open('Colima.JSON', 'r', encoding='latin-1') as file:
        colima_geojson = json.load(file)
except Exception as e:
    st.error(f"No se pudo cargar el archivo GeoJSON: {e}")
    st.stop()

st.subheader("Mapa con datos de estaciones de la CONAGUA")

st.markdown("""
<div style="text-align: justify;">
<p>En la siguiente gráfica se muestran las estaciones del estado de Colima y zonas circundantes que registraron datos climáticos para el periodo de tiempo seleccionado (los colores representan una aproximación al valor registrado por cada estación).</p>
</div>
""", unsafe_allow_html=True)

import plotly.io as pio

# Mostrar mapa con estaciones
if not df_resultado.empty:
    # Menú desplegable para seleccionar la columna numérica a graficar
    columna_grafico = st.selectbox("Selecciona la columna para el color del mapa", options=columnas_numericas)

    # Filtrar estaciones con valores NaN en la columna seleccionada
    if columna_grafico in df_resultado.columns:
        df_filtrado = df_resultado.dropna(subset=[columna_grafico])
        #df_filtrado = df_resultado
        if not df_filtrado.empty:
            # Crear el mapa base con las estaciones
            fig = px.scatter_mapbox(
                df_filtrado,
                lat="Latitud",
                lon="Longitud",
                color=columna_grafico,
                hover_name="Clave",
                hover_data=["Estado", columna_grafico],
                title=f"Mapa de estaciones en Colima y alrededores ({columna_grafico.strip()} para el año {ano}, mes {mes})",
                mapbox_style="carto-positron",
                center={"lat": 19.0, "lon": -104.0},  # Ajusta el centro del mapa según sea necesario
                zoom=8,
                width=1000,
                height=600,
                color_continuous_scale=coolwarm_colorscale   # Usar escala coolwarm personalizada
            )

            # Cambiar tamaño de los puntos
            fig.update_traces(marker=dict(size=12))  # Ajusta el tamaño como desees

            # Añadir los polígonos de los municipios como trazas adicionales
            for feature in colima_geojson["features"]:
                geometry = feature["geometry"]
                properties = feature["properties"]

                # Excluir islas si es necesario
                if "isla" not in properties.get("name", "").lower():
                    if geometry["type"] == "Polygon":
                        for coordinates in geometry["coordinates"]:
                            x_coords, y_coords = zip(*coordinates)
                            fig.add_trace(
                                go.Scattermapbox(
                                    lon=x_coords,
                                    lat=y_coords,
                                    mode="lines",
                                    line=dict(color="black", width=2),
                                    showlegend=False
                                )
                            )
                    elif geometry["type"] == "MultiPolygon":
                        for polygon in geometry["coordinates"]:
                            for coordinates in polygon:
                                x_coords, y_coords = zip(*coordinates)
                                fig.add_trace(
                                    go.Scattermapbox(
                                        lon=x_coords,
                                        lat=y_coords,
                                        mode="lines",
                                        line=dict(color="black", width=2),
                                        showlegend=False
                                    )
                                )

            # Mostrar el mapa
            st.plotly_chart(fig, use_container_width=True)
            # Exportar el gráfico a un archivo PNG con 300 DPI
            #fig.write_image("chart_300dpi.png", format="png", width=1200, height=900, scale=3)
            # Exportar a SVG
            #fig.write_image("chart.svg", format="svg")

            # Convertir a PNG con 300 DPI usando cairosvg
            #cairosvg.svg2png(url="chart.svg", write_to="chart_300dpi.png", dpi=300)


        else:
            st.warning(f"No hay estaciones con datos válidos en la columna '{columna_grafico}'.")
    else:
        st.warning("La columna seleccionada no está disponible en el DataFrame.")
else:
    st.write("No hay datos disponibles para mostrar en el mapa.")

#
#import streamlit as st
#import plotly.express as px
#import plotly.graph_objects as go
#import numpy as np
#import json
#import matplotlib.pyplot as plt

# Crear el esquema de colores 'coolwarm'
#coolwarm_colorscale = plt.cm.coolwarm(np.linspace(0, 1, 256))
#coolwarm_colorscale = [
#    [i / 255.0, f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"]
#    for i, (r, g, b, _) in enumerate(coolwarm_colorscale)
#]

## Cargar el archivo GeoJSON (Colima.JSON) para referencia del mapa
#try:
#    with open('Colima.JSON', 'r', encoding='latin-1') as file:
#        colima_geojson = json.load(file)
#except Exception as e:
#    st.error(f"No se pudo cargar el archivo GeoJSON: {e}")
#    st.stop()

## Mostrar mapa con estaciones
#if not df_resultado.empty:
#    # Menú desplegable para seleccionar la columna numérica a graficar
#    columna_grafico = st.selectbox("Selecciona columna para el color del mapa", options=columnas_numericas)

#    # Checkbox para mostrar u ocultar estaciones inactivas (valores NaN en la columna seleccionada)
#    mostrar_inactivas = st.checkbox("Mostrar estaciones inactivas (valores NaN)", value=True)

#    if columna_grafico in df_resultado.columns:
#        if mostrar_inactivas:
#            df_filtrado = df_resultado  # Incluir todas las estaciones
#        else:
#            df_filtrado = df_resultado.dropna(subset=[columna_grafico])  # Excluir estaciones inactivas

#        if not df_filtrado.empty:
#            # Crear el mapa base con las estaciones
#            fig = px.scatter_mapbox(
#                df_filtrado,
#                lat="Latitud",
#                lon="Longitud",
#                color=columna_grafico,
#                hover_name="Clave",
#                hover_data=["Estado", columna_grafico],
#                title=f"Mapa de estaciones en Colima y alrededores ({columna_grafico.strip()})",
#                mapbox_style="carto-positron",
#                center={"lat": 19.0, "lon": -104.0},  # Ajusta el centro del mapa según sea necesario
#                zoom=8,
#                width=1000,
#                height=600,
#                color_continuous_scale=coolwarm_colorscale  # Usar escala coolwarm personalizada
#            )

#            # Cambiar tamaño de los puntos
#            fig.update_traces(marker=dict(size=12))  # Ajusta el tamaño como desees

#            # Añadir los polígonos de los municipios como trazas adicionales
#            for feature in colima_geojson["features"]:
#                geometry = feature["geometry"]
#                properties = feature["properties"]

#                # Excluir islas si es necesario
#                if "isla" not in properties.get("name", "").lower():
#                    if geometry["type"] == "Polygon":
#                        for coordinates in geometry["coordinates"]:
#                            x_coords, y_coords = zip(*coordinates)
#                            fig.add_trace(
#                                go.Scattermapbox(
#                                    lon=x_coords,
#                                    lat=y_coords,
#                                    mode="lines",
#                                    line=dict(color="black", width=2),
#                                    showlegend=False
#                                )
#                            )
#                    elif geometry["type"] == "MultiPolygon":
#                        for polygon in geometry["coordinates"]:
#                            for coordinates in polygon:
#                                x_coords, y_coords = zip(*coordinates)
#                                fig.add_trace(
#                                    go.Scattermapbox(
#                                        lon=x_coords,
#                                        lat=y_coords,
#                                        mode="lines",
#                                        line=dict(color="black", width=2),
#                                        showlegend=False
#                                    )
#                                )

#            # Mostrar el mapa
#            st.plotly_chart(fig, use_container_width=True)
#        else:
#            st.warning(f"No hay estaciones con datos válidos en la columna '{columna_grafico}'.")
#    else:
#        st.warning("La columna seleccionada no está disponible en el DataFrame.")
#else:
#    st.write("No hay datos disponibles para mostrar en el mapa.")


#




import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objects as go
import json

# Definir una escala coolwarm personalizada
coolwarm_scale = [
    [0.0, 'rgb(59,76,192)'],  # Azul oscuro
    [0.35, 'rgb(116,173,209)'],  # Azul claro
    [0.5, 'rgb(221,221,221)'],  # Blanco/neutral
    [0.65, 'rgb(244,109,67)'],  # Naranja claro
    [1.0, 'rgb(180,4,38)']  # Rojo oscuro
]

# Cargar el archivo GeoJSON (Colima.JSON) para referencia del mapa
try:
    with open('Colima.JSON', 'r', encoding='latin-1') as file:
        colima_geojson = json.load(file)
except Exception as e:
    st.error(f"No se pudo cargar el archivo GeoJSON: {e}")
    st.stop()

st.subheader("Mapa con valores interpolados")

st.markdown("""
<div style="text-align: justify;">
<p>Se utilizaron distintos métodos de interpolación para generar un mapa continuo del parámetro seleccionado. A continuación, puede seleccionar entre los distintos métodos y el mapa se generará a partir de la interpolación de los valores de las estaciones cercanas.</p>
</div>
""", unsafe_allow_html=True)


# Mostrar mapa con estaciones
if not df_resultado.empty:
    # Menú desplegable para seleccionar la columna numérica a graficar
    columna_grafico = st.selectbox("Selecciona el parámetro para graficar", options=columnas_numericas)

    # Filtrar estaciones con valores NaN en la columna seleccionada
    if columna_grafico in df_resultado.columns:
        df_filtrado = df_resultado.dropna(subset=[columna_grafico])

        if not df_filtrado.empty:
            # Preparar datos para interpolación
            latitudes = df_filtrado["Latitud"].values
            longitudes = df_filtrado["Longitud"].values
            valores = df_filtrado[columna_grafico].values

            margen_long = 0.08 * (longitudes.max() - longitudes.min())  # 5% del rango en longitud
            margen_lat = 0.08 * (latitudes.max() - latitudes.min())    # 5% del rango en latitud

            grid_lon, grid_lat = np.meshgrid(
                np.linspace(longitudes.min() - margen_long, longitudes.max() + margen_long, 100),
                np.linspace(latitudes.min() - margen_lat, latitudes.max() + margen_lat, 100)
            )

            # Interpolar los datos
            metodo_interpolacion = st.selectbox("Selecciona el método de interpolación", ["Linear", "Nearest", "IDW"])
            if metodo_interpolacion in ["Linear", "Nearest"]:
                interpolados = griddata(
                    (longitudes, latitudes),
                    valores,
                    (grid_lon, grid_lat),
                    method=metodo_interpolacion.lower()
                )
            elif metodo_interpolacion == "IDW":
                # Implementación básica de IDW
                def idw_interpolation(x, y, values, xi, yi):
                    weights = 1 / np.sqrt((x - xi) ** 2 + (y - yi) ** 2 + 1e-10)
                    return np.sum(weights * values) / np.sum(weights)

                interpolados = np.zeros_like(grid_lon)
                for i in range(grid_lon.shape[0]):
                    for j in range(grid_lon.shape[1]):
                        interpolados[i, j] = idw_interpolation(longitudes, latitudes, valores, grid_lon[i, j], grid_lat[i, j])

            # Crear la figura
            fig = go.Figure()

            # Añadir contornos de valores interpolados
            fig.add_trace(
                go.Contour(
                    z=interpolados,
                    x=grid_lon[0],
                    y=grid_lat[:, 0],
                    colorscale=coolwarm_colorscale,
                    line=dict(color="black", width=1.0),  # Líneas más gruesas
                    opacity=0.7,
                    contours=dict(
                        coloring="fill",  # Las zonas entre curvas tienen color
                        showlabels=True,  # Mostrar etiquetas en los contornos
                        labelfont=dict(size=10, color="black")
                    ),
                    colorbar=dict(
                        title=f"{columna_grafico.strip()}",
                        len=0.8  # Reducir la longitud de la barra de color
                    ),
                    name=f"Interpolación ({columna_grafico.strip()})"
                )
            )

            # Añadir puntos de las estaciones
            fig.add_trace(
                go.Scatter(
                    x=longitudes,
                    y=latitudes,
                    mode="markers",
                    marker=dict(
                        size=10,
                        color="black",
                        opacity=1.0,
                        #colorscale=coolwarm_scale,
                        showscale=False  # Ocultar barra de colores adicional
                    ),
                    text=df_filtrado["Clave"],
                    hoverinfo="text",
                    name="Estaciones"
                )
            )

            # Añadir contornos de los municipios
            for feature in colima_geojson["features"]:
                geometry = feature["geometry"]
                properties = feature["properties"]

                if "isla" not in properties.get("name", "").lower():
                    if geometry["type"] == "Polygon":
                        for coordinates in geometry["coordinates"]:
                            x_coords, y_coords = zip(*coordinates)
                            fig.add_trace(
                                go.Scatter(
                                    x=x_coords,
                                    y=y_coords,
                                    mode="lines",
                                    line=dict(color="black", width=2),
                                    showlegend=False
                                )
                            )
                    elif geometry["type"] == "MultiPolygon":
                        for polygon in geometry["coordinates"]:
                            for coordinates in polygon:
                                x_coords, y_coords = zip(*coordinates)
                                fig.add_trace(
                                    go.Scatter(
                                        x=x_coords,
                                        y=y_coords,
                                        mode="lines",
                                        line=dict(color="black", width=2),
                                        showlegend=False
                                    )
                                )

            # Configuración del diseño
            fig.update_layout(
                title=f"Mapa de estaciones y contornos interpolados ({columna_grafico.strip()})",
                xaxis_title="Longitud",
                yaxis_title="Latitud",
                margin=dict(l=0, r=0, t=50, b=0)
            )

            fig.update_layout(
                xaxis=dict(
                    title="Longitud",
                    titlefont=dict(size=14, family="Arial"),
                    tickfont=dict(size=12, family="Arial"),
                    range=[-104.7, -103.3]  # Ajustar los límites iniciales del eje X (Longitud)
                ),
                    yaxis=dict(
                    title="Latitud",
                    titlefont=dict(size=14, family="Arial"),
                    tickfont=dict(size=12, family="Arial"),
                    range=[18.5, 19.7]  # Ajustar los límites iniciales del eje Y (Latitud)
                ),
                geo=dict(
                    center=dict(
                        lon=-104.0,  # Longitud central
                        lat=19.3     # Latitud central
                    ),
                    projection_scale=1  # Ajustar el zoom inicial
                ),
                margin=dict(l=20, r=20, t=50, b=20) 
            )

            fig.update_layout(
                width=1000,  # Ancho del gráfico
                height=600,  # Altura del gráfico
                title=f"Mapa de estaciones y contornos interpolados ({columna_grafico.strip()} para el año {ano}, mes {mes})",
                xaxis_title="Longitud",
                yaxis_title="Latitud",
                margin=dict(l=0, r=0, t=50, b=0)  # Márgenes del gráfico
            )


            # Mostrar el gráfico
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No hay estaciones con datos válidos en la columna '{columna_grafico}'.")
    else:
        st.warning("La columna seleccionada no está disponible en el DataFrame.")
else:
    st.write("No hay datos disponibles para mostrar en el mapa.")

import streamlit as st

# Expander con pestañas para explicar los métodos de interpolación
with st.expander("Métodos de Interpolación", expanded=True):
    tab1, tab2, tab3 = st.tabs(["Linear", "Nearest", "IDW"])
    
    with tab1:
        st.markdown("""
        ### Interpolación Linear
        Este método calcula los valores interpolados mediante un ajuste lineal entre los puntos más cercanos a la ubicación deseada. 
        Es ideal para datos que se distribuyen suavemente en el espacio, ya que evita cambios bruscos entre las zonas interpoladas.

        **Implementación en el código:**
        - En el código, se utiliza la función `griddata` del paquete `scipy.interpolate`.
        - El parámetro `method="linear"` indica que se debe realizar una interpolación lineal.
        - Se calcula una grilla regular de valores interpolados basada en las latitudes, longitudes y valores de las estaciones cercanas.
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        ### Interpolación Nearest (Vecino más cercano)
        Este método asigna el valor del punto más cercano a cada posición interpolada. 
        Es útil para datos discretos o cuando se desea mantener los valores originales sin suavizar la información.

        **Implementación en el código:**
        - En el código, también se utiliza la función `griddata` del paquete `scipy.interpolate`.
        - El parámetro `method="nearest"` asegura que el valor asignado en cada celda de la grilla corresponde al de la estación más cercana.
        - Este método es más rápido pero menos suave en comparación con la interpolación lineal.
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("""
        ### Interpolación Inverse Distance Weighting (IDW)
        Este método utiliza una fórmula basada en la distancia inversa para asignar valores interpolados.
        Los puntos más cercanos tienen mayor influencia en el valor final, mientras que los puntos más lejanos tienen menor impacto.

        **Implementación en el código:**
        - Se define una función `idw_interpolation` personalizada.
        - Para cada punto de la grilla, se calcula la distancia a todas las estaciones y se asigna un peso inversamente proporcional a esta distancia.
        - Los valores interpolados se calculan como la suma ponderada de los valores de las estaciones.
        - Este método es más computacionalmente intensivo, pero permite un control más detallado sobre la influencia de las estaciones cercanas.
        """, unsafe_allow_html=True)



import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objects as go
import json

# Parámetros para corrección
gradiente_temperatura = -6.5  # °C/km
incremento_radiacion = 0.12   # W/m²/km

## Definir una escala coolwarm personalizada
#coolwarm_scale = [
#    [0.0, 'rgb(59,76,192)'],  # Azul oscuro
#    [0.35, 'rgb(116,173,209)'],  # Azul claro
#    [0.5, 'rgb(221,221,221)'],  # Blanco/neutral
#    [0.65, 'rgb(244,109,67)'],  # Naranja claro
#    [1.0, 'rgb(180,4,38)']  # Rojo oscuro
#]

## Cargar el archivo GeoJSON
#try:
#    with open('Colima.JSON', 'r', encoding='latin-1') as file:
#        colima_geojson = json.load(file)
#except Exception as e:
#    st.error(f"No se pudo cargar el archivo GeoJSON: {e}")
#    st.stop()

## Obtener elevación interpolada para la malla
#def obtener_elevacion_interpolada(grid_lon, grid_lat, elevation_data, tile_size):
#    elevacion = np.zeros_like(grid_lon)
#    for i in range(grid_lon.shape[0]):
#        for j in range(grid_lon.shape[1]):
#            lon, lat = grid_lon[i, j], grid_lat[i, j]
#            lat_idx = int(max(0, min((30 - lat) * tile_size[0] / 15, tile_size[0] - 1)))
#            lon_idx = int(max(0, min((lon + 105) * tile_size[1] / 15, tile_size[1] - 1)))
#            elevacion[i, j] = elevation_data[lat_idx, lon_idx] / 1000  # Convertir a km
#    return elevacion

## Mostrar mapa con corrección de valores interpolados
#if not df_resultado.empty:
#    # Selección del parámetro y método de interpolación
#    columna_grafico = st.selectbox("Seleccione el parámetro para graficar", options=columnas_numericas)
#    #metodo_interpolacion = st.selectbox("Selecciona el método de interpolación", ["Linear", "Nearest", "IDW"])

#    # Filtrar estaciones válidas
#    df_filtrado = df_resultado.dropna(subset=[columna_grafico])

#    if not df_filtrado.empty:
#        latitudes = df_filtrado["Latitud"].values
#        longitudes = df_filtrado["Longitud"].values
#        valores = df_filtrado[columna_grafico].values
        
#        # Crear una malla de puntos para la interpolación con márgenes
#        margen_long = 0.08 * (longitudes.max() - longitudes.min())  # 5% del rango en longitud
#        margen_lat = 0.08 * (latitudes.max() - latitudes.min())    # 5% del rango en latitud

#        grid_lon, grid_lat = np.meshgrid(
#            np.linspace(longitudes.min() - margen_long, longitudes.max() + margen_long, 100),
#            np.linspace(latitudes.min() - margen_lat, latitudes.max() + margen_lat, 100)
#        )

#        # Interpolación inicial
#        if metodo_interpolacion in ["Linear", "Nearest"]:
#            interpolados = griddata(
#                (longitudes, latitudes),
#                valores,
#                (grid_lon, grid_lat),
#                method=metodo_interpolacion.lower()
#            )
#        elif metodo_interpolacion == "IDW":
#            def idw_interpolation(x, y, values, xi, yi):
#                weights = 1 / np.sqrt((x - xi) ** 2 + (y - yi) ** 2 + 1e-10)
#                return np.sum(weights * values) / np.sum(weights)

#            interpolados = np.zeros_like(grid_lon)
#            for i in range(grid_lon.shape[0]):
#                for j in range(grid_lon.shape[1]):
#                    interpolados[i, j] = idw_interpolation(longitudes, latitudes, valores, grid_lon[i, j], grid_lat[i, j])

#        # Obtener elevación interpolada
#        elevacion_interpolada = obtener_elevacion_interpolada(grid_lon, grid_lat, elevation_data, tile_size)

#        # Corregir valores interpolados
#        if "Temperatura" in columna_grafico:
#            valores_corregidos = interpolados + (gradiente_temperatura * elevacion_interpolada)
#        elif "Radiación" in columna_grafico:
#            valores_corregidos = interpolados * (1 + incremento_radiacion * elevacion_interpolada)
#        else:
#            valores_corregidos = interpolados  # Sin corrección para otros parámetros

#        # Crear figura
#        fig = go.Figure()

#        # Añadir contornos corregidos
#        fig.add_trace(
#            go.Contour(
#                z=valores_corregidos,
#                x=grid_lon[0],
#                y=grid_lat[:, 0],
#                colorscale=coolwarm_scale,
#                opacity=0.8,
#                line=dict(color="black", width=1.0),  # Líneas de contorno más gruesas
#                contours=dict(
#                    coloring="fill",  # Las zonas entre curvas tienen color
#                    showlabels=True,  # Mostrar etiquetas en los contornos
#                    labelfont=dict(size=10, color="black")
#                ),
#                colorbar=dict(
#                    title=f"{columna_grafico.strip()}",
#                    len=0.8  # Longitud de la barra de color
#                ),
#                name=f"Interpolación corregida ({columna_grafico.strip()})"
#            )
#        )

#        # Añadir puntos de las estaciones
#        fig.add_trace(
#            go.Scatter(
#                x=longitudes,
#                y=latitudes,
#                mode="markers",
#                marker=dict(
#                    size=10,
#                    color="black"  # Puntos negros
#                ),
#                text=df_filtrado["Clave"],
#                hoverinfo="text",
#                name="Estaciones"
#            )
#        )

#        # Añadir contornos de los municipios
#        for feature in colima_geojson["features"]:
#            geometry = feature["geometry"]
#            properties = feature["properties"]

#            if "isla" not in properties.get("name", "").lower():
#                if geometry["type"] == "Polygon":
#                    for coordinates in geometry["coordinates"]:
#                        x_coords, y_coords = zip(*coordinates)
#                        fig.add_trace(
#                            go.Scatter(
#                                x=x_coords,
#                                y=y_coords,
#                                mode="lines",
#                                line=dict(color="black", width=2),
#                                showlegend=False
#                            )
#                        )
#                elif geometry["type"] == "MultiPolygon":
#                    for polygon in geometry["coordinates"]:
#                        for coordinates in polygon:
#                            x_coords, y_coords = zip(*coordinates)
#                            fig.add_trace(
#                                go.Scatter(
#                                    x=x_coords,
#                                    y=y_coords,
#                                    mode="lines",
#                                    line=dict(color="black", width=2),
#                                    showlegend=False
#                                )
#                            )

#        fig.update_layout(
#            xaxis=dict(
#                title="Longitud",
#                titlefont=dict(size=14, family="Arial"),  # Tamaño y tipo de letra del título del eje X
#                tickfont=dict(size=12, family="Arial"),  # Tamaño y tipo de letra de las etiquetas del eje X
#            ),
#            yaxis=dict(
#                title="Latitud",
#                titlefont=dict(size=14, family="Arial"),  # Tamaño y tipo de letra del título del eje Y
#                tickfont=dict(size=12, family="Arial"),  # Tamaño y tipo de letra de las etiquetas del eje Y
#            )
#        )

#        fig.update_layout(
#            width=1000,  # Ancho del gráfico
#            height=600,  # Altura del gráfico
#            title=f"Mapa de estaciones y contornos interpolados ({columna_grafico.strip()} corregido)",
#            xaxis_title="Longitud",
#            yaxis_title="Latitud",
#            margin=dict(l=0, r=0, t=50, b=0)  # Márgenes del gráfico
#        )

#        fig.update_layout(
#            title=f"Mapa de estaciones y contornos interpolados ({columna_grafico.strip()} corregido)",
#            xaxis=dict(
#                title="Longitud",
#                titlefont=dict(size=14, family="Arial"),
#                tickfont=dict(size=12, family="Arial"),
#                range=[-104.8, -103.5]  # Centrar en Colima
#            ),
#            yaxis=dict(
#                title="Latitud",
#                titlefont=dict(size=14, family="Arial"),
#                tickfont=dict(size=12, family="Arial"),
#                range=[18.5, 19.5]  # Centrar en Colima
#            ),
#            margin=dict(l=20, r=20, t=50, b=20)
#        )


#        fig.update_layout(
#            xaxis=dict(
#                title="Longitud",
#                titlefont=dict(size=14, family="Arial"),
#                tickfont=dict(size=12, family="Arial"),
#                range=[-104.7, -103.3]  # Ajustar los límites iniciales del eje X (Longitud)
#            ),
#            yaxis=dict(
#                title="Latitud",
#                titlefont=dict(size=14, family="Arial"),
#                tickfont=dict(size=12, family="Arial"),
#                range=[18.5, 19.7]  # Ajustar los límites iniciales del eje Y (Latitud)
#            ),
#            geo=dict(
#                center=dict(
#                    lon=-104.0,  # Longitud central
#                    lat=19.3     # Latitud central
#                ),
#                projection_scale=1  # Ajustar el zoom inicial
#            ),
#            margin=dict(l=20, r=20, t=50, b=20) 
#        )




#        # Mostrar gráfico
#        st.plotly_chart(fig, use_container_width=True)
#    else:
#        st.warning("No hay estaciones válidas para la columna seleccionada.")
#else:
#    st.write("No hay datos disponibles.")


import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objects as go
import json

st.subheader("Mapa con valores corregidos para la altura")

st.markdown("""
<div style="text-align: justify;">
<p>Los siguientes gráficos muestran los valores interpolados de los parámetros seleccionados, corregidos para considerar los efectos de la altitud sobre el nivel del mar. Estas correcciones incluyen:</p>
<ul>
<li>Ajustes en las temperaturas basados en un gradiente ambiental estándar, que reduce la temperatura en función del aumento de la elevación.</li>
<li>Cálculos de la radiación solar, incluyendo un incremento del 12% por cada kilómetro sobre el nivel del mar, siguiendo el modelo descrito previamente.</li>
</ul>
<p>Estas modificaciones aseguran que los valores representados en los mapas reflejen de manera más precisa las condiciones climáticas reales ajustadas por la altitud.</p>
</div>
""", unsafe_allow_html=True)


# Parámetros para corrección
gradiente_temperatura = -6.5  # °C/km
incremento_radiacion = 0.12   # W/m²/km

# Escala de color personalizada
coolwarm_scale = [
    [0.0, 'rgb(59,76,192)'],
    [0.35, 'rgb(116,173,209)'],
    [0.5, 'rgb(221,221,221)'],
    [0.65, 'rgb(244,109,67)'],
    [1.0, 'rgb(180,4,38)']
]

# Cargar el archivo GeoJSON
try:
    with open('Colima.JSON', 'r', encoding='latin-1') as file:
        colima_geojson = json.load(file)
except Exception as e:
    st.error(f"No se pudo cargar el archivo GeoJSON: {e}")
    st.stop()

# Función para obtener elevación interpolada
def obtener_elevacion_interpolada(grid_lon, grid_lat, elevation_data, tile_size):
    elevacion = np.zeros_like(grid_lon)
    for i in range(grid_lon.shape[0]):
        for j in range(grid_lon.shape[1]):
            lon, lat = grid_lon[i, j], grid_lat[i, j]
            lat_idx = int(max(0, min((30 - lat) * tile_size[0] / 15, tile_size[0] - 1)))
            lon_idx = int(max(0, min((lon + 105) * tile_size[1] / 15, tile_size[1] - 1)))
            elevacion[i, j] = elevation_data[lat_idx, lon_idx] / 1000  # Convertir a km
    return elevacion

# Función de interpolación IDW
def idw_interpolation(x, y, values, xi, yi, power=2):
    weights = 1 / ((x - xi) ** 2 + (y - yi) ** 2 + 1e-10) ** (power / 2)
    return np.sum(weights * values) / np.sum(weights)

# Mostrar mapa con corrección de valores interpolados
if not df_resultado.empty:
    # Selección del parámetro y método de interpolación
    columna_grafico = st.selectbox(
        "Seleccionar el parámetro para graficar",
        options=columnas_numericas + [
            "Radiación Solar Promedio (W/m²)",
            "Radiación Solar Corregida (W/m²)"
        ]
    )
    metodo_interpolacion = st.selectbox(
        "Seleccionar el método de interpolación",
        ["Linear", "Nearest", "IDW"]
    )

    # Filtrar estaciones válidas
    df_filtrado = df_resultado.dropna(subset=[columna_grafico])

    if not df_filtrado.empty:
        latitudes = df_filtrado["Latitud"].values
        longitudes = df_filtrado["Longitud"].values
        valores = df_filtrado[columna_grafico].values

        # Crear una malla de puntos para la interpolación con márgenes
        margen_long = 0.08 * (longitudes.max() - longitudes.min())
        margen_lat = 0.08 * (latitudes.max() - latitudes.min())

        grid_lon, grid_lat = np.meshgrid(
            np.linspace(longitudes.min() - margen_long, longitudes.max() + margen_long, 100),
            np.linspace(latitudes.min() - margen_lat, latitudes.max() + margen_lat, 100)
        )

        # Realizar la interpolación
        if metodo_interpolacion in ["Linear", "Nearest"]:
            interpolados = griddata(
                (longitudes, latitudes),
                valores,
                (grid_lon, grid_lat),
                method=metodo_interpolacion.lower()
            )
        elif metodo_interpolacion == "IDW":
            interpolados = np.zeros_like(grid_lon)
            for i in range(grid_lon.shape[0]):
                for j in range(grid_lon.shape[1]):
                    interpolados[i, j] = idw_interpolation(longitudes, latitudes, valores, grid_lon[i, j], grid_lat[i, j])

        # Obtener elevaciones interpoladas
        elevacion_interpolada = obtener_elevacion_interpolada(grid_lon, grid_lat, elevation_data, tile_size)

        # Corregir valores interpolados
        if "Temperatura" in columna_grafico:
            valores_corregidos = interpolados + (gradiente_temperatura * elevacion_interpolada)
        elif "Radiación" in columna_grafico:
            valores_corregidos = interpolados * (1 + 0.0*incremento_radiacion * elevacion_interpolada)
        else:
            valores_corregidos = interpolados

        # Diccionario para las unidades según el parámetro
        unidades = {
            " Temperatura Media(ºC)": "ºC",
            " Temperatura Máxima(ºC)": "ºC",
            " Temperatura Mínima(ºC)": "ºC",
            " Precipitación(mm)": "mm",
            " Evaporación(mm)": "mm",
            "Radiación Solar Promedio (W/m²)": "W/m²",
            "Radiación Solar Corregida (W/m²)": "W/m²"
        }

        # Crear la figura
        fig = go.Figure()

        # Añadir contornos corregidos
        fig.add_trace(
            go.Contour(
                z=valores_corregidos,
                x=grid_lon[0],
                y=grid_lat[:, 0],
                colorscale=coolwarm_colorscale,
                opacity=0.7,
                line=dict(color="black", width=1.0),  # Líneas de contorno más gruesas
                contours=dict(
                    coloring="fill",
                    showlabels=True,
                    labelfont=dict(size=10, color="black")
                ),
                colorbar=dict(
                    title=unidades.get(columna_grafico, ""),  # Solo las unidades
                    len=0.8,
                    thickness=20,
                    x=1.1,
                    y=0.5
                ),
                name=f"Interpolación corregida ({columna_grafico.strip()})"
            )
        )


        # Añadir puntos de las estaciones
        fig.add_trace(
            go.Scatter(
                x=longitudes,
                y=latitudes,
                mode="markers",
                marker=dict(
                    size=10,
                    color="black"
                ),
                text=df_filtrado["Clave"],
                hoverinfo="text",
                name="Estaciones"
            )
        )

        # Añadir contornos de los municipios
        for feature in colima_geojson["features"]:
            geometry = feature["geometry"]
            properties = feature["properties"]

            if "isla" not in properties.get("name", "").lower():
                if geometry["type"] == "Polygon":
                    for coordinates in geometry["coordinates"]:
                        x_coords, y_coords = zip(*coordinates)
                        fig.add_trace(
                            go.Scatter(
                                x=x_coords,
                                y=y_coords,
                                mode="lines",
                                line=dict(color="black", width=2),
                                showlegend=False
                            )
                        )
                elif geometry["type"] == "MultiPolygon":
                    for polygon in geometry["coordinates"]:
                        for coordinates in polygon:
                            x_coords, y_coords = zip(*coordinates)
                            fig.add_trace(
                                go.Scatter(
                                    x=x_coords,
                                    y=y_coords,
                                    mode="lines",
                                    line=dict(color="black", width=2),
                                    showlegend=False
                                )
                            )

        # Configuración del diseño
        fig.update_layout(
            title=f"Mapa de estaciones y contornos interpolados ({columna_grafico.strip()} para el año {ano}, mes {mes})",
            xaxis=dict(
                title="Longitud",
                titlefont=dict(size=14, family="Arial"),
                tickfont=dict(size=12, family="Arial"),
                range=[-104.7, -103.3]
            ),
            yaxis=dict(
                title="Latitud",
                titlefont=dict(size=14, family="Arial"),
                tickfont=dict(size=12, family="Arial"),
                range=[18.5, 19.7]
            ),
            width=1000,
            height=600,
            margin=dict(l=20, r=20, t=50, b=20)
        )

        # Mostrar gráfico
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No hay estaciones válidas para la columna seleccionada.")
else:
    st.write("No hay datos disponibles.")


# Nueva sección para cálculo de interpolación en un punto específico
st.subheader("Interpolación de parámetros en un punto específico")

st.markdown("""
<div style="text-align: justify;">
<p>En esta sección, puede ingresar las coordenadas de un lugar (latitud y longitud) o seleccionar una de las capitales municipales de Colima. El sistema calculará los valores interpolados para el parámetro seleccionado utilizando cada uno de los métodos de interpolación disponibles.</p>
</div>
""", unsafe_allow_html=True)

# Opciones de entrada
opcion_punto = st.radio(
    "Seleccione la forma de definir el punto:",
    options=["Ingresar coordenadas", "Seleccionar capital municipal"]
)

# Diccionario con las capitales municipales y sus coordenadas
capitales_municipales = {
    "Colima": (19.2433, -103.7247),
    "Villa de Álvarez": (19.2673, -103.7377),
    "Manzanillo": (19.0561, -104.3188),
    "Tecomán": (18.9092, -103.8770),
    "Comala": (19.3278, -103.7578),
    "Coquimatlán": (19.1801, -103.8181),
    "Armería": (18.9398, -103.9632),
    "Minatitlán": (19.3830, -104.0475),
    "Cuauhtémoc": (19.2363, -103.6618),
    "Ixtlahuacán": (18.8955, -103.7260)
}

if opcion_punto == "Ingresar coordenadas":
    latitud_punto = st.number_input("Ingrese la latitud", value=19.0, step=0.01)
    longitud_punto = st.number_input("Ingrese la longitud", value=-104.0, step=0.01)
elif opcion_punto == "Seleccionar capital municipal":
    capital_seleccionada = st.selectbox("Seleccione la capital municipal", options=capitales_municipales.keys())
    latitud_punto, longitud_punto = capitales_municipales[capital_seleccionada]

# Seleccionar el parámetro y método de interpolación
parametro_seleccionado = st.selectbox(
    "Seleccione el parámetro climático",
    options=columnas_numericas + [
        "Radiación Solar Promedio (W/m²)",
        "Radiación Solar Corregida (W/m²)"
    ]
)

# Cálculo de valores interpolados
if st.button("Calcular interpolación"):
    if not df_resultado.empty:
        df_filtrado = df_resultado.dropna(subset=[parametro_seleccionado])
        if not df_filtrado.empty:
            latitudes = df_filtrado["Latitud"].values
            longitudes = df_filtrado["Longitud"].values
            valores = df_filtrado[parametro_seleccionado].values

            # Calcular interpolaciones
            resultados_interpolacion = []

            # Interpolación Lineal
            valor_lineal = griddata(
                (longitudes, latitudes),
                valores,
                (longitud_punto, latitud_punto),
                method="linear"
            )
            resultados_interpolacion.append({"Método": "Lineal", "Valor Interpolado": valor_lineal})

            # Interpolación Nearest
            valor_nearest = griddata(
                (longitudes, latitudes),
                valores,
                (longitud_punto, latitud_punto),
                method="nearest"
            )
            resultados_interpolacion.append({"Método": "Nearest", "Valor Interpolado": valor_nearest})

            # Interpolación IDW
            valor_idw = idw_interpolation(longitudes, latitudes, valores, longitud_punto, latitud_punto, power=2)
            resultados_interpolacion.append({"Método": "IDW", "Valor Interpolado": valor_idw})

            # Crear DataFrame con los resultados
            df_resultados = pd.DataFrame(resultados_interpolacion)

            # Mostrar resultados
            st.subheader("Resultados de la interpolación")
            st.dataframe(df_resultados)

            # Descarga de los resultados
            csv_resultados = df_resultados.to_csv(index=False)
            st.download_button(
                label="Descargar resultados como CSV",
                data=csv_resultados,
                file_name=f"resultados_interpolacion_{parametro_seleccionado.strip()}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No hay estaciones válidas para el parámetro seleccionado.")
    else:
        st.error("No hay datos disponibles para realizar la interpolación.")
