import streamlit as st
import pandas as pd
import requests
import io  # Importar StringIO desde io
import os  # Asegurar la importación de os

import requests
import streamlit as st

# Token de acceso personal
#GITHUB_TOKEN = "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"
GITHUB_TOKEN = "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"
# URL base del repositorio
#github_base_url = "https://api.github.com/repos/SArcD/MapasClimaticosIA/contents/"


# URL base del repositorio en GitHub
github_base_url = "https://api.github.com/repos/SArcD/MapasClimaticosIA/contents/"

@st.cache_data
# Función para obtener la lista de archivos en una carpeta desde GitHub
def list_files_from_github(folder_path):
    url = github_base_url + folder_path
    try:
        response = requests.get(url)
        response.raise_for_status()
        files = response.json()
        return [file['path'] for file in files if file['type'] == 'file']
    except Exception as e:
        st.error(f"Error al listar archivos en {folder_path}: {e}")
        return []

@st.cache_data
# Función para leer un archivo CSV desde GitHub
def read_csv_from_github(file_path):
    # Convertir la URL a RAW para obtener el contenido del archivo
    raw_url = file_path.replace("https://github.com", "https://raw.githubusercontent.com").replace("/blob/", "/")
    try:
        response = requests.get(raw_url)
        response.raise_for_status()
        # Usar io.StringIO para leer el contenido como un CSV
        return pd.read_csv(io.StringIO(response.text))
    except Exception as e:
        st.error(f"Error al leer {file_path}: {e}")
        return None

# Listar archivos en las carpetas
colima_files = list_files_from_github("datos_estaciones_colima")
cerca_files = list_files_from_github("datos_estaciones_cerca_colima")

# Mostrar archivos y datos
#st.title("Análisis de Datos Meteorológicos")

# Leer y mostrar archivos de la carpeta Colima
#st.subheader("Datos de estaciones en Colima")
#for file in colima_files:
#    st.write(f"Leyendo archivo: {file}")
#    df = read_csv_from_github(f"https://github.com/SArcD/MapasClimaticosIA/blob/main/{file}")
#    if df is not None:
#        st.dataframe(df)

# Leer y mostrar archivos de la carpeta Cerca
#st.subheader("Datos de estaciones cerca de Colima")
#for file in cerca_files:
#    st.write(f"Leyendo archivo: {file}")
#    df = read_csv_from_github(f"https://github.com/SArcD/MapasClimaticosIA/blob/main/{file}")
#    if df is not None:
#        st.dataframe(df)
# Carpetas en el repositorio
output_dir_colima = "datos_estaciones_colima"
output_dir_cerca = "datos_estaciones_cerca_colima"


@st.cache_data
# Descargar archivos desde enlaces utilizando requests
def download_files_from_links(file_links, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(file_links, "r") as f:
        links = f.readlines()
    for link in links:
        link = link.strip()
        if link:
            # Extraer el ID o el nombre del archivo del enlace
            file_name = link.split("id=")[-1] if "id=" in link else os.path.basename(link)
            output_file = os.path.join(output_dir, file_name)
            if not os.path.exists(output_file):
                try:
                    st.write(f"Descargando {file_name}...")
                    response = requests.get(link, stream=True)
                    response.raise_for_status()  # Levantar excepción para códigos de estado HTTP 4xx/5xx
                    with open(output_file, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    st.success(f"Archivo descargado: {file_name}")
                except Exception as e:
                    st.error(f"Error al descargar {link}: {e}")
            else:
                st.write(f"Archivo ya existe: {file_name}")

# Descargar los archivos
#download_files_from_links(links_colima, output_dir_colima)#
#download_files_from_links(links_cerca, output_dir_cerca)


import numpy as np
import streamlit as st
import requests
import os

#############import requests
import numpy as np
import os
import streamlit as st
import requests

# URL del archivo en Dropbox
dropbox_url = "https://www.dropbox.com/scl/fi/y61orc7bzt2p2d22sxtcu/Colima_ACE2.ace2?rlkey=8asyjm6pjqjo0z02gofpg9l2b&st=eqnn71an&dl=1"
file_path = "Colima_ACE2.ace2"

# Descargar el archivo ACE2 si no existe
if not os.path.exists(file_path):
    st.write("Descargando el archivo ACE2 desde Dropbox...")
    try:
        response = requests.get(dropbox_url, stream=True)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success("Archivo ACE2 descargado correctamente.")
    except Exception as e:
        st.error(f"Error al descargar el archivo ACE2: {e}")
        st.stop()

@st.cache_data
# Diagnóstico de archivo
def diagnosticar_archivo(file_path):
    """
    Verifica el tamaño del archivo y calcula dimensiones potenciales.
    """
    try:
        # Tamaño del archivo en bytes
        file_size_bytes = os.path.getsize(file_path)
        #st.write(f"Tamaño del archivo: {file_size_bytes} bytes")

        # Verificar divisibilidad por el tamaño de float32 (4 bytes)
        if file_size_bytes % 4 != 0:
            st.error("El tamaño del archivo no es divisible por 4. Puede estar corrupto o no ser un archivo válido.")
            return None

        # Calcular número total de elementos
        num_elements = file_size_bytes // 4
        #st.write(f"Número total de elementos (float32): {num_elements}")

        # Buscar dimensiones cuadradas o rectangulares
        possible_dims = []
        for rows in range(1, int(np.sqrt(num_elements)) + 1):
            if num_elements % rows == 0:
                cols = num_elements // rows
                possible_dims.append((rows, cols))

        #st.write(f"Dimensiones posibles: {possible_dims}")
        return possible_dims
    except Exception as e:
        st.error(f"Error al diagnosticar el archivo: {e}")
        return None


@st.cache_data
# Leer y calcular dimensiones automáticamente
def read_ace2(file_path, selected_dims):
    """
    Lee el archivo ACE2 y lo convierte a una matriz según las dimensiones seleccionadas.
    """
    try:
        data = np.fromfile(file_path, dtype=np.float32)
        st.write(f"Archivo leído con {data.size} elementos.")

        # Convertir los datos a la matriz con las dimensiones seleccionadas
        rows, cols = selected_dims
        return data.reshape((rows, cols))
    except Exception as e:
        st.error(f"Error al procesar el archivo ACE2: {e}")
        return None

# Diagnosticar el archivo
dimensiones_posibles = diagnosticar_archivo(file_path)

# Si se encuentran dimensiones válidas
if dimensiones_posibles:
    # Permitir que el usuario seleccione dimensiones
    seleccion = st.selectbox("Seleccione las dimensiones para el archivo:", dimensiones_posibles)
    elevation_data = read_ace2(file_path, seleccion)

    if elevation_data is not None:
        tile_size = elevation_data.shape
        st.success(f"Archivo procesado correctamente con dimensiones: {tile_size}.")
else:
    st.error("No se pudieron determinar dimensiones válidas para el archivo.")

## Usar dimensiones por defecto (6000x6000)
#dim_por_defecto = (5760, 5420)

#elevation_data = read_ace2(file_path, dim_por_defecto)

#if elevation_data is not None:
#    tile_size = elevation_data.shape
#    st.success(f"Archivo procesado correctamente con dimensiones predeterminadas: {tile_size}.")
#else:
#    st.error("No se pudo procesar el archivo ACE2 con dimensiones por defecto.")



#############

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
    'Precipitación(mm)', 'Temperatura Media(ºC)', 
    'Temperatura Máxima(ºC)', 'Temperatura Mínima(ºC)', 'Evaporación(mm)'
]

@st.cache_data
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
#def obtener_elevacion(lat, lon, tile_size, elevation_data):
#    """
#    Obtiene la elevación en kilómetros desde el archivo ACE2 usando latitud y longitud.
#    """
#    # Calcular índices en la matriz ACE2 basados en la latitud y longitud
#    lat_idx = int(max(0, min((30 - lat) * tile_size[0] / 15, tile_size[0] - 1)))  # Ajusta para el rango ACE2
#    lon_idx = int(max(0, min((lon + 105) * tile_size[1] / 15, tile_size[1] - 1)))  # Ajusta para el rango ACE2
#    elevacion = elevation_data[lat_idx, lon_idx] / 1000  # Convertir de metros a kilómetros
#    return max(0, elevacion)  # Evitar valores negativos

@st.cache_data
def obtener_elevacion(lat, lon, tile_size, elevation_data):
    """
    Obtiene la elevación en kilómetros desde el archivo ACE2 usando latitud y longitud.
    """
    try:
        # Validar dimensiones de tile_size con elevation_data
        if elevation_data.shape != tile_size:
            raise ValueError(f"Las dimensiones de elevation_data {elevation_data.shape} no coinciden con tile_size {tile_size}")
        
        # Calcular índices en la matriz ACE2 basados en la latitud y longitud
        lat_idx = int((30 - lat) * tile_size[0] / 15)  # Ajusta para el rango ACE2
        lon_idx = int((lon + 105) * tile_size[1] / 15)  # Ajusta para el rango ACE2

        # Asegurar que los índices están dentro del rango válido
        lat_idx = np.clip(lat_idx, 0, tile_size[0] - 1)
        lon_idx = np.clip(lon_idx, 0, tile_size[1] - 1)

        # Obtener elevación
        elevacion = elevation_data[lat_idx, lon_idx] / 1000  # Convertir de metros a kilómetros

        # Depuración opcional: verificar índices y elevación calculada
        # st.write(f"Lat: {lat}, Lon: {lon}, Indices: ({lat_idx}, {lon_idx}), Elevación: {elevacion} km")

        return max(0, elevacion)  # Evitar valores negativos
    except Exception as e:
        raise RuntimeError(f"Error al calcular elevación para lat={lat}, lon={lon}: {e}")

@st.cache_data
def procesar_datos(ano, mes, claves, output_dirs):
    datos_procesados = []

    for output_dir in output_dirs:
        for clave in claves:
            archivo = os.path.join(output_dir, f"{clave}_df.csv")
            if os.path.exists(archivo):
                df = pd.read_csv(archivo) #aqui se agrego un eliminador de espacios
                df.columns = df.columns.str.strip()
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
df_resultado.columns = df_resultado.columns.str.strip()

# Parámetros para radiación solar
S0 = 1361  # Constante solar (W/m²)
Ta = 0.75  # Transmisión atmosférica promedio
k = 0.12   # Incremento de radiación por km de altitud


@st.cache_data
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


@st.cache_data
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
        temp_media_original = row['Temperatura Media(ºC)'] if 'Temperatura Media(ºC)' in row else np.nan
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
    with open('Colima.json', 'r', encoding='latin-1') as file:
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
            #c
            # Ajustar el título dinámicamente según la selección de mes
            if mes == 0:
                titulo_mes = "Promedio Anual"
            else:
                titulo_mes = f"Mes {mes}"

            # Crear el mapa base con las estaciones
            fig = px.scatter_mapbox(
                df_filtrado,
                lat="Latitud",
                lon="Longitud",
                color=columna_grafico,
                hover_name="Clave",
                hover_data=["Estado", columna_grafico],
                title=f"Mapa de estaciones en Colima y alrededores ({columna_grafico.strip()} para el año {ano}, {titulo_mes})",
                mapbox_style="carto-positron",
                center={"lat": 19.0, "lon": -104.0},  # Ajusta el centro del mapa según sea necesario
                zoom=8,
                width=1000,
                height=600,
                color_continuous_scale=coolwarm_colorscale   # Usar escala coolwarm personalizada
                )

                # Configuración del diseño del gráfico
            fig.update_layout(
                title=f"Mapa de estaciones en Colima y alrededores ({columna_grafico.strip()} para el año {ano}, {titulo_mes})",
                margin=dict(l=0, r=0, t=50, b=0)
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

#st.write("Columnas disponibles:", df_resultado.columns.tolist())
#st.write("Número total de filas:", len(df_resultado))
#st.write("Ejemplo de filas:", df_resultado[[columna_grafico, 'Latitud', 'Longitud']].dropna())


#
#latitudes = df_filtrado["Latitud"].values
#longitudes = df_filtrado["Longitud"].values

#st.write("Número total de estaciones:", len(df_resultado))
#st.write("Valores únicos de coordenadas:", len(np.unique(list(zip(longitudes, latitudes)), axis=0)))
#st.write("Valores NaN en columna seleccionada:", df_resultado[columna_grafico].isna().sum())
#st.write("Latitudes únicas:", np.unique(latitudes))
#st.write("Longitudes únicas:", np.unique(longitudes))
#st.write("Valores válidos:", len(valores))
#st.write("Shape grid_lon:", grid_lon.shape)



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
    with open('Colima.json', 'r', encoding='latin-1') as file:
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

#            # Interpolar los datos
#            metodo_interpolacion = st.selectbox("Selecciona el método de interpolación", ["Linear", "Nearest", "IDW"])
#            if metodo_interpolacion in ["Linear", "Nearest"]:
#                interpolados = griddata(
#                    (longitudes, latitudes),
#                    valores,
#                    (grid_lon, grid_lat),
#                    method=metodo_interpolacion.lower()
#                )
#            elif metodo_interpolacion == "IDW":
#                # Implementación básica de IDW
#                def idw_interpolation(x, y, values, xi, yi):
#                    weights = 1 / np.sqrt((x - xi) ** 2 + (y - yi) ** 2 + 1e-10)
#                    return np.sum(weights * values) / np.sum(weights)

#                interpolados = np.zeros_like(grid_lon)
#                for i in range(grid_lon.shape[0]):
#                    for j in range(grid_lon.shape[1]):
#                        interpolados[i, j] = idw_interpolation(longitudes, latitudes, valores, grid_lon[i, j], grid_lat[i, j])

#bloque inicia
            # Interpolar los datos
            metodo_interpolacion = st.selectbox("Selecciona el método de interpolación", ["Linear", "Nearest", "IDW"])
            #columna = columna_grafico.strip()

            # 1. Filtrar valores válidos y sincronizar coordenadas
            mascara_validos = ~df_filtrado[columna_grafico].isna()
            longitudes = df_filtrado.loc[mascara_validos, "Longitud"].values
            latitudes = df_filtrado.loc[mascara_validos, "Latitud"].values
            valores = df_filtrado.loc[mascara_validos, columna_grafico].values

            # 2. Verificar si hay suficientes puntos para interpolar
            if len(valores) < 4:
                st.warning("No hay suficientes estaciones con valores válidos para realizar la interpolación.")
                st.stop()

            # 3. Interpolación
            if metodo_interpolacion in ["Linear", "Nearest"]:
                try:
                    interpolados = griddata(
                        (longitudes, latitudes),
                        valores,
                        (grid_lon, grid_lat),
                        method=metodo_interpolacion.lower()
                    )
                except Exception as e:
                    st.warning(f"Ocurrió un error con '{metodo_interpolacion}'. Usando 'nearest' como alternativa.")
                    interpolados = griddata(
                        (longitudes, latitudes),
                        valores,
                        (grid_lon, grid_lat),
                        method="nearest"
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
#bloque nuevo termina

            
#            # Crear la figura
#            fig = go.Figure()

#            # Añadir contornos de valores interpolados
#            fig.add_trace(
#                go.Contour(
#                    z=interpolados,
#                    x=grid_lon[0],
#                    y=grid_lat[:, 0],
#                    colorscale=coolwarm_colorscale,
#                    line=dict(color="black", width=1.0),  # Líneas más gruesas
#                    opacity=0.7,
#                    contours=dict(
#                        coloring="fill",  # Las zonas entre curvas tienen color
#                        showlabels=True,  # Mostrar etiquetas en los contornos
#                        labelfont=dict(size=10, color="black")
#                    ),
#                    colorbar=dict(
#                        title=f"{columna_grafico.strip()}",
#                        len=0.8  # Reducir la longitud de la barra de color
#                    ),
#                    name=f"Interpolación ({columna_grafico.strip()})"
#                )
#            )

#            # Añadir puntos de las estaciones
#            fig.add_trace(
#                go.Scatter(
#                    x=longitudes,
#                    y=latitudes,
#                    mode="markers",
 #                   marker=dict(
 #                       size=10,
 #                       color="black",
 #                       opacity=1.0,
 #                       #colorscale=coolwarm_scale,
 #                       showscale=False  # Ocultar barra de colores adicional
 #                   ),
 #                   text=df_filtrado["Clave"],
 #                   hoverinfo="text",
 #                   name="Estaciones"
 #               )
 #           )

 #           # Añadir contornos de los municipios
 #           for feature in colima_geojson["features"]:
 #               geometry = feature["geometry"]
 #               properties = feature["properties"]

 #               if "isla" not in properties.get("name", "").lower():
 #                   if geometry["type"] == "Polygon":
 #                       for coordinates in geometry["coordinates"]:
 #                           x_coords, y_coords = zip(*coordinates)
 #                           fig.add_trace(
 #                               go.Scatter(
  #                                  x=x_coords,
  #                                  y=y_coords,
  #                                  mode="lines",
  #                                  line=dict(color="black", width=2),
   #                                 showlegend=False
   #                             )
    #                        )
    #                elif geometry["type"] == "MultiPolygon":
    #                    for polygon in geometry["coordinates"]:
    #                        for coordinates in polygon:
 #                               x_coords, y_coords = zip(*coordinates)
 #                               fig.add_trace(
 #                                   go.Scatter(
 #                                       x=x_coords,
 #                                       y=y_coords,
 #                                       mode="lines",
 #                                       line=dict(color="black", width=2),
 #                                       showlegend=False
 #                                   )
 #                               )

            # Configuración del diseño
##            fig.update_layout(
###                title=f"Mapa de estaciones y contornos interpolados ({columna_grafico.strip()})",
##                xaxis_title="Longitud",
# #               yaxis_title="Latitud",
##                margin=dict(l=0, r=0, t=50, b=0)
##            )

#            #fig.update_layout(
#            #    xaxis=dict(
#            #        title="Longitud",
#            #        titlefont=dict(size=14, family="Arial"),
#            #        tickfont=dict(size=12, family="Arial"),
#            #        range=[-104.7, -103.3]  # Ajustar los límites iniciales del eje X (Longitud)
#            #    ),
#            #        yaxis=dict(
#            #        title="Latitud",
#            #        titlefont=dict(size=14, family="Arial"),
#            #        tickfont=dict(size=12, family="Arial"),
#            #        range=[18.5, 19.7]  # Ajustar los límites iniciales del eje Y (Latitud)
#            #    ),
#                geo=dict(
##                    center=dict(
##                        lon=-104.0,  # Longitud central
##                        lat=19.3     # Latitud central
##                    ),
##                    projection_scale=1  # Ajustar el zoom inicial
#            #    ),
#            #    margin=dict(l=20, r=20, t=50, b=20) 
#            #)

##            fig.update_layout(
##                xaxis=dict(
##                    title="Longitud",
##                    titlefont=dict(size=14, family="Arial"),
# #                   tickfont=dict(size=12, family="Arial"),
# #                   range=[-104.7, -103.3]
# #               ),
# #               yaxis=dict(
# #                   title="Latitud",
# #                   titlefont=dict(size=14, family="Arial"),
# #                   tickfont=dict(size=12, family="Arial"),
# #                   range=[18.5, 19.7]
# #               ),
# #               margin=dict(l=20, r=20, t=50, b=20)
# #           )


            
##            fig.update_layout(
##                width=1000,  # Ancho del gráfico
##                height=600,  # Altura del gráfico
##                title=f"Mapa de estaciones y contornos interpolados ({columna_grafico.strip()} para el año {ano}, mes {mes})",
##                xaxis_title="Longitud",
##                yaxis_title="Latitud",
##                margin=dict(l=0, r=0, t=50, b=0)  # Márgenes del gráfico
##            )

            # Ajustar el título dinámicamente según la selección de mes
##            if mes == 0:
##                titulo_mes = "Promedio Anual"
##            else:
##                titulo_mes = f"Mes {mes}"

##            # Configuración del título del gráfico
#            fig.update_layout(
##            title=f"Mapa de estaciones y contornos interpolados ({columna_grafico.strip()} para el año {ano}, {titulo_mes})",
##            xaxis_title="Longitud",
##            yaxis_title="Latitud",
##            margin=dict(l=0, r=0, t=50, b=0)
##            )

#            # Ajustar el título dinámicamente según la selección de mes
#            if mes == 0:
#                titulo_mes = "Promedio Anual"
#            else:
#                titulo_mes = f"Mes {mes}"
#
#            # Configuración consolidada del layout del gráfico
#            fig.update_layout(
#                title=dict(
#                    text=f"Mapa de estaciones y contornos interpolados ({columna_grafico.strip()} para el año {ano}, {titulo_mes})",
#                    x=0.5,
#                    xanchor='center',
#                    font=dict(size=18)
#                ),
#                xaxis=dict(
#                    title=dict(
#                        text="Longitud",
#                        font=dict(size=14, family="Arial", color='black')
#                    ),
#                    tickfont=dict(size=12, family="Arial", color='black'),
#                    range=[-104.7, -103.3],
#                    showgrid=False
#                ),
#                yaxis=dict(
#                    title=dict(
#                        text="Latitud",
#                        font=dict(size=14, family="Arial", color='black')
#                    ),
#                    tickfont=dict(size=12, family="Arial", color='black'),
#                    range=[18.5, 19.7],
#                    scaleanchor="x",
#                    showgrid=False
#                ),
#                plot_bgcolor="white",
#                paper_bgcolor="white",
#                width=1000,
#                height=600,
#                margin=dict(l=20, r=20, t=50, b=20),
#                showlegend=True
#            )


            

#            # Mostrar el gráfico
#            st.plotly_chart(fig, use_container_width=True)
#        else:
#            st.warning(f"No hay estaciones con datos válidos en la columna '{columna_grafico}'.")
#    else:
#        st.warning("La columna seleccionada no está disponible en el DataFrame.")
#else:
#    st.write("No hay datos disponibles para mostrar en el mapa.")


            # Crear la figura
            fig = go.Figure()

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
                    #titlefont=dict(size=14, family="Arial"),
                    tickfont=dict(size=12, family="Arial"),
                    range=[-104.7, -103.3]  # Ajustar los límites iniciales del eje X (Longitud)
                ),
                    yaxis=dict(
                    title="Latitud",
                    #titlefont=dict(size=14, family="Arial"),
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


## Parámetros para corrección
#gradiente_temperatura = -6.5  # °C/km
#incremento_radiacion = 0.12   # W/m²/km

## Escala de color personalizada
#coolwarm_scale = [
#    [0.0, 'rgb(59,76,192)'],
#    [0.35, 'rgb(116,173,209)'],
#    [0.5, 'rgb(221,221,221)'],
#    [0.65, 'rgb(244,109,67)'],
#    [1.0, 'rgb(180,4,38)']
#]

## Cargar el archivo GeoJSON
#try:
#    with open('Colima.json', 'r', encoding='latin-1') as file:
#        colima_geojson = json.load(file)
#except Exception as e:
#    st.error(f"No se pudo cargar el archivo GeoJSON: {e}")
#    st.stop()

## Función para obtener elevación interpolada
#def obtener_elevacion_interpolada(grid_lon, grid_lat, elevation_data, tile_size):
#    elevacion = np.zeros_like(grid_lon)
#    for i in range(grid_lon.shape[0]):
#        for j in range(grid_lon.shape[1]):
#            lon, lat = grid_lon[i, j], grid_lat[i, j]
#            lat_idx = int(max(0, min((30 - lat) * tile_size[0] / 15, tile_size[0] - 1)))
#            lon_idx = int(max(0, min((lon + 105) * tile_size[1] / 15, tile_size[1] - 1)))
#            elevacion[i, j] = elevation_data[lat_idx, lon_idx] / 1000  # Convertir a km
#    return elevacion

## Función de interpolación IDW
#def idw_interpolation(x, y, values, xi, yi, power=2):
#    weights = 1 / ((x - xi) ** 2 + (y - yi) ** 2 + 1e-10) ** (power / 2)
#    return np.sum(weights * values) / np.sum(weights)

## Mostrar mapa con corrección de valores interpolados
#if not df_resultado.empty:
#    # Selección del parámetro y método de interpolación
#    columna_grafico = st.selectbox(
#        "Seleccionar el parámetro para graficar",
#        options=columnas_numericas + [
#            "Radiación Solar Promedio (W/m²)",
#            "Radiación Solar Corregida (W/m²)"
#        ]
#    )
#    metodo_interpolacion = st.selectbox(
#        "Seleccionar el método de interpolación",
#        ["Linear", "Nearest", "IDW"]
#    )

#    # Filtrar estaciones válidas
#    df_filtrado = df_resultado.dropna(subset=[columna_grafico])

#    if not df_filtrado.empty:
#        latitudes = df_filtrado["Latitud"].values
#        longitudes = df_filtrado["Longitud"].values
#        valores = df_filtrado[columna_grafico].values

#        # Crear una malla de puntos para la interpolación con márgenes
#        margen_long = 0.08 * (longitudes.max() - longitudes.min())
#        margen_lat = 0.08 * (latitudes.max() - latitudes.min())
#
#        grid_lon, grid_lat = np.meshgrid(
#            np.linspace(longitudes.min() - margen_long, longitudes.max() + margen_long, 100),
#            np.linspace(latitudes.min() - margen_lat, latitudes.max() + margen_lat, 100)
#        )

#        # Realizar la interpolación
#        if metodo_interpolacion in ["Linear", "Nearest"]:
#            interpolados = griddata(
#                (longitudes, latitudes),
#                valores,
#                (grid_lon, grid_lat),
#                method=metodo_interpolacion.lower()
#            )
#        elif metodo_interpolacion == "IDW":
#            interpolados = np.zeros_like(grid_lon)
#            for i in range(grid_lon.shape[0]):
#                for j in range(grid_lon.shape[1]):
#                    interpolados[i, j] = idw_interpolation(longitudes, latitudes, valores, grid_lon[i, j], grid_lat[i, j])

#        # Obtener elevaciones interpoladas
#        elevacion_interpolada = obtener_elevacion_interpolada(grid_lon, grid_lat, elevation_data, tile_size)

#        # Corregir valores interpolados
#        if "Temperatura" in columna_grafico:
#            valores_corregidos = interpolados + (gradiente_temperatura * elevacion_interpolada)
#        elif "Radiación" in columna_grafico:
#            valores_corregidos = interpolados * (1 + 0.0*incremento_radiacion * elevacion_interpolada)
#        else:
#            valores_corregidos = interpolados

#        # Diccionario para las unidades según el parámetro
#        unidades = {
#            "Temperatura Media(ºC)": "ºC",
#            "Temperatura Máxima(ºC)": "ºC",
#            "Temperatura Mínima(ºC)": "ºC",
#            "Precipitación(mm)": "mm",
 #           "Evaporación(mm)": "mm",
 #           "Radiación Solar Promedio (W/m²)": "W/m²",
  #          "Radiación Solar Corregida (W/m²)": "W/m²"
#        }

#        # Crear la figura
#        fig = go.Figure()

#        # Añadir contornos corregidos
#        fig.add_trace(
#            go.Contour(
#                z=valores_corregidos,
#                x=grid_lon[0],
#                y=grid_lat[:, 0],
#                colorscale=coolwarm_colorscale,
#                opacity=0.7,
#                line=dict(color="black", width=1.0),  # Líneas de contorno más gruesas
#                contours=dict(
#                    coloring="fill",
#                    showlabels=True,
#                    labelfont=dict(size=10, color="black")
#                ),
#                colorbar=dict(
#                    title=unidades.get(columna_grafico, ""),  # Solo las unidades
#                    len=0.8,
#                    thickness=20,
#                    x=1.1,
#                    y=0.5
#                ),
#                name=f"Interpolación corregida ({columna_grafico.strip()})"
#            )
#        )


#        # Añadir puntos de las estaciones
#        fig.add_trace(
#            go.Scatter(
#                x=longitudes,
 #               y=latitudes,
#                mode="markers",
#                marker=dict(
#                    size=10,
#                    color="black"
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

#        # Configuración del diseño
#        #fig.update_layout(
#        #    title=f"Mapa de estaciones y contornos interpolados ({columna_grafico.strip()} para el año {ano}, mes {mes})",
#        #    xaxis=dict(
#        #        title="Longitud",
#                #titlefont=dict(size=14, family="Arial"),
#        #        tickfont=dict(size=12, family="Arial"),
#        #        range=[-104.7, -103.3]
#        #    ),
#        #    yaxis=dict(
#        #        title="Latitud",
#                #titlefont=dict(size=14, family="Arial"),
#        #        tickfont=dict(size=12, family="Arial"),
#        #        range=[18.5, 19.7],
#        #        scaleanchor="x"
#        #    ),
#        #    width=1000,
#        #    height=600,
#        #    margin=dict(l=20, r=20, t=50, b=20)
#        #)

#        # Configuración del diseño
                        # Configuración consolidada del layout del gráfico
#        fig.update_layout(
#            title=dict(
#                text=f"Mapa de estaciones y contornos interpolados ({columna_grafico.strip()} para el año {ano}, {titulo_mes})",
#                x=0.5,
#                xanchor='center',
#                font=dict(size=18)
#            ),
#            xaxis=dict(
#                title=dict(
#                    text="Longitud",
#                    font=dict(size=14, family="Arial", color='black')
#                ),
#                tickfont=dict(size=12, family="Arial", color='black'),
#                range=[-104.7, -103.3],
#                showgrid=False
#            ),
#            yaxis=dict(
#                title=dict(
#                    text="Latitud",
#                    font=dict(size=14, family="Arial", color='black')
#                ),
#                tickfont=dict(size=12, family="Arial", color='black'),
#                range=[18.5, 19.7],
#                scaleanchor="x",
#                showgrid=False
#            ),
#            plot_bgcolor="white",
#            paper_bgcolor="white",
#            width=1000,
#            height=600,
#            margin=dict(l=20, r=20, t=50, b=20),
#            showlegend=True
#        )

#        # Mostrar gráfico
#        st.plotly_chart(fig, use_container_width=True)
        
##        # Configuración del diseño
##        fig.update_layout(
##            title=f"Mapa de estaciones y contornos interpolados ({columna_grafico.strip()} para el año {ano}, mes {mes})",
##            xaxis=dict(
##                title="Longitud",
##                titlefont=dict(size=14, family="Arial"),
##                tickfont=dict(size=12, family="Arial"),
##                range=[-104.7, -103.3]
##            ),
##            yaxis=dict(
##                title="Latitud",
##                titlefont=dict(size=14, family="Arial"),
##                tickfont=dict(size=12, family="Arial"),
##                range=[18.5, 19.7]
##            ),
##            width=1000,
##            height=600,
##            margin=dict(l=20, r=20, t=50, b=20)
##        )

#                    # Ajustar el título dinámicamente según la selección de mes
##        if mes == 0:
##            titulo_mes = "Promedio Anual"
##        else:
##            titulo_mes = f"Mes {mes}"

##        # Configuración del título del gráfico
##        fig.update_layout(
##        title=f"Mapa de estaciones y contornos interpolados ({columna_grafico.strip()} para el año {ano}, {titulo_mes})",
##        xaxis_title="Longitud",
##        yaxis_title="Latitud",
##        margin=dict(l=0, r=0, t=50, b=0)
##        )

#        # Mostrar gráfico
#        #st.plotly_chart(fig, use_container_width=True)
#    else:
#        st.warning("No hay estaciones válidas para la columna seleccionada.")
#else:
#    st.write("No hay datos disponibles.")


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
    with open('Colima.json', 'r', encoding='latin-1') as file:
        colima_geojson = json.load(file)
except Exception as e:
    st.error(f"No se pudo cargar el archivo GeoJSON: {e}")
    st.stop()

@st.cache_data
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

@st.cache_data
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
            "Temperatura Media(ºC)": "ºC",
            "Temperatura Máxima(ºC)": "ºC",
            "Temperatura Mínima(ºC)": "ºC",
            "Precipitación(mm)": "mm",
            "Evaporación(mm)": "mm",
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
#                titlefont=dict(size=14, family="Arial"),
                tickfont=dict(size=12, family="Arial"),
                range=[-104.7, -103.3]
            ),
            yaxis=dict(
                title="Latitud",
#                titlefont=dict(size=14, family="Arial"),
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

#############################################################################

# Parámetro a graficar
parametro = st.selectbox(
    "Selecciona el parámetro para graficar",
    ['Precipitación(mm)', 'Temperatura Media(ºC)', 'Temperatura Máxima(ºC)', 
     'Temperatura Mínima(ºC)', 'Evaporación(mm)'],
    key="parametro_selectbox"
)

# Calcular la cantidad de registros válidos por estación para el parámetro seleccionado
estaciones_datos = {}
for estacion in claves_colima:
    archivo_estacion = os.path.join(output_dir_colima, f"{estacion}_df.csv")
    if os.path.exists(archivo_estacion):
        try:
            df_estacion = pd.read_csv(archivo_estacion)
            #df_estacion = pd.read_csv(archivo_estacion)
            df_estacion.columns = df_estacion.columns.str.strip()

            df_estacion['Fecha'] = pd.to_datetime(df_estacion['Fecha'], format='%Y/%m/%d', errors='coerce')
            
            # Asegurar que el parámetro es numérico
            if parametro in df_estacion.columns:
                df_estacion[parametro] = pd.to_numeric(
                    df_estacion[parametro].astype(str).str.replace('[^0-9.-]', '', regex=True), errors='coerce'
                )
                # Contar solo registros válidos (no vacíos y no cero)
                registros_validos = df_estacion[parametro][(df_estacion[parametro] > 0)].count()
                estaciones_datos[estacion] = registros_validos
            else:
                estaciones_datos[estacion] = 0
        except Exception as e:
            st.warning(f"Error al procesar la estación {estacion}: {e}")
    else:
        estaciones_datos[estacion] = 0

# Identificar la estación con más registros válidos
estacion_max_datos = max(estaciones_datos, key=estaciones_datos.get)
max_datos = estaciones_datos[estacion_max_datos]

# Definir los rangos de grupos
rango_100_50 = (max_datos * 0.5, max_datos)
rango_50_25 = (max_datos * 0.25, max_datos * 0.5)
rango_menos_25 = (0, max_datos * 0.25)

# Clasificar estaciones en grupos
grupos_estaciones = {
    "100% - 50% de registros válidos": [est for est, datos in estaciones_datos.items() if rango_100_50[0] <= datos <= rango_100_50[1]],
    "50% - 25% de registros válidos": [est for est, datos in estaciones_datos.items() if rango_50_25[0] <= datos < rango_50_25[1]],
    "Menos del 25% de registros válidos": [est for est, datos in estaciones_datos.items() if rango_menos_25[0] <= datos < rango_menos_25[1]],
}

# Mostrar un resumen de los grupos
st.subheader("Resumen de las estaciones por cantidad de datos válidos")
st.write(f"Estación con más registros válidos: {estacion_max_datos} ({max_datos} registros válidos)")

# Menú desplegable para seleccionar el grupo
grupo_seleccionado = st.selectbox("Selecciona el grupo de estaciones según cantidad de datos válidos", list(grupos_estaciones.keys()), key="grupo_selectbox")

# Filtrar estaciones según el grupo seleccionado
estaciones_filtradas = grupos_estaciones[grupo_seleccionado]

# Nuevo menú desplegable con estaciones filtradas
estacion = st.selectbox("Selecciona una estación meteorológica", estaciones_filtradas, key="estacion_filtrada_selectbox")

# Ruta del archivo de la estación seleccionada
archivo_estacion = os.path.join(output_dir_colima, f"{estacion}_df.csv")

# Leer el archivo CSV de la estación seleccionada
try:
    df_estacion = pd.read_csv(archivo_estacion)
    #df_estacion = pd.read_csv(archivo_estacion)
    df_estacion.columns = df_estacion.columns.str.strip()

    df_estacion['Fecha'] = pd.to_datetime(df_estacion['Fecha'], format='%Y/%m/%d', errors='coerce')
    df_estacion['Año'] = df_estacion['Fecha'].dt.year
    df_estacion['Mes'] = df_estacion['Fecha'].dt.month

    # Asegurar que el parámetro es numérico
    if parametro in df_estacion.columns:
        df_estacion[parametro] = pd.to_numeric(
            df_estacion[parametro].astype(str).str.replace('[^0-9.-]', '', regex=True), errors='coerce'
        )

    # Opciones de análisis: anual o mensual
    analisis = st.radio("Selecciona el tipo de análisis", ["Anual", "Mensual"], key="analisis_radio")

    if analisis == "Anual":
        # Calcular promedios anuales y asegurarse de incluir años con 0 registros
        all_years = pd.DataFrame({'Año': range(df_estacion['Año'].min(), df_estacion['Año'].max() + 1)})
        promedios = df_estacion.groupby('Año')[parametro].mean().reset_index()
        promedios.columns = ['Año', f"Promedio de {parametro.strip()}"]

        # Combinar con todos los años para incluir años sin datos
        promedios = all_years.merge(promedios, on='Año', how='left')
        promedios[f"Promedio de {parametro.strip()}"] = promedios[f"Promedio de {parametro.strip()}"].fillna(0)

        #    Gráfico de barras con espacios para años sin datos
        st.subheader(f"Promedios anuales de {parametro.strip()} en la estación {estacion} (Coordenadas: {df_estacion['Latitud'].iloc[0]}, {df_estacion['Longitud'].iloc[0]})")
        fig = go.Figure()

        # Determinar cuartiles para la escala de colores
        valores_validos = promedios[f"Promedio de {parametro.strip()}"][promedios[f"Promedio de {parametro.strip()}"] > 0]
        q1, q2, q3 = valores_validos.quantile([0.25, 0.5, 0.75]).values if not valores_validos.empty else (0, 0, 0)

        # Asignar colores según cuartiles
        for _, row in promedios.iterrows():
            color = "rgb(49,130,189)"  # Azul para Q1
            if row[f"Promedio de {parametro.strip()}"] > q3:
                color = "rgb(214,39,40)"  # Rojo para Q4
            elif row[f"Promedio de {parametro.strip()}"] > q2:
                color = "rgb(255,127,14)"  # Naranja para Q3
            elif row[f"Promedio de {parametro.strip()}"] > q1:
                color = "rgb(255,215,0)"  # Amarillo para Q2

            fig.add_trace(go.Bar(
                x=[row['Año']],
                y=[row[f"Promedio de {parametro.strip()}"]],
                marker_color=color,
                name=f"Año {int(row['Año'])}"
            ))

        # Configuración del gráfico
        fig.update_layout(
            title=f"Promedios anuales de {parametro.strip()} en la estación {estacion} (Coordenadas: {df_estacion['Latitud'].iloc[0]}, {df_estacion['Longitud'].iloc[0]})",
            xaxis_title="Año",
            yaxis_title=f"Promedio de {parametro.strip()}",
            showlegend=False
        )

        st.plotly_chart(fig)

    else:
        # Seleccionar año para análisis mensual
        ano_seleccionado = st.selectbox(
            "Selecciona el año",
            df_estacion['Año'].unique(),
            key="ano_seleccionado_selectbox"
        )

        # Filtrar por año seleccionado y calcular promedios mensuales
        df_anual = df_estacion[df_estacion['Año'] == ano_seleccionado]
        all_months = pd.Series(range(1, 13))
        promedios = df_anual.groupby('Mes')[parametro].mean().reindex(all_months, fill_value=0).reset_index()
        promedios.columns = ['Mes', f"Promedio de {parametro.strip()}"]

        # Gráfico de barras
        st.subheader(f"Promedios mensuales de {parametro.strip()} en {ano_seleccionado} para la estación {estacion}")
        st.bar_chart(promedios.set_index('Mes'))

except FileNotFoundError:
    st.error(f"No se encontró el archivo para la estación seleccionada: {estacion}")
except Exception as e:
    st.error(f"Error al procesar el archivo de la estación {estacion}: {e}")

columna_grafico = parametro
import plotly.express as px
import plotly.graph_objects as go

import plotly.graph_objects as go

# Crear el agrupamiento de estaciones según la cantidad de registros por estación
agrupamiento_estaciones = {
    "50-100%": [],
    "25-50%": [],
    "0-25%": []
}

# Calcular el máximo de registros para una estación
if not df_resultado.empty:
    max_registros = df_resultado['Clave'].value_counts().max()

    # Crear los grupos
    for clave, count in df_resultado['Clave'].value_counts().items():
        if count >= 0.5 * max_registros:
            agrupamiento_estaciones["50-100%"].append(clave)
        elif 0.25 * max_registros <= count < 0.5 * max_registros:
            agrupamiento_estaciones["25-50%"].append(clave)
        else:
            agrupamiento_estaciones["0-25%"].append(clave)

# Identificar el grupo de la estación seleccionada
grupo_seleccionado = None
for grupo, estaciones in agrupamiento_estaciones.items():
    if estacion in estaciones:
        grupo_seleccionado = grupo
        break

# Verificar si el grupo seleccionado existe y filtrar las estaciones
if grupo_seleccionado:
    estaciones_del_grupo = agrupamiento_estaciones[grupo_seleccionado]
    df_filtrado = df_resultado[df_resultado['Clave'].isin(estaciones_del_grupo)].dropna(subset=["Latitud", "Longitud"])
else:
    df_filtrado = pd.DataFrame()  # Si no se encuentra grupo, dejar vacío

if not df_filtrado.empty:
    # Crear una figura base con fondo blanco
    fig = go.Figure()

    # Añadir las estaciones como puntos
    fig.add_trace(
        go.Scatter(
            x=df_filtrado['Longitud'],
            y=df_filtrado['Latitud'],
            mode='markers',
            marker=dict(size=8, color='blue', symbol='circle'),
            text=df_filtrado['Clave'],
            hoverinfo='text',
            name="Estaciones"
        )
    )

    # Añadir la estación seleccionada con un símbolo destacado
    estacion_seleccionada = df_filtrado[df_filtrado['Clave'] == estacion]
    if not estacion_seleccionada.empty:
        fig.add_trace(
            go.Scatter(
                x=estacion_seleccionada['Longitud'],
                y=estacion_seleccionada['Latitud'],
                mode='markers',
                marker=dict(size=14, color='gold', symbol='star'),
                hoverinfo='none',  # Eliminar texto al pasar el cursor
                name="Estación seleccionada"
            )
        )

    # Añadir los polígonos de los municipios desde GeoJSON
    for feature in colima_geojson["features"]:
        geometry = feature["geometry"]
        properties = feature["properties"]

        # Excluir islas si es necesario
        if "isla" not in properties.get("name", "").lower():
            if geometry["type"] == "Polygon":
                for coordinates in geometry["coordinates"]:
                    x_coords, y_coords = zip(*coordinates)
                    fig.add_trace(
                        go.Scatter(
                            x=x_coords,
                            y=y_coords,
                            mode="lines",
                            line=dict(color="black", width=1.5),
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
                                line=dict(color="black", width=1.5),
                                showlegend=False
                            )
                        )

    # Configurar el diseño del gráfico
    fig.update_layout(
        title="Mapa de Estaciones en Colima (Grupo Actual)",
        xaxis_title="Longitud",
        yaxis_title="Latitud",
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1,
            showgrid=False,
            title_font=dict(color='blue'),
            tickfont=dict(color='blue')  # Ticks azules en el eje X
        ),
        yaxis=dict(
            showgrid=False,
            title_font=dict(color='blue'),
            tickfont=dict(color='blue')  # Ticks azules en el eje Y
        ),
        plot_bgcolor="white",  # Fondo blanco
        paper_bgcolor="white",  # Fondo blanco fuera del área de trazado
        legend=dict(
        font=dict(color='blue')),
        showlegend=True,
        width=800,
        height=600
    )

    # Centrar la vista inicial en la capital de Colima
    fig.update_xaxes(range=[-104.5, -103.5])  # Ajustar según las coordenadas de Colima
    fig.update_yaxes(range=[18.5, 19.5])  # Ajustar según las coordenadas de Colima

    # Mostrar el gráfico
    st.plotly_chart(fig)

##########################################

import os
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import streamlit as st

@st.cache_data
def recolectar_coordenadas_nombres(claves, output_dirs):
    """
    Recolecta los nombres y coordenadas geográficas de las estaciones.

    Args:
        claves (list): Lista de claves de las estaciones.
        output_dirs (list): Lista de directorios donde se encuentran los archivos CSV.

    Returns:
        dict: Diccionario con coordenadas (latitud, longitud) como claves y nombres de estaciones como valores.
    """
    coordenadas_nombres = {}

    for output_dir in output_dirs:
        for clave in claves:
            archivo = os.path.join(output_dir, f"{clave}_df.csv")
            if os.path.exists(archivo):
                try:
                    df = pd.read_csv(archivo)
                    #df = pd.read_csv(archivo)
                    df.columns = df.columns.str.strip()


                    # Verificar si las columnas de coordenadas están presentes
                    if 'Latitud' in df.columns and 'Longitud' in df.columns:
                        latitud = round(df['Latitud'].iloc[0], 6)
                        longitud = round(df['Longitud'].iloc[0], 6)
                        coordenadas_nombres[(latitud, longitud)] = clave
                except Exception as e:
                    st.warning(f"Error al procesar la estación {clave} para recolección de coordenadas: {e}")

    return coordenadas_nombres

#@st.cache_data
#def consolidar_datos_estaciones(claves, output_dirs, elevation_data, tile_size):
#    """
#    Consolida los datos de todas las estaciones en un solo DataFrame.

#    Args:
#        claves (list): Lista de claves de estaciones.
#        output_dirs (list): Lista de directorios donde se encuentran los archivos CSV.
#        elevation_data (array): Datos de elevación para corrección.
#        tile_size (tuple): Tamaño de la grilla de elevación.

#    Returns:
#        pd.DataFrame: DataFrame consolidado.
#    """
#    datos_consolidados = []

#    for output_dir in output_dirs:
#        for clave in claves:
#            archivo = os.path.join(output_dir, f"{clave}_df.csv")
#            if os.path.exists(archivo):
#                try:
#                    # Leer datos de la estación
#                    df = pd.read_csv(archivo)
#                    #df = pd.read_csv(archivo)
#                    df.columns = df.columns.str.strip()

#                    df['Fecha'] = pd.to_datetime(df['Fecha'], format='%Y/%m/%d', errors='coerce')
#                    df['Año'] = df['Fecha'].dt.year
#                    df['Mes'] = df['Fecha'].dt.month

#                    # Limpiar columnas numéricas
#                    for col in df.columns:
#                        if col not in ['Fecha', 'Latitud', 'Longitud']:
#                            df[col] = pd.to_numeric(
#                                df[col].astype(str).str.replace('[^0-9.-]', '', regex=True),
#                                errors='coerce'
#                            )

#                    # Calcular promedios anuales y mensuales
#                    promedios = df.groupby(['Año', 'Mes']).mean().reset_index()

#                    # Agregar información geoespacial
#                    if 'Latitud' in df.columns and 'Longitud' in df.columns:
#                        latitud = round(df['Latitud'].iloc[0], 6)
#                        longitud = round(df['Longitud'].iloc[0], 6)
#                        elevacion = obtener_elevacion(latitud, longitud, tile_size, elevation_data)
#                    else:
#                        latitud = np.nan
#                        longitud = np.nan
#                        elevacion = np.nan

#                    # Crear registro consolidado
#                    for _, row in promedios.iterrows():
#                        registro = {
#                            'Clave': clave,
#                            'Latitud': latitud,
#                            'Longitud': longitud,
#                            'Elevación (km)': elevacion,
#                            'Año': row['Año'],
#                            'Mes': row['Mes']
#                        }
#                        registro.update(row.drop(['Año', 'Mes']).to_dict())
#                        datos_consolidados.append(registro)
#                except Exception as e:
#                    st.warning(f"Error al procesar la estación {clave}: {e}")

#    return pd.DataFrame(datos_consolidados)

def consolidar_datos_estaciones(claves, output_dirs, elevation_data, tile_size):
    """
    Consolida los datos de todas las estaciones en un solo DataFrame,
    conservando claves como texto y ordenando por Clave, Año y Mes.
    """
    datos_consolidados = []

    for output_dir in output_dirs:
        for clave in claves:
            archivo = os.path.join(output_dir, f"{clave}_df.csv")
            if os.path.exists(archivo):
                try:
                    # Leer CSV forzando la clave como texto
                    df = pd.read_csv(archivo, dtype={'Clave': str})
                    df.columns = df.columns.str.strip()

                    # Forzar asignación de clave (por si no viene en CSV)
                    df['Clave'] = str(clave)

                    # Convertir fechas y extraer año y mes
                    df['Fecha'] = pd.to_datetime(df['Fecha'], format='%Y/%m/%d', errors='coerce')
                    df['Año'] = df['Fecha'].dt.year
                    df['Mes'] = df['Fecha'].dt.month

                    # Limpiar columnas numéricas
                    for col in df.columns:
                        if col not in ['Clave', 'Fecha', 'Latitud', 'Longitud']:
                            df[col] = pd.to_numeric(
                                df[col].astype(str).str.replace('[^0-9.-]', '', regex=True),
                                errors='coerce'
                            )

                    # Evitar que la clave entre al cálculo del promedio
                    df_prom = df.drop(columns=['Clave', 'Fecha'], errors='ignore')

                    # Calcular promedios por año y mes
                    promedios = df_prom.groupby(['Año', 'Mes']).mean().reset_index()

                    # Obtener coordenadas y elevación
                    if 'Latitud' in df.columns and 'Longitud' in df.columns:
                        latitud = round(df['Latitud'].iloc[0], 6)
                        longitud = round(df['Longitud'].iloc[0], 6)
                        elevacion = obtener_elevacion(latitud, longitud, tile_size, elevation_data)
                    else:
                        latitud, longitud, elevacion = np.nan, np.nan, np.nan

                    # Generar registros
                    for _, row in promedios.iterrows():
                        fecha_str = f"{int(row['Año'])}-{int(row['Mes']):02d}-15"
                        registro = {
                            'Clave': str(clave),
                            'Latitud': latitud,
                            'Longitud': longitud,
                            'Elevación (km)': elevacion,
                            'Año': int(row['Año']),
                            'Mes': int(row['Mes']),
                            'Fecha': pd.to_datetime(fecha_str)
                        }
                        # Agregar valores promedio del resto de columnas
                        registro.update(row.drop(['Año', 'Mes']).to_dict())
                        datos_consolidados.append(registro)

                except Exception as e:
                    st.warning(f"Error al procesar la estación {clave}: {e}")

    # Convertir a DataFrame y ordenar por Clave, Año, Mes
    df_consolidado = pd.DataFrame(datos_consolidados)
    if not df_consolidado.empty:
        df_consolidado = df_consolidado.sort_values(by=['Clave', 'Año', 'Mes'])

    return df_consolidado



#@st.cache_data
def imputar_geoespacial(fila, columnas_imputar, df, tree, k=3):
    """
    Imputa valores faltantes usando una media ponderada inversa de estaciones cercanas.

    Args:
        fila (pd.Series): Fila actual del DataFrame.
        columnas_imputar (list): Columnas a imputar.
        df (pd.DataFrame): DataFrame completo.
        tree (cKDTree): Árbol KD para buscar vecinos.
        k (int): Número de vecinos a considerar.

    Returns:
        pd.Series: Fila con valores imputados.
    """
    # Si no hay coordenadas, no se puede imputar
    if pd.isnull(fila['Latitud']) or pd.isnull(fila['Longitud']):
        return fila

    coord = [fila['Latitud'], fila['Longitud']]
    distancias, indices = tree.query([coord], k=k)
    estaciones_cercanas = df.iloc[indices[0]]

    for columna in columnas_imputar:
        if pd.isnull(fila[columna]):
            valores_cercanos = estaciones_cercanas[columna].dropna()
            if not valores_cercanos.empty:
                pesos = 1 / (distancias[0][:len(valores_cercanos)] + 1e-5)
                fila[columna] = np.average(valores_cercanos, weights=pesos)
    return fila

# Ejecución del flujo
try:
    # Recolectar coordenadas
    coordenadas_estaciones = recolectar_coordenadas_nombres(claves, output_dirs)

    # Consolidar datos
    df_consolidado = consolidar_datos_estaciones(claves, output_dirs, elevation_data, tile_size)

    # Ordenar por Clave, Año y Mes
    df_consolidado = df_consolidado.sort_values(by=['Clave', 'Año', 'Mes'])

    # Interpolación temporal
    df_consolidado_interpolado = df_consolidado.groupby('Clave').apply(
        lambda group: group.interpolate(method='linear', limit_direction='forward', axis=0)
    )

    # Crear árbol para imputación geoespacial
    coordenadas = df_consolidado[['Latitud', 'Longitud']].dropna().values
    tree = cKDTree(coordenadas)

    # Columnas a imputar
    columnas_imputar = ['Temperatura Media(ºC)', 'Temperatura Máxima(ºC)',
                        'Temperatura Mínima(ºC)', 'Precipitación(mm)']

    # Imputación geoespacial
    df_consolidado_imputado = df_consolidado_interpolado.apply(
        lambda row: imputar_geoespacial(row, columnas_imputar, df_consolidado_interpolado, tree),
        axis=1
    )

    df_consolidado_imputado
    
    # Mostrar resultados
    st.subheader("Valores Faltantes Después de la Imputación Geoespacial")
    faltantes_finales = df_consolidado_imputado.isnull().sum()
    st.write(faltantes_finales)

    # Guardar el DataFrame procesado
    df_consolidado_imputado.to_csv("df_consolidado_procesado.csv", index=False)
    st.success("Archivo consolidado con datos imputados guardado correctamente.")

    #from prophet import Prophet

    ## Seleccionar los datos para una estación y variable
    #df_estacion = df_consolidado_imputado[df_consolidado_imputado['Clave'] == 'C06001']
    #df_estacion = df_estacion[['Fecha', 'Temperatura Media(ºC)']].rename(columns={'Fecha': 'ds', 'Temperatura Media(ºC)': 'y'})

    ## Crear y entrenar el modelo
    #modelo = Prophet()
    #modelo.fit(df_estacion)

    ## Generar predicciones futuras
    #futuro = modelo.make_future_dataframe(periods=365)  # Predicciones para 1 año
    #predicciones = modelo.predict(futuro)

    ## Visualizar resultados
    #modelo.plot(predicciones)


#    from prophet import Prophet
#    import matplotlib.pyplot as plt
#    import seaborn as sns
#    # Obtener lista de estaciones con datos
#    estaciones_disponibles = df_consolidado_imputado['Clave'].unique().tolist()
#    variables_disponibles = ['Temperatura Media(ºC)', 'Temperatura Máxima(ºC)', 'Temperatura Mínima(ºC)', 'Precipitación(mm)', 'Evaporación(mm)']

#    # Menús interactivos
#    st.subheader("Predicción con Prophet")
#    estacion_seleccionada = st.selectbox("Selecciona una estación", estaciones_disponibles)
#    variable_seleccionada = st.selectbox("Selecciona una variable climática", variables_disponibles)

#    # Filtrar la estación
#    df_estacion = df_consolidado_imputado[df_consolidado_imputado['Clave'] == estacion_seleccionada]

#    # Verificar columna 'Fecha' y variable seleccionada
#    if 'Fecha' not in df_estacion.columns or variable_seleccionada not in df_estacion.columns:
#        st.warning("No se encuentra la columna 'Fecha' o la variable seleccionada en los datos.")
#    else:
#        # Preparar datos para Prophet
#        df_estacion = df_estacion[['Fecha', variable_seleccionada]].rename(columns={'Fecha': 'ds', variable_seleccionada: 'y'})

#        # Eliminar valores faltantes y fechas duplicadas
#        df_estacion = df_estacion.dropna(subset=['ds', 'y'])
#        df_estacion = df_estacion.drop_duplicates(subset=['ds'])

#        if df_estacion['y'].notna().sum() < 2:
#            st.warning("No hay suficientes datos válidos para entrenar el modelo Prophet.")
#        else:
#            with st.spinner("Entrenando modelo Prophet..."):
#                try:
#                    modelo = Prophet()
#                    modelo.fit(df_estacion)#
#
#                    #modelo = Prophet()
#                    #modelo.fit(df_estacion)

#                    # Generar predicción para 365 días
#                    futuro = modelo.make_future_dataframe(periods=365)
#                    predicciones = modelo.predict(futuro)

#                    # Mostrar predicción completa
#                    fig_pred = modelo.plot(predicciones)
#                    st.subheader(f"Predicción para {variable_seleccionada} ({estacion_seleccionada})")
#                    st.pyplot(fig_pred)

#                    # Mostrar componentes individuales
#                    fig_componentes = modelo.plot_components(predicciones)
#                    st.subheader("Componentes de la predicción")
#                    st.pyplot(fig_componentes)

#                    # Resumen por década
#                    df_estacion['Década'] = (df_estacion['ds'].dt.year // 10) * 10
#                    resumen_decadas = df_estacion.groupby('Década')['y'].mean().reset_index()
#                    resumen_decadas.columns = ['Década', f'Promedio de {variable_seleccionada} (°C)']

#                    # Gráfico de resumen por década
#                    st.subheader("Resumen por Década")
#                    fig_bar, ax = plt.subplots(figsize=(10, 5))
#                    sns.barplot(data=resumen_decadas, x='Década', y=f'Promedio de {variable_seleccionada} (°C)', palette='coolwarm', ax=ax)
#                    ax.set_title('Temperatura Media Promedio por Década')
#                    ax.set_ylabel('°C')
#                    ax.grid(axis='y')

#                    st.pyplot(fig_bar)




    from prophet import Prophet
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Crear el agrupamiento de estaciones según la cantidad de registros por estación
    agrupamiento_estaciones = {
        "50-100%": [],
        "25-50%": [],
        "0-25%": []
    }

    # Calcular el máximo de registros por estación
    if not df_consolidado_imputado.empty:
        max_registros = df_consolidado_imputado['Clave'].value_counts().max()

        # Agrupar estaciones según proporción de datos
        for clave, count in df_consolidado_imputado['Clave'].value_counts().items():
            if count >= 0.5 * max_registros:
                agrupamiento_estaciones["50-100%"].append(clave)
            elif 0.25 * max_registros <= count < 0.5 * max_registros:
                agrupamiento_estaciones["25-50%"].append(clave)
            else:
                agrupamiento_estaciones["0-25%"].append(clave)

        # Menú para seleccionar grupo
        grupo_filtrado = st.selectbox("Selecciona el grupo de estaciones con más datos", ["50-100%", "25-50%"])
        estaciones_disponibles = agrupamiento_estaciones[grupo_filtrado]
    else:
        st.warning("No hay datos suficientes para realizar la agrupación.")
        estaciones_disponibles = []

    # Si hay estaciones disponibles, continuar con Prophet
    if estaciones_disponibles:
        variables_disponibles = ['Temperatura Media(ºC)', 'Temperatura Máxima(ºC)', 'Temperatura Mínima(ºC)', 'Precipitación(mm)', 'Evaporación(mm)']

        st.subheader("Predicción con Prophet")
        estacion_seleccionada = st.selectbox("Selecciona una estación", estaciones_disponibles)
        variable_seleccionada = st.selectbox("Selecciona una variable climática", variables_disponibles)

        # Filtrar la estación
        df_estacion = df_consolidado_imputado[df_consolidado_imputado['Clave'] == estacion_seleccionada]

        if 'Fecha' not in df_estacion.columns or variable_seleccionada not in df_estacion.columns:
            st.warning("No se encuentra la columna 'Fecha' o la variable seleccionada en los datos.")
        else:
            # Preparar datos para Prophet
            df_estacion = df_estacion[['Fecha', variable_seleccionada]].rename(columns={'Fecha': 'ds', variable_seleccionada: 'y'})
            df_estacion = df_estacion.dropna(subset=['ds', 'y']).drop_duplicates(subset=['ds'])

            if df_estacion['y'].notna().sum() < 2:
                st.warning("No hay suficientes datos válidos para entrenar el modelo Prophet.")
            else:
                with st.spinner("Entrenando modelo Prophet..."):
                    try:
                        modelo = Prophet()
                        modelo.fit(df_estacion)

                        futuro = modelo.make_future_dataframe(periods=365)
                        predicciones = modelo.predict(futuro)

                        fig_pred = modelo.plot(predicciones)
                        st.subheader(f"Predicción para {variable_seleccionada} ({estacion_seleccionada})")
                        st.pyplot(fig_pred)

                        fig_componentes = modelo.plot_components(predicciones)
                        st.subheader("Componentes de la predicción")
                        st.pyplot(fig_componentes)

                        # Resumen por década
                        df_estacion['Década'] = (df_estacion['ds'].dt.year // 10) * 10
                        resumen_decadas = df_estacion.groupby('Década')['y'].mean().reset_index()
                        resumen_decadas.columns = ['Década', f'Promedio de {variable_seleccionada} (°C)']

                        st.subheader("Resumen por Década")
                        fig_bar, ax = plt.subplots(figsize=(10, 5))
                        sns.barplot(data=resumen_decadas, x='Década', y=f'Promedio de {variable_seleccionada} (°C)', palette='coolwarm', ax=ax)
                        ax.set_title('Promedio por Década')
                        ax.set_ylabel('°C')
                        ax.grid(axis='y')
                        st.pyplot(fig_bar)

                    except Exception as e:
                        st.error(f"Ocurrió un error al entrenar el modelo Prophet: {e}")
    else:
        st.warning("No hay estaciones suficientes en este grupo.")

                
#                except Exception as e:
#                    st.error(f"Ocurrió un error al entrenar el modelo Prophet: {e}")



except Exception as e:
    st.error(f"Error en el flujo de procesamiento: {e}")




    
