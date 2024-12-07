import pandas as pd
import os

# Configuración del directorio y salida
output_dir = "datos_estaciones_cerca_colima"  # Directorio donde se encuentran los archivos procesados
os.makedirs(output_dir, exist_ok=True)  # Crear la carpeta si no existe
output_file = "Colima_cerca_{}_df.csv"  # Archivo de salida

# Lista de claves para Colima
claves_colima = [
    "C14008", "C14018", "MRZJL", "C14019", "C14046", "C14390", "ELCJL", "TMLJL", "C14027", "CHFJL",
    "C14148", "C14112", "C14029", "C14094", "C14043", "C14343", "C14050", "BBAJL", "C14051", "C14315",
    "VIHJL", "C14348", "C14011", "C14042", "C14086", "C14099", "C14336", "C14109", "TRJCM", "C14031",
    "C14368", "C14034", "ECAJL", "C14141", "C14095", "C14052", "NOGJL", "C14142", "C14184", "TAPJL",
    "C14005", "SLTJL", "C14322", "C14311", "C14151", "C14190", "C14024", "CPEJL", "CP4JL", "CP3JL",
    "CP1JL", "C14067", "HIGJL", "RTOJL", "C14387", "C14350", "C14155", "C14022", "C14118", "C14342",
    "ALCJL", "C14395", "IVAJL", "C14197", "C14158", "C14007", "C14079", "C14117", "C14166", "C14170",
    "C14120", "C14352", "C14030", "CGZJL"
]

# Crear un DataFrame general por año
def procesar_datos_por_anio(ano, claves_colima, output_dir):
    datos_anuales = []

    for clave in claves_colima:
        archivo = os.path.join(output_dir, f"{clave}_df.csv")
        if os.path.exists(archivo):
            # Leer los datos
            df = pd.read_csv(archivo)

            # Limpiar datos: convertir '-' y 0 en NaN
            df.replace(['-', 0], pd.NA, inplace=True)

            # Asegurarse de que las columnas relevantes sean numéricas
            columnas_numericas = [' Temperatura Media(ºC)', ' Temperatura Máxima(ºC)', ' Temperatura Mínima(ºC)', ' Evaporación(mm)']
            for col in columnas_numericas:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Filtrar por el año
            df['Fecha'] = pd.to_datetime(df['Fecha'], format='%Y/%m/%d', errors='coerce')
            df['Año'] = df['Fecha'].dt.year
            df['Mes'] = df['Fecha'].dt.month
            df_anual = df[df['Año'] == ano]
            
            if not df_anual.empty:
                # Calcular el promedio mensual
                promedio_mensual = df_anual.groupby('Mes')[' Temperatura Media(ºC)'].mean().reindex(range(1, 13), fill_value=pd.NA)
                promedio_anual = df_anual[' Temperatura Media(ºC)'].mean()

                # Crear un registro con los datos procesados
                estacion_data = {
                    'Estación': clave,
                    **{f'Mes_{mes}': promedio_mensual[mes] for mes in range(1, 13)},
                    'Promedio Anual': promedio_anual,
                    'Latitud': df['Latitud'].iloc[0],
                    'Longitud': df['Longitud'].iloc[0]
                }
                datos_anuales.append(estacion_data)
        else:
            print(f"Advertencia: No se encontró el archivo para la estación {clave}.")

    # Crear un DataFrame con los datos procesados
    df_anual_final = pd.DataFrame(datos_anuales)
    return df_anual_final

# Procesar un año específico
ano = 2021  # Cambia al año que desees procesar
df_resultado = procesar_datos_por_anio(ano, claves_colima, output_dir)

# Guardar el DataFrame en un archivo
df_resultado.to_csv(output_file.format(ano), index=False)
print(f"Archivo guardado como: {output_file.format(ano)}")
