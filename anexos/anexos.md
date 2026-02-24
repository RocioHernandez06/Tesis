# Librerías
import pandas as pd
import numpy as np
import os
import networkx as nx
import osmnx as ox
from math import sqrt
import folium
import matplotlib.pyplot as plt

# ===============================
# PASO 1: CARGA DE DATOS
# ===============================

# Carpeta de trabajo local
carpeta_resultados = "resultados/"
os.makedirs(carpeta_resultados, exist_ok=True)

# Rutas locales de los archivos CSV
pesos_df = pd.read_csv("data/cacahoatan_pesos.csv", encoding="latin-1")
censo_df = pd.read_csv("data/cacahoatan.csv", encoding="latin-1")
inundaciones_df = pd.read_csv("data/Inundaciones_Chiapas.csv", encoding="latin-1")


# ===============================
# PASO 2: DEFINICIÓN DE GRUPOS DE EDAD Y PRODUCTOS
# ===============================

print("👥 DEFINICIÓN DE GRUPOS DE EDAD Y PRODUCTOS")
print("=" * 60)

# Grupos de edad simplificados
grupos_edad = {
    '0-14': {
        'nombre': 'Niños y Adolescentes (0-14 años)',
        'productos': {
            'Leche_Infantil': {'necesidad': 0.5, 'dias': 7, 'costo_unitario': 45, 'unidad': 'litros'},
            'Papilla_Niños': {'necesidad': 2, 'dias': 7, 'costo_unitario': 25, 'unidad': 'raciones'},
            'Pañales': {'necesidad': 4, 'dias': 7, 'costo_unitario': 6, 'unidad': 'unidades'},
        }
    },
    '15-29_M': {
        'nombre': 'Hombres Jóvenes (15-29 años)',
        'productos': {
            'Alimento_Alta_Energia': {'necesidad': 1.5, 'dias': 7, 'costo_unitario': 45, 'unidad': 'raciones'},
        }
    },
    '15-29_F': {
        'nombre': 'Mujeres Jóvenes (15-29 años)',
        'productos': {
            'Alimento_Balanceado': {'necesidad': 1.2, 'dias': 7, 'costo_unitario': 42, 'unidad': 'raciones'},
        }
    },
    '30-59_M': {
        'nombre': 'Hombres Adultos (30-59 años)',
        'productos': {
            'Alimento_Energia': {'necesidad': 1.4, 'dias': 7, 'costo_unitario': 43, 'unidad': 'raciones'},
        }
    },
    '30-59_F': {
        'nombre': 'Mujeres Adultas (30-59 años)',
        'productos': {
            'Alimento_Nutritivo': {'necesidad': 1.1, 'dias': 7, 'costo_unitario': 41, 'unidad': 'raciones'},
        }
    },
    '60+': {
        'nombre': 'Adultos Mayores (60+ años)',
        'productos': {
            'Alimento_Masticacion_Facil': {'necesidad': 0.9, 'dias': 7, 'costo_unitario': 48, 'unidad': 'raciones'},
            'Medicamentos_Basicos': {'necesidad': 0.2, 'dias': 7, 'costo_unitario': 30, 'unidad': 'kits'},
        }
    }
}

# PRODUCTOS BÁSICOS PARA TODOS (los 5 que mencionas)
productos_basicos = {
    'Agua': {'necesidad': 2, 'dias': 7, 'costo_unitario': 15, 'unidad': 'litros'},
    'Alimentos': {'necesidad': 1, 'dias': 7, 'costo_unitario': 120, 'unidad': 'kits'},
    'Medicamentos': {'necesidad': 0.1, 'dias': 1, 'costo_unitario': 85, 'unidad': 'kits'},
    'Ropa': {'necesidad': 1, 'dias': 1, 'costo_unitario': 200, 'unidad': 'kits'},
    'Higiene': {'necesidad': 1, 'dias': 7, 'costo_unitario': 65, 'unidad': 'kits'}
}

print("📦 PRODUCTOS BÁSICOS DEFINIDOS:")
for producto, specs in productos_basicos.items():
    print(f"• {producto}: {specs['necesidad']} {specs['unidad']}/persona/día × {specs['dias']} días")

print("\n👥 GRUPOS DE EDAD DEFINIDOS:")
for grupo, info in grupos_edad.items():
    print(f"• {grupo}: {info['nombre']}")

# ===============================
# PASO 3: LOCALIDADES INUNDADAS
# ===============================

inundaciones_cacahoatan = inundaciones_df[inundaciones_df['Municipio'] == 'CACAHOATAN']

# Filtrar directamente excluyendo cabeceras
localidades_inundables = [loc for loc in inundaciones_cacahoatan['Localidad'].unique()
                         if 'CABECERA' not in str(loc).upper()
                         and 'MUNICIPAL' not in str(loc).upper()]

print(f"📌 Localidades inundadas (sin cabeceras): {localidades_inundables}")
print(f"📊 Total localidades inundadas: {len(localidades_inundables)}")

# ===============================
# PASO 4: CÁLCULO DE POBLACIÓN POR GRUPOS DE EDAD
# ===============================

print("\n" + "=" * 60)
print("CÁLCULO DE POBLACIÓN POR GRUPOS DE EDAD")
print("=" * 60)

# INICIALIZAR VARIABLES
poblacion_localidad = {}
poblacion_por_grupo = {grupo: 0 for grupo in grupos_edad.keys()}
poblacion_total_afectada = 0

# Distribución estimada de población por grupos
distribucion_grupos = {
    '0-14': 0.30,    # 30% niños y adolescentes
    '15-29_M': 0.14, # 14% hombres jóvenes
    '15-29_F': 0.16, # 16% mujeres jóvenes
    '30-59_M': 0.18, # 18% hombres adultos
    '30-59_F': 0.17, # 17% mujeres adultas
    '60+': 0.05      # 5% adultos mayores
}

# Calcular población total afectada
factor_afectacion = 0.3

for loc in localidades_inundables:
    df = censo_df[censo_df['NOM_LOC'] == loc]
    if not df.empty:
        total = pd.to_numeric(df['POBTOT'], errors='coerce').fillna(0).values[0]
        poblacion_afectada = total * factor_afectacion
        poblacion_localidad[loc] = poblacion_afectada
        poblacion_total_afectada += poblacion_afectada

        # Distribuir por grupos de edad
        for grupo, porcentaje in distribucion_grupos.items():
            poblacion_por_grupo[grupo] += poblacion_afectada * porcentaje
        print(f"✅ {loc}: {poblacion_afectada:,.0f} personas afectadas")
    else:
        print(f"⚠️ No se encontró datos para: {loc}")

print(f"\n👥 POBLACIÓN TOTAL AFECTADA: {poblacion_total_afectada:,.0f} personas")
print("\n📊 DISTRIBUCIÓN POR GRUPOS DE EDAD:")
for grupo, poblacion in poblacion_por_grupo.items():
    porcentaje = (poblacion / poblacion_total_afectada * 100) if poblacion_total_afectada > 0 else 0
    print(f"• {grupos_edad[grupo]['nombre']}: {poblacion:,.0f} personas ({porcentaje:.1f}%)")

# ===============================
# PASO 5: CANDIDATAS PARA ALMACENES
# ===============================
todas_localidades = censo_df['NOM_LOC'].tolist()
localidades_no_inundables = [loc for loc in todas_localidades if loc not in localidades_inundables]

candidatas = []
for loc in localidades_no_inundables:
    peso_match = pesos_df[pesos_df['Localidad'] == loc]
    if not peso_match.empty:
        peso = peso_match['Peso_Posicional'].values[0]
        candidatas.append((loc, peso))

candidatas.sort(key=lambda x: x[1], reverse=True)
almacen_1, almacen_2 = candidatas[:2]
print(f"\n📍 ALMACENES SELECCIONADOS: 1. {almacen_1[0]}  2. {almacen_2[0]}")

# ===============================
# PASO 6: NODOS EN EL GRAFO Y ASIGNACIÓN
# ===============================
print("\n" + "=" * 60)
print("CALCULANDO RUTAS Y ASIGNACIONES")
print("=" * 60)

# CORRECCIÓN: Crear mapeo de coordenadas para TODAS las localidades (no solo inundables)
mapeo_coordenadas_completo = {}
for _, row in censo_df.iterrows():
    mapeo_coordenadas_completo[row['NOM_LOC']] = {
        'lat': row['LATITUD'],
        'lon': row['LONGITUD']
    }

asignacion_localidades = {}

try:
    municipio = "Cacahoatán, Chiapas, Mexico"
    G = ox.graph_from_place(municipio, network_type='drive')
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    def nodo_cercano(lat, lon):
        return ox.nearest_nodes(G, lon, lat)

    # CORRECCIÓN: Obtener coordenadas de almacenes desde el mapeo completo
    coords_alm1 = mapeo_coordenadas_completo.get(almacen_1[0])
    coords_alm2 = mapeo_coordenadas_completo.get(almacen_2[0])

    if coords_alm1:
        nodo_almacen_1 = nodo_cercano(coords_alm1['lat'], coords_alm1['lon'])
    else:
        print(f"⚠️ No se encontraron coordenadas para {almacen_1[0]}")
        nodo_almacen_1 = None

    if coords_alm2:
        nodo_almacen_2 = nodo_cercano(coords_alm2['lat'], coords_alm2['lon'])
    else:
        print(f"⚠️ No se encontraron coordenadas para {almacen_2[0]}")
        nodo_almacen_2 = None

    # CORRECCIÓN: Asignación única verificando que ambos nodos existan
    for loc in localidades_inundables:
        if loc in mapeo_coordenadas_completo:
            coords = mapeo_coordenadas_completo[loc]
            nodo_loc = nodo_cercano(coords['lat'], coords['lon'])

            # Solo calcular si tenemos ambos nodos de almacén
            if nodo_almacen_1 and nodo_almacen_2 and nodo_loc:
                try:
                    # Calcular distancias usando tiempo de viaje (más realista)
                    d1 = nx.shortest_path_length(G, nodo_almacen_1, nodo_loc, weight='travel_time')
                    d2 = nx.shortest_path_length(G, nodo_almacen_2, nodo_loc, weight='travel_time')

                    # Asignar al más cercano
                    if d1 <= d2:
                        asignacion_localidades[loc] = 'Almacén 1'
                    else:
                        asignacion_localidades[loc] = 'Almacén 2'

                    print(f"📍 {loc}: {asignacion_localidades[loc]} (D1: {d1:.0f}s, D2: {d2:.0f}s)")

                except nx.NetworkXNoPath:
                    # Si no hay ruta, asignar por defecto
                    asignacion_localidades[loc] = 'Almacén 1' if localidades_inundables.index(loc) % 2 == 0 else 'Almacén 2'
                    print(f"⚠️ {loc}: No hay ruta, asignación por defecto a {asignacion_localidades[loc]}")
            else:
                # Asignación por defecto si falta algún nodo
                asignacion_localidades[loc] = 'Almacén 1' if localidades_inundables.index(loc) % 2 == 0 else 'Almacén 2'
                print(f"⚠️ {loc}: Nodos incompletos, asignación por defecto a {asignacion_localidades[loc]}")

    print(f"\n📌 RESUMEN DE ASIGNACIÓN:")
    print(f"   • Almacén 1: {len([loc for loc, alm in asignacion_localidades.items() if alm == 'Almacén 1'])} localidades")
    print(f"   • Almacén 2: {len([loc for loc, alm in asignacion_localidades.items() if alm == 'Almacén 2'])} localidades")

except Exception as e:
    print(f"⚠️ Error en cálculo de rutas: {e}")
    # CORRECCIÓN: Asignación por defecto más robusta
    asignacion_localidades = {}
    for i, loc in enumerate(localidades_inundables):
        # Distribuir equitativamente entre los dos almacenes
        asignacion_localidades[loc] = 'Almacén 1' if i % 2 == 0 else 'Almacén 2'
    print("📌 Usando asignación alterna por fallo en NetworkX")

# ===============================
# PASO 7: OPTIMIZACIÓN EXPLÍCITA DE LA FUNCIÓN DE COSTO TOTAL
# ===============================

print("\n" + "=" * 60)
print("OPTIMIZACIÓN DE LA FUNCIÓN DE COSTO TOTAL")
print("=" * 60)

from scipy.optimize import minimize
import numpy as np

def funcion_costo_total(Q, demanda_anual, S, H, C, Z, L, demanda_diaria):
    """
    Función objetivo a minimizar:
    Z = Costo_Pedido + Costo_Mantenimiento + Costo_Compra + Costo_Faltante

    Donde:
    - Costo_Pedido = (D/Q) * S
    - Costo_Mantenimiento = (Q/2 + SS) * H * C
    - Costo_Compra = D * C
    - Costo_Faltante = (D/Q) * ES * B (simplificado)
    """
    # Asegurar que Q sea positivo
    Q = max(Q, 1)

    # Costo de pedido
    costo_pedido = (demanda_anual / Q) * S

    # Costo de mantenimiento (con inventario de seguridad)
    sigma = demanda_diaria * 0.20  # Desviación estándar (20% de variabilidad)
    SS = Z * sigma * np.sqrt(L)    # Inventario de seguridad
    costo_mantenimiento = (Q/2 + SS) * H * C

    # Costo de compra
    costo_compra = demanda_anual * C

    # Costo de faltante (simplificado)
    costo_faltante = (demanda_anual / Q) * 0.05 * C * 2  # 5% probabilidad de faltante

    # Función objetivo total
    Z_total = costo_pedido + costo_mantenimiento + costo_compra + costo_faltante

    return Z_total

# Parámetros de inventario
S = 1500  # Costo de pedido
H = 0.2   # Tasa de mantenimiento
Z = 1.65  # Nivel de servicio (95%)
L = 2     # Tiempo de entrega (días)
B = 2.0   # Factor de costo por faltante

# Optimizar para cada producto
resultados_inventario = []
costos_totales_almacen = {'Almacén 1': 0, 'Almacén 2': 0}

for almacen in ['Almacén 1', 'Almacén 2']:
    print(f"\n📦 OPTIMIZANDO {almacen}")
    print("=" * 80)

    # Calcular población por grupo en este almacén
    poblacion_almacen = sum(poblacion_localidad[loc] for loc, alm in asignacion_localidades.items() if alm == almacen)

    for grupo, info_grupo in grupos_edad.items():
        # Población del grupo en este almacén
        poblacion_grupo = poblacion_por_grupo[grupo] * (poblacion_almacen / poblacion_total_afectada)

        print(f"\n👥 {info_grupo['nombre']}: {poblacion_grupo:,.0f} personas")
        print("-" * 50)

        # PRODUCTOS ESPECÍFICOS DEL GRUPO
        for producto, specs in info_grupo['productos'].items():
            # Calcular demanda para 7 días
            demanda = poblacion_grupo * specs['necesidad'] * specs['dias']
            demanda_anual = demanda * 12
            C = specs['costo_unitario']

            if demanda_anual > 0:
                demanda_diaria = demanda / 30

                # VALOR INICIAL (EOQ tradicional)
                Q_eoq = np.sqrt((2 * demanda_anual * S) / (H * C))

                # OPTIMIZACIÓN NUMÉRICA
                try:
                    resultado = minimize(
                        funcion_costo_total,
                        x0=Q_eoq,
                        args=(demanda_anual, S, H, C, Z, L, demanda_diaria),
                        method='L-BFGS-B',
                        bounds=[(1, None)],  # Q debe ser ≥ 1
                        options={'maxiter': 1000}
                    )

                    if resultado.success:
                        Q_optimo = max(1, resultado.x[0])  # Asegurar Q ≥ 1
                        costo_minimo = resultado.fun

                        # Calcular costo con EOQ tradicional para comparación
                        costo_eoq = funcion_costo_total(Q_eoq, demanda_anual, S, H, C, Z, L, demanda_diaria)

                        # Calcular otros parámetros con Q óptimo
                        R_optimo = demanda_diaria * L
                        sigma = demanda_diaria * 0.20
                        SS_optimo = Z * sigma * np.sqrt(L)

                        # CÁLCULO DE COSTOS DESGLOSADOS
                        costo_pedido = (demanda_anual / Q_optimo) * S
                        costo_mantenimiento = (Q_optimo/2 + SS_optimo) * H * C
                        costo_compra = demanda_anual * C

                        ahorro = costo_eoq - costo_minimo
                        porcentaje_ahorro = (ahorro / costo_eoq * 100) if costo_eoq > 0 else 0

                        costos_totales_almacen[almacen] += costo_minimo

                        # Guardar resultados
                        resultados_inventario.append({
                            'Almacen': almacen,
                            'Grupo_Edad': grupo,
                            'Nombre_Grupo': info_grupo['nombre'],
                            'Producto': producto,
                            'Tipo': 'Específico',
                            'Poblacion_Grupo': poblacion_grupo,
                            'Demanda_7dias': demanda,
                            'Demanda_Anual': demanda_anual,
                            'Costo_Unitario': C,
                            'Cantidad_EOQ': Q_eoq,
                            'Cantidad_Optima': Q_optimo,
                            'Punto_Reorden_R': R_optimo,
                            'Inventario_Seguridad_SS': SS_optimo,
                            'Costo_Pedido': costo_pedido,
                            'Costo_Mantenimiento': costo_mantenimiento,
                            'Costo_Compra': costo_compra,
                            'Costo_Total': costo_minimo,
                            'Costo_EOQ': costo_eoq,
                            'Ahorro': ahorro,
                            'Porcentaje_Ahorro': porcentaje_ahorro,
                            'Unidad': specs['unidad'],
                            'Optimizacion_Exitosa': True
                        })

                        print(f"   ✅ {producto:<25} | EOQ: {Q_eoq:6.0f} | Óptimo: {Q_optimo:6.0f} | Ahorro: {porcentaje_ahorro:5.1f}% (${ahorro:>8,.0f})")

                    else:
                        # Si falla la optimización, usar EOQ
                        Q_optimo = Q_eoq
                        costo_minimo = costo_eoq
                        print(f"   ⚠️ {producto:<25} | Usando EOQ (falló optimización)")

                except Exception as e:
                    # En caso de error, usar EOQ
                    Q_optimo = Q_eoq
                    costo_minimo = funcion_costo_total(Q_eoq, demanda_anual, S, H, C, Z, L, demanda_diaria)
                    print(f"   ❌ {producto:<25} | Error: {str(e)[:50]}...")

    # PRODUCTOS BÁSICOS PARA TODOS EN ESTE ALMACÉN
    print(f"\n📦 OPTIMIZANDO PRODUCTOS BÁSICOS (Todos los grupos):")
    print("-" * 50)

    for producto, specs in productos_basicos.items():
        # Calcular demanda para 7 días
        demanda = poblacion_almacen * specs['necesidad'] * specs['dias']
        demanda_anual = demanda * 12
        C = specs['costo_unitario']

        if demanda_anual > 0:
            demanda_diaria = demanda / 30

            # VALOR INICIAL (EOQ tradicional)
            Q_eoq = np.sqrt((2 * demanda_anual * S) / (H * C))

            # OPTIMIZACIÓN NUMÉRICA
            try:
                resultado = minimize(
                    funcion_costo_total,
                    x0=Q_eoq,
                    args=(demanda_anual, S, H, C, Z, L, demanda_diaria),
                    method='L-BFGS-B',
                    bounds=[(1, None)],
                    options={'maxiter': 1000}
                )

                if resultado.success:
                    Q_optimo = max(1, resultado.x[0])
                    costo_minimo = resultado.fun
                    costo_eoq = funcion_costo_total(Q_eoq, demanda_anual, S, H, C, Z, L, demanda_diaria)

                    # Calcular otros parámetros
                    R_optimo = demanda_diaria * L
                    sigma = demanda_diaria * 0.20
                    SS_optimo = Z * sigma * np.sqrt(L)

                    costo_pedido = (demanda_anual / Q_optimo) * S
                    costo_mantenimiento = (Q_optimo/2 + SS_optimo) * H * C
                    costo_compra = demanda_anual * C

                    ahorro = costo_eoq - costo_minimo
                    porcentaje_ahorro = (ahorro / costo_eoq * 100) if costo_eoq > 0 else 0

                    costos_totales_almacen[almacen] += costo_minimo

                    resultados_inventario.append({
                        'Almacen': almacen,
                        'Grupo_Edad': 'Todos',
                        'Nombre_Grupo': 'Todos los grupos',
                        'Producto': producto,
                        'Tipo': 'Básico',
                        'Poblacion_Grupo': poblacion_almacen,
                        'Demanda_7dias': demanda,
                        'Demanda_Anual': demanda_anual,
                        'Costo_Unitario': C,
                        'Cantidad_EOQ': Q_eoq,
                        'Cantidad_Optima': Q_optimo,
                        'Punto_Reorden_R': R_optimo,
                        'Inventario_Seguridad_SS': SS_optimo,
                        'Costo_Pedido': costo_pedido,
                        'Costo_Mantenimiento': costo_mantenimiento,
                        'Costo_Compra': costo_compra,
                        'Costo_Total': costo_minimo,
                        'Costo_EOQ': costo_eoq,
                        'Ahorro': ahorro,
                        'Porcentaje_Ahorro': porcentaje_ahorro,
                        'Unidad': specs['unidad'],
                        'Optimizacion_Exitosa': True
                    })

                    print(f"   ✅ {producto:<25} | EOQ: {Q_eoq:6.0f} | Óptimo: {Q_optimo:6.0f} | Ahorro: {porcentaje_ahorro:5.1f}% (${ahorro:>8,.0f})")

                else:
                    Q_optimo = Q_eoq
                    costo_minimo = costo_eoq
                    print(f"   ⚠️ {producto:<25} | Usando EOQ (falló optimización)")

            except Exception as e:
                Q_optimo = Q_eoq
                costo_minimo = funcion_costo_total(Q_eoq, demanda_anual, S, H, C, Z, L, demanda_diaria)
                print(f"   ❌ {producto:<25} | Error: {str(e)[:50]}...")

# RESUMEN DE OPTIMIZACIÓN
print("\n" + "=" * 60)
print("📊 RESUMEN DE OPTIMIZACIÓN")
print("=" * 60)

if resultados_inventario:
    df_optimizacion = pd.DataFrame(resultados_inventario)

    ahorro_total = df_optimizacion['Ahorro'].sum()
    costo_total_optimizado = df_optimizacion['Costo_Total'].sum()
    costo_total_eoq = df_optimizacion['Costo_EOQ'].sum()

    optimizaciones_exitosas = df_optimizacion['Optimizacion_Exitosa'].sum()
    total_optimizaciones = len(df_optimizacion)

    print(f"✅ Optimizaciones exitosas: {optimizaciones_exitosas}/{total_optimizaciones} ({optimizaciones_exitosas/total_optimizaciones*100:.1f}%)")
    print(f"💰 Costo total con EOQ:     ${costo_total_eoq:,.0f}")
    print(f"💰 Costo total optimizado:  ${costo_total_optimizado:,.0f}")
    print(f"💵 Ahorro total:            ${ahorro_total:,.0f}")
    print(f"📈 Reducción de costos:     {ahorro_total/costo_total_eoq*100:.2f}%")

    # Mostrar top 5 mejores ahorros
    top_ahorros = df_optimizacion.nlargest(5, 'Ahorro')[['Producto', 'Almacen', 'Ahorro', 'Porcentaje_Ahorro']]
    print(f"\n🏆 TOP 5 MEJORES AHORROS:")
    for idx, row in top_ahorros.iterrows():
        print(f"   {row['Producto']:<25} | {row['Almacen']:<12} | Ahorro: ${row['Ahorro']:>8,.0f} ({row['Porcentaje_Ahorro']:.1f}%)")
else:
    print("⚠️ No se generaron resultados de optimización")

# ===============================
# PASO 8: TABLAS ORGANIZADAS
# ===============================

print("\n" + "=" * 60)
print("TABLAS DE RESULTADOS ORGANIZADAS - OPTIMIZACIÓN")
print("=" * 60)

# Crear DataFrame con todos los resultados
df_resultados = pd.DataFrame(resultados_inventario)

if not df_resultados.empty:
    # ===========================================
    # TABLA 1: POBLACIÓN POR LOCALIDAD
    # ===========================================
    print("\n📋 TABLA 1: POBLACIÓN AFECTADA POR LOCALIDAD")
    print("=" * 70)

    tabla_poblacion = []
    for loc in localidades_inundables:
        if loc in poblacion_localidad:
            tabla_poblacion.append({
                'Localidad': loc,
                'Población_Afectada': poblacion_localidad[loc],
                'Almacén_Asignado': asignacion_localidades.get(loc, 'No asignado')
            })

    df_poblacion = pd.DataFrame(tabla_poblacion)
    df_poblacion['Población_Afectada'] = df_poblacion['Población_Afectada'].round(0).astype(int)
    print(df_poblacion.to_string(index=False))

    # ===========================================
    # TABLA 2: DISTRIBUCIÓN POR GRUPOS DE EDAD
    # ===========================================
    print("\n\n📊 TABLA 2: DISTRIBUCIÓN POR GRUPOS DE EDAD")
    print("=" * 70)

    tabla_grupos = []
    for grupo, poblacion in poblacion_por_grupo.items():
        porcentaje = (poblacion / poblacion_total_afectada * 100) if poblacion_total_afectada > 0 else 0
        tabla_grupos.append({
            'Grupo_Edad': grupos_edad[grupo]['nombre'],
            'Población': round(poblacion, 0),
            'Porcentaje': f"{porcentaje:.1f}%"
        })

    df_grupos = pd.DataFrame(tabla_grupos)
    print(df_grupos.to_string(index=False))

    # ===========================================
    # TABLA 3: ASIGNACIÓN DE ALMACENES (MEJORADA)
    # ===========================================
    print("\n\n📍 TABLA 3: ASIGNACIÓN DE LOCALIDADES A ALMACENES")
    print("=" * 70)

    # Obtener listas de localidades por almacén
    localidades_almacen1 = [loc for loc, alm in asignacion_localidades.items() if alm == 'Almacén 1']
    localidades_almacen2 = [loc for loc, alm in asignacion_localidades.items() if alm == 'Almacén 2']

    # Crear tabla
    tabla_asignacion_mejorada = pd.DataFrame({
        'Almacén': ['Almacén 1', 'Almacén 2'],
        'Ubicación': [almacen_1[0], almacen_2[0]],
        'Peso_Posicional': [almacen_1[1], almacen_2[1]],
        'Localidades_Asignadas': [len(localidades_almacen1), len(localidades_almacen2)],
        'Lista_de_Localidades': [', '.join(localidades_almacen1), ', '.join(localidades_almacen2)]
    })

    print(tabla_asignacion_mejorada.to_string(index=False))

    # ===========================================
    # TABLA 4: INVENTARIO POR GRUPO DE EDAD
    # ===========================================
    print("\n\n📦 TABLA 4: INVENTARIO POR GRUPO DE EDAD - OPTIMIZADO")
    print("=" * 70)

    inventario_grupo = df_resultados.groupby(['Nombre_Grupo', 'Almacen']).agg({
        'Costo_Total': 'sum',
        'Demanda_7dias': 'sum',
        'Poblacion_Grupo': 'first',
        'Ahorro': 'sum'  # NUEVA COLUMNA DE OPTIMIZACIÓN
    }).reset_index()

    # Formatear números para mejor presentación
    inventario_grupo['Costo_Total'] = inventario_grupo['Costo_Total'].round(0)
    inventario_grupo['Demanda_7dias'] = inventario_grupo['Demanda_7dias'].round(0)
    inventario_grupo['Poblacion_Grupo'] = inventario_grupo['Poblacion_Grupo'].round(0)
    inventario_grupo['Ahorro'] = inventario_grupo['Ahorro'].round(0)

    print(inventario_grupo.to_string(index=False))

    # ===========================================
    # TABLA 5: INVENTARIO DETALLADO POR PRODUCTO
    # ===========================================
    print("\n\n📋 TABLA 5: INVENTARIO DETALLADO POR PRODUCTO - OPTIMIZADO")
    print("=" * 90)

    inventario_producto = df_resultados.groupby(['Producto', 'Almacen', 'Unidad']).agg({
        'Demanda_7dias': 'sum',
        'Cantidad_EOQ': 'mean',           # CAMBIADO: 'Cantidad_Economica_Q' → 'Cantidad_EOQ'
        'Cantidad_Optima': 'mean',        # NUEVA COLUMNA DE OPTIMIZACIÓN
        'Punto_Reorden_R': 'mean',
        'Inventario_Seguridad_SS': 'mean',
        'Costo_Total': 'sum',
        'Ahorro': 'sum'                   # NUEVA COLUMNA DE OPTIMIZACIÓN
    }).reset_index()

    # Formatear números
    for col in ['Demanda_7dias', 'Cantidad_EOQ', 'Cantidad_Optima', 'Punto_Reorden_R', 'Inventario_Seguridad_SS', 'Costo_Total', 'Ahorro']:
        inventario_producto[col] = inventario_producto[col].round(0)

    print(inventario_producto.to_string(index=False))

    # ===========================================
    # TABLA 6: RESUMEN FINAL DE COSTOS
    # ===========================================
    print("\n\n💰 TABLA 6: RESUMEN FINAL DE COSTOS - OPTIMIZACIÓN")
    print("=" * 60)

    costo_total = df_resultados['Costo_Total'].sum()
    costo_almacen1 = df_resultados[df_resultados['Almacen'] == 'Almacén 1']['Costo_Total'].sum()
    costo_almacen2 = df_resultados[df_resultados['Almacen'] == 'Almacén 2']['Costo_Total'].sum()
    ahorro_total = df_resultados['Ahorro'].sum()
    costo_total_eoq = costo_total + ahorro_total  # Costo si no hubiera optimización

    tabla_costos = pd.DataFrame({
        'Concepto': [
            'Costo Total con EOQ (sin optimizar)',
            'Costo Total Optimizado',
            'Ahorro por Optimización',
            'Costo Almacén 1',
            'Costo Almacén 2',
            'Costo Mensual Promedio'
        ],
        'Monto_MXN': [
            f"${costo_total_eoq:,.0f}",
            f"${costo_total:,.0f}",
            f"${ahorro_total:,.0f} ({ahorro_total/costo_total_eoq*100:.1f}%)",
            f"${costo_almacen1:,.0f}",
            f"${costo_almacen2:,.0f}",
            f"${costo_total/12:,.0f}"
        ]
    })
    print(tabla_costos.to_string(index=False))

    # ===========================================
    # TABLA 7: COMPARACIÓN EOQ vs OPTIMIZACIÓN (NUEVA)
    # ===========================================
    print("\n\n📊 TABLA 7: COMPARACIÓN EOQ vs OPTIMIZACIÓN")
    print("=" * 70)

    comparacion = df_resultados.groupby('Almacen').agg({
        'Cantidad_EOQ': 'mean',
        'Cantidad_Optima': 'mean',
        'Costo_EOQ': 'sum',
        'Costo_Total': 'sum',
        'Ahorro': 'sum',
        'Optimizacion_Exitosa': 'sum',
        'Producto': 'count'
    }).reset_index()

    comparacion['Reduccion_Cantidad'] = ((comparacion['Cantidad_EOQ'] - comparacion['Cantidad_Optima']) / comparacion['Cantidad_EOQ'] * 100)
    comparacion['Reduccion_Costo'] = (comparacion['Ahorro'] / comparacion['Costo_EOQ'] * 100)
    comparacion['Tasa_Exito'] = (comparacion['Optimizacion_Exitosa'] / comparacion['Producto'] * 100)

    # Formatear números
    for col in ['Cantidad_EOQ', 'Cantidad_Optima', 'Costo_EOQ', 'Costo_Total', 'Ahorro']:
        comparacion[col] = comparacion[col].round(0)
    for col in ['Reduccion_Cantidad', 'Reduccion_Costo', 'Tasa_Exito']:
        comparacion[col] = comparacion[col].round(1)

    print(comparacion.to_string(index=False))

    # ===========================================
    # GUARDAR TODAS LAS TABLAS EN CSV
    # ===========================================
    print(f"\n💾 GUARDANDO TABLAS EN {carpeta_resultados}")

    df_poblacion.to_csv(os.path.join(carpeta_resultados, 'tabla_poblacion_localidades.csv'), index=False)
    df_grupos.to_csv(os.path.join(carpeta_resultados, 'tabla_distribucion_grupos.csv'), index=False)
    tabla_asignacion_mejorada.to_csv(os.path.join(carpeta_resultados, 'tabla_asignacion_almacenes.csv'), index=False)
    inventario_grupo.to_csv(os.path.join(carpeta_resultados, 'tabla_inventario_grupos.csv'), index=False)
    inventario_producto.to_csv(os.path.join(carpeta_resultados, 'tabla_inventario_productos.csv'), index=False)
    tabla_costos.to_csv(os.path.join(carpeta_resultados, 'tabla_resumen_costos.csv'), index=False)
    comparacion.to_csv(os.path.join(carpeta_resultados, 'tabla_comparacion_optimizacion.csv'), index=False)

    print("✅ Tablas guardadas:")
    print("   • tabla_poblacion_localidades.csv")
    print("   • tabla_distribucion_grupos.csv")
    print("   • tabla_asignacion_almacenes.csv")
    print("   • tabla_inventario_grupos.csv")
    print("   • tabla_inventario_productos.csv")
    print("   • tabla_resumen_costos.csv")
    print("   • tabla_comparacion_optimizacion.csv (NUEVA)")

else:
    print("⚠️ No se generaron resultados de inventario")
