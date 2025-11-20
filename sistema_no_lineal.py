"""
Sistemas No Lineales 2×2
Proyecto Final - Ecuaciones Diferenciales 1

Este módulo define los sistemas no lineales sin solución analítica:
1. Modelo de Lotka-Volterra (Predador-Presa)
2. Modelo SIR de Epidemias (Simplificado)
"""

import numpy as np
from typing import Tuple, Dict


# MODELO DE LOTKA-VOLTERRA (PREDADOR-PRESA)

def lotka_volterra(parametros: Dict[str, float] = None):
    """
    Modelo de Lotka-Volterra para dinámica predador-presa.
    
    Sistema:
    dx/dt = αx - βxy
    dy/dt = δxy - γy
    
    Variables:
    - x(t): Población de presas
    - y(t): Población de predadores
    
    Parámetros por defecto:
    - α = 1.5: Tasa de crecimiento de presas
    - β = 1.0: Tasa de depredación
    - γ = 3.0: Tasa de mortalidad de predadores
    - δ = 1.0: Eficiencia de conversión de presas en predadores
    
    Condiciones iniciales por defecto:
    - x(0) = 1.0 (presas)
    - y(0) = 1.0 (predadores)
    
    Referencia:
    Murray, J. D. (2002). Mathematical Biology I: An Introduction (3rd ed.). Springer.
    
    Retorna:
    --------
    f : función del sistema
    t0 : tiempo inicial
    X0 : condiciones iniciales
    params : diccionario de parámetros
    info : información del modelo
    """
    # Parámetros por defecto
    if parametros is None:
        parametros = {
            'alpha': 1.5,  # Tasa de crecimiento de presas
            'beta': 1.0,   # Tasa de depredación
            'gamma': 3.0,  # Tasa de mortalidad de predadores
            'delta': 1.0   # Eficiencia de conversión
        }
    
    alpha = parametros['alpha']
    beta = parametros['beta']
    gamma = parametros['gamma']
    delta = parametros['delta']
    
    def f(t, X):
        """
        Sistema de Lotka-Volterra.
        
        Parámetros:
        -----------
        t : float
            Tiempo
        X : np.ndarray
            Vector [x, y] donde x=presas, y=predadores
        
        Retorna:
        --------
        dX : np.ndarray
            Vector [dx/dt, dy/dt]
        """
        x, y = X
        dx_dt = alpha * x - beta * x * y
        dy_dt = delta * x * y - gamma * y
        return np.array([dx_dt, dy_dt])
    
    # Condiciones iniciales
    t0 = 0.0
    X0 = np.array([1.0, 1.0])  # [presas, predadores]
    
    # Información del modelo
    info = {
        'nombre': 'Modelo de Lotka-Volterra (Predador-Presa)',
        'variables': ['x(t): Población de presas', 'y(t): Población de predadores'],
        'ecuaciones': [
            f'dx/dt = {alpha}x - {beta}xy',
            f'dy/dt = {delta}xy - {gamma}y'
        ],
        'referencia': 'Murray, J. D. (2002). Mathematical Biology I (3rd ed.). Springer.',
        'punto_equilibrio_trivial': (0, 0),
        'punto_equilibrio_no_trivial': (gamma/delta, alpha/beta)
    }
    
    return f, t0, X0, parametros, info


def analisis_lotka_volterra(X: np.ndarray, t: np.ndarray, params: Dict[str, float]):
    """
    Análisis del comportamiento del modelo de Lotka-Volterra.
    
    Calcula:
    - Promedio de poblaciones
    - Valores máximos y mínimos
    - Estimación de período de oscilación
    
    Parámetros:
    -----------
    X : np.ndarray
        Matriz de soluciones [tiempo, 2]
    t : np.ndarray
        Vector de tiempos
    params : dict
        Parámetros del modelo
    
    Retorna:
    --------
    analisis : dict
        Diccionario con resultados del análisis
    """
    x = X[:, 0]  # Presas
    y = X[:, 1]  # Predadores
    
    analisis = {
        'presas': {
            'promedio': np.mean(x),
            'maximo': np.max(x),
            'minimo': np.min(x),
            'valor_final': x[-1]
        },
        'predadores': {
            'promedio': np.mean(y),
            'maximo': np.max(y),
            'minimo': np.min(y),
            'valor_final': y[-1]
        },
        'punto_equilibrio': (params['gamma']/params['delta'], 
                            params['alpha']/params['beta'])
    }
    
    # Estimar período aproximado (si hay oscilaciones)
    if len(x) > 10:
        # Buscar picos en población de presas
        from scipy.signal import find_peaks
        try:
            picos, _ = find_peaks(x, height=np.mean(x))
            if len(picos) > 1:
                periodos = np.diff(t[picos])
                analisis['periodo_estimado'] = np.mean(periodos)
        except:
            analisis['periodo_estimado'] = None
    
    return analisis


#MODELO SIR DE EPIDEMIAS (SIMPLIFICADO)

def modelo_sir(parametros: Dict[str, float] = None):
    """
    Modelo SIR simplificado para dinámica de epidemias.
    
    Sistema:
    dS/dt = -βSI
    dI/dt = βSI - γI
    
    Variables:
    - S(t): Proporción de susceptibles
    - I(t): Proporción de infectados
    - R(t) = 1 - S(t) - I(t): Proporción de recuperados (implícita)
    
    Parámetros por defecto:
    - β = 0.5: Tasa de transmisión
    - γ = 0.1: Tasa de recuperación
    - R₀ = β/γ = 5.0: Número básico de reproducción
    
    Condiciones iniciales por defecto:
    - S(0) = 0.99 (99% susceptibles)
    - I(0) = 0.01 (1% infectados)
    
    Referencia:
    Kermack, W. O., & McKendrick, A. G. (1927). A contribution to the 
    mathematical theory of epidemics. Proceedings of the Royal Society of 
    London. Series A, 115(772), 700-721.
    
    Retorna:
    --------
    f : función del sistema
    t0 : tiempo inicial
    X0 : condiciones iniciales
    params : diccionario de parámetros
    info : información del modelo
    """
    # Parámetros por defecto
    if parametros is None:
        parametros = {
            'beta': 0.5,   # Tasa de transmisión
            'gamma': 0.1   # Tasa de recuperación
        }
    
    beta = parametros['beta']
    gamma = parametros['gamma']
    R0 = beta / gamma  # Número básico de reproducción
    
    def f(t, X):
        """
        Sistema SIR.
        
        Parámetros:
        -----------
        t : float
            Tiempo
        X : np.ndarray
            Vector [S, I] donde S=susceptibles, I=infectados
        
        Retorna:
        --------
        dX : np.ndarray
            Vector [dS/dt, dI/dt]
        """
        S, I = X
        dS_dt = -beta * S * I
        dI_dt = beta * S * I - gamma * I
        return np.array([dS_dt, dI_dt])
    
    # Condiciones iniciales
    t0 = 0.0
    X0 = np.array([0.99, 0.01])  # [susceptibles, infectados]
    
    # Información del modelo
    info = {
        'nombre': 'Modelo SIR de Epidemias (Simplificado)',
        'variables': [
            'S(t): Proporción de susceptibles',
            'I(t): Proporción de infectados',
            'R(t) = 1 - S(t) - I(t): Proporción de recuperados'
        ],
        'ecuaciones': [
            f'dS/dt = -{beta}SI',
            f'dI/dt = {beta}SI - {gamma}I'
        ],
        'parametros': {
            'beta': beta,
            'gamma': gamma,
            'R0': R0
        },
        'referencia': 'Kermack & McKendrick (1927). Proc. Royal Society A, 115(772), 700-721.',
        'interpretacion_R0': f'R₀ = {R0:.2f} → ' + (
            'Epidemia se propaga' if R0 > 1 else 'Epidemia no se propaga'
        )
    }
    
    return f, t0, X0, parametros, info


def analisis_sir(X: np.ndarray, t: np.ndarray, params: Dict[str, float]):
    """
    Análisis del comportamiento del modelo SIR.
    
    Calcula:
    - Pico de infectados
    - Proporción final de afectados
    - Tiempo al pico
    
    Parámetros:
    -----------
    X : np.ndarray
        Matriz de soluciones [tiempo, 2]
    t : np.ndarray
        Vector de tiempos
    params : dict
        Parámetros del modelo
    
    Retorna:
    --------
    analisis : dict
        Diccionario con resultados del análisis
    """
    S = X[:, 0]  # Susceptibles
    I = X[:, 1]  # Infectados
    R = 1 - S - I  # Recuperados
    
    # Encontrar pico de infectados
    idx_pico = np.argmax(I)
    
    analisis = {
        'R0': params['beta'] / params['gamma'],
        'pico_infectados': {
            'proporcion': I[idx_pico],
            'porcentaje': I[idx_pico] * 100,
            'tiempo': t[idx_pico]
        },
        'estado_final': {
            'susceptibles': S[-1],
            'infectados': I[-1],
            'recuperados': R[-1],
            'total_afectados': 1 - S[-1]
        },
        'porcentaje_final_afectados': (1 - S[-1]) * 100
    }
    
    return analisis


#FUNCIÓN PRINCIPAL PARA RESOLVER SISTEMAS NO LINEALES

def resolver_sistema_no_lineal(sistema: str, metodo, nombre_metodo: str,
                               h: float = 0.01, tf: float = 50.0,
                               parametros: Dict[str, float] = None):
    """
    Resuelve un sistema no lineal con un método numérico dado.
    
    Parámetros:
    -----------
    sistema : str
        'lotka-volterra' o 'sir'
    metodo : Callable
        Método numérico (euler o rk4)
    nombre_metodo : str
        Nombre del método
    h : float
        Tamaño de paso
    tf : float
        Tiempo final
    parametros : dict
        Parámetros del sistema (opcional)
    
    Retorna:
    --------
    resultados : dict
        Diccionario con tiempos, soluciones, info y análisis
    """
    # Obtener sistema
    if sistema.lower() == 'lotka-volterra':
        f, t0, X0, params, info = lotka_volterra(parametros)
        analisis_func = analisis_lotka_volterra
    elif sistema.lower() == 'sir':
        f, t0, X0, params, info = modelo_sir(parametros)
        analisis_func = analisis_sir
    else:
        raise ValueError(f"Sistema '{sistema}' no reconocido. Use 'lotka-volterra' o 'sir'")
    
    # Resolver
    t, X = metodo(f, t0, X0, tf, h)
    
    # Análisis
    analisis = analisis_func(X, t, params)
    
    resultados = {
        'tiempo': t,
        'solucion': X,
        'parametros': params,
        'info': info,
        'analisis': analisis,
        'metodo': nombre_metodo,
        'h': h
    }
    
    return resultados


if __name__ == "__main__":
    print("="*70)
    print("SISTEMAS NO LINEALES - INFORMACIÓN")
    print("="*70)
    
    # Lotka-Volterra
    print("\n1. MODELO DE LOTKA-VOLTERRA")
    print("-" * 70)
    f_lv, t0_lv, X0_lv, params_lv, info_lv = lotka_volterra()
    print(f"Nombre: {info_lv['nombre']}")
    print(f"Variables:")
    for var in info_lv['variables']:
        print(f"  - {var}")
    print(f"Ecuaciones:")
    for eq in info_lv['ecuaciones']:
        print(f"  - {eq}")
    print(f"Punto de equilibrio no trivial: {info_lv['punto_equilibrio_no_trivial']}")
    
    # SIR
    print("\n2. MODELO SIR DE EPIDEMIAS")
    print("-" * 70)
    f_sir, t0_sir, X0_sir, params_sir, info_sir = modelo_sir()
    print(f"Nombre: {info_sir['nombre']}")
    print(f"Variables:")
    for var in info_sir['variables']:
        print(f"  - {var}")
    print(f"Ecuaciones:")
    for eq in info_sir['ecuaciones']:
        print(f"  - {eq}")
    print(f"{info_sir['interpretacion_R0']}")
    
    print("\n✓ Módulo de sistemas no lineales cargado exitosamente")