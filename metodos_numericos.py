"""
Métodos Numéricos para Ecuaciones Diferenciales Ordinarias
Proyecto Final - Ecuaciones Diferenciales 1
Segundo Ciclo 2025

Este módulo implementa métodos numéricos iterativos para resolver EDOs:
- Método de Euler (Euler Explícito)
- Método de Runge-Kutta de 4º orden (RK4)

Generado con asistencia de IA como copiloto.
"""

import numpy as np
from typing import Callable, Tuple, Union


def euler(f: Callable, t0: float, y0: Union[float, np.ndarray], 
          tf: float, h: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Método de Euler para resolver EDOs de primer orden o sistemas de EDOs.
    
    Fórmula: y_{n+1} = y_n + h * f(t_n, y_n)
    
    Parámetros:
    -----------
    f : Callable
        Función que define la EDO: dy/dt = f(t, y)
        Para sistemas: debe retornar array de derivadas
    t0 : float
        Tiempo inicial
    y0 : float o np.ndarray
        Condición inicial (escalar para EDO simple, array para sistemas)
    tf : float
        Tiempo final
    h : float
        Tamaño de paso (step size)
    
    Retorna:
    --------
    t : np.ndarray
        Array de valores de tiempo
    y : np.ndarray
        Array de soluciones (1D para EDO simple, 2D para sistemas)
    
    Ejemplo:
    --------
    >>> def f(t, y): return -2*y + t
    >>> t, y = euler(f, 0, 1, 2, 0.1)
    """
    # Generar array de tiempos
    t = np.arange(t0, tf + h, h)
    n = len(t)
    
    # Determinar si es sistema o EDO simple
    y0_array = np.atleast_1d(y0)
    dim = len(y0_array)
    
    # Inicializar array de soluciones
    y = np.zeros((n, dim))
    y[0] = y0_array
    
    # Iteración de Euler
    for i in range(n - 1):
        y[i + 1] = y[i] + h * np.atleast_1d(f(t[i], y[i]))
    
    # Retornar en formato apropiado
    if dim == 1:
        return t, y.flatten()
    return t, y


def rk4(f: Callable, t0: float, y0: Union[float, np.ndarray], 
        tf: float, h: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Método de Runge-Kutta de 4º orden (RK4) para resolver EDOs o sistemas.
    
    Fórmulas:
    k1 = f(t_n, y_n)
    k2 = f(t_n + h/2, y_n + h*k1/2)
    k3 = f(t_n + h/2, y_n + h*k2/2)
    k4 = f(t_n + h, y_n + h*k3)
    y_{n+1} = y_n + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    Parámetros:
    -----------
    f : Callable
        Función que define la EDO: dy/dt = f(t, y)
        Para sistemas: debe retornar array de derivadas
    t0 : float
        Tiempo inicial
    y0 : float o np.ndarray
        Condición inicial (escalar para EDO simple, array para sistemas)
    tf : float
        Tiempo final
    h : float
        Tamaño de paso (step size)
    
    Retorna:
    --------
    t : np.ndarray
        Array de valores de tiempo
    y : np.ndarray
        Array de soluciones (1D para EDO simple, 2D para sistemas)
    
    Ejemplo:
    --------
    >>> def f(t, y): return -2*y + t
    >>> t, y = rk4(f, 0, 1, 2, 0.1)
    """
    # Generar array de tiempos
    t = np.arange(t0, tf + h, h)
    n = len(t)
    
    # Determinar si es sistema o EDO simple
    y0_array = np.atleast_1d(y0)
    dim = len(y0_array)
    
    # Inicializar array de soluciones
    y = np.zeros((n, dim))
    y[0] = y0_array
    
    # Iteración de RK4
    for i in range(n - 1):
        k1 = np.atleast_1d(f(t[i], y[i]))
        k2 = np.atleast_1d(f(t[i] + h/2, y[i] + h*k1/2))
        k3 = np.atleast_1d(f(t[i] + h/2, y[i] + h*k2/2))
        k4 = np.atleast_1d(f(t[i] + h, y[i] + h*k3))
        
        y[i + 1] = y[i] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    # Retornar en formato apropiado
    if dim == 1:
        return t, y.flatten()
    return t, y


def edo_segundo_orden_a_sistema(g: Callable, y0: float, yp0: float) -> Tuple[Callable, np.ndarray]:
    """
    Convierte una EDO de segundo orden en un sistema de primer orden.
    
    De: d²y/dt² = g(t, y, dy/dt)
    A:  dy₁/dt = y₂
        dy₂/dt = g(t, y₁, y₂)
    
    Parámetros:
    -----------
    g : Callable
        Función g(t, y, yp) que representa d²y/dt² = g(t, y, dy/dt)
    y0 : float
        Condición inicial y(t0)
    yp0 : float
        Condición inicial y'(t0)
    
    Retorna:
    --------
    f_sistema : Callable
        Función que define el sistema de primer orden
    condiciones_iniciales : np.ndarray
        Array [y0, yp0]
    
    Ejemplo:
    --------
    >>> # Para d²y/dt² + y = 0
    >>> def g(t, y, yp): return -y
    >>> f_sys, ic = edo_segundo_orden_a_sistema(g, 0, 1)
    """
    def f_sistema(t, Y):
        y, yp = Y
        return np.array([yp, g(t, y, yp)])
    
    condiciones_iniciales = np.array([y0, yp0])
    
    return f_sistema, condiciones_iniciales


# FUNCIONES DE UTILIDAD

def calcular_error(y_numerica: np.ndarray, y_analitica: np.ndarray, 
                   norma: str = 'max') -> float:
    """
    Calcula el error entre solución numérica y analítica.
    
    Parámetros:
    -----------
    y_numerica : np.ndarray
        Solución numérica
    y_analitica : np.ndarray
        Solución analítica
    norma : str
        Tipo de norma: 'max' (infinito), 'l2', 'mean'
    
    Retorna:
    --------
    error : float
        Error calculado según la norma especificada
    """
    diff = np.abs(y_numerica - y_analitica)
    
    if norma == 'max':
        return np.max(diff)
    elif norma == 'l2':
        return np.sqrt(np.mean(diff**2))
    elif norma == 'mean':
        return np.mean(diff)
    else:
        raise ValueError(f"Norma '{norma}' no reconocida. Use 'max', 'l2' o 'mean'")


def estudio_convergencia(f: Callable, metodo: Callable, 
                        t0: float, y0: Union[float, np.ndarray],
                        tf: float, tamaños_paso: list,
                        solucion_analitica: Callable) -> dict:
    """
    Realiza estudio de convergencia variando el tamaño de paso.
    
    Parámetros:
    -----------
    f : Callable
        Función que define la EDO
    metodo : Callable
        Método numérico (euler o rk4)
    t0, y0, tf : parámetros de la EDO
    tamaños_paso : list
        Lista de tamaños de paso a probar
    solucion_analitica : Callable
        Función que calcula la solución analítica
    
    Retorna:
    --------
    resultados : dict
        Diccionario con pasos, errores y soluciones
    """
    resultados = {
        'pasos': tamaños_paso,
        'errores_max': [],
        'errores_l2': [],
        'soluciones': []
    }
    
    for h in tamaños_paso:
        t, y = metodo(f, t0, y0, tf, h)
        y_exacta = solucion_analitica(t)
        
        # Calcular errores
        if y.ndim == 1:  # EDO simple
            error_max = calcular_error(y, y_exacta, 'max')
            error_l2 = calcular_error(y, y_exacta, 'l2')
        else:  # Sistema
            # Para sistemas, calcular error en cada componente
            error_max = max(calcular_error(y[:, i], y_exacta[:, i], 'max') 
                           for i in range(y.shape[1]))
            error_l2 = max(calcular_error(y[:, i], y_exacta[:, i], 'l2') 
                          for i in range(y.shape[1]))
        
        resultados['errores_max'].append(error_max)
        resultados['errores_l2'].append(error_l2)
        resultados['soluciones'].append((t, y))
    
    return resultados


if __name__ == "__main__":
    # Ejemplo de uso: EDO simple dy/dt = -2y + t, y(0) = 1
    print("="*60)
    print("EJEMPLO DE USO: dy/dt = -2y + t, y(0) = 1")
    print("="*60)
    
    def f_ejemplo(t, y):
        return -2*y + t
    
    # Resolver con ambos métodos
    h = 0.1
    t_euler, y_euler = euler(f_ejemplo, 0, 1, 2, h)
    t_rk4, y_rk4 = rk4(f_ejemplo, 0, 1, 2, h)
    
    print(f"\nCon h = {h}")
    print(f"Euler - Valor final: y({t_euler[-1]:.2f}) = {y_euler[-1]:.6f}")
    print(f"RK4   - Valor final: y({t_rk4[-1]:.2f}) = {y_rk4[-1]:.6f}")
    
    print("\n✓ Módulo cargado exitosamente")