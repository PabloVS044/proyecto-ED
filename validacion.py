"""
Módulo de Validación - Soluciones Analíticas
Proyecto Final - Ecuaciones Diferenciales 1

Este módulo contiene las soluciones analíticas de las EDOs
utilizadas para validar los métodos numéricos.
"""

import numpy as np
from typing import Callable


# ============================================================================
# EDO DE PRIMER ORDEN
# ============================================================================

def edo_primer_orden():
    """
    EDO: dy/dt = -2y + t, y(0) = 1
    
    Solución analítica:
    Factor integrante: μ(t) = e^(2t)
    Solución: y(t) = (1/2)t - 1/4 + (5/4)e^(-2t)
    
    Retorna:
    --------
    f : función de la EDO
    solucion_analitica : función con solución exacta
    t0, y0 : condiciones iniciales
    """
    def f(t, y):
        return -2*y + t
    
    def solucion_analitica(t):
        return 0.5*t - 0.25 + 1.25*np.exp(-2*t)
    
    t0 = 0.0
    y0 = 1.0
    
    return f, solucion_analitica, t0, y0


def derivar_solucion_primer_orden():
    """
    Derivación de la solución analítica para:
    dy/dt = -2y + t, y(0) = 1
    
    Pasos:
    1. EDO lineal de primer orden: dy/dt + 2y = t
    2. Factor integrante: μ(t) = e^(∫2 dt) = e^(2t)
    3. Multiplicar: e^(2t)dy/dt + 2e^(2t)y = te^(2t)
    4. Lado izquierdo es d/dt[e^(2t)y]
    5. Integrar: e^(2t)y = ∫te^(2t)dt
    6. Integral por partes: ∫te^(2t)dt = (t/2)e^(2t) - (1/4)e^(2t) + C
    7. y = (t/2) - 1/4 + Ce^(-2t)
    8. Aplicar y(0) = 1: 1 = 0 - 1/4 + C → C = 5/4
    9. Solución: y(t) = (t/2) - 1/4 + (5/4)e^(-2t)
    """
    return """
    EDO: dy/dt = -2y + t, y(0) = 1
    
    Forma estándar: dy/dt + 2y = t
    Factor integrante: μ(t) = e^(2t)
    
    Solución general: y(t) = (t/2) - 1/4 + Ce^(-2t)
    Con y(0) = 1: C = 5/4
    
    Solución particular: y(t) = (1/2)t - 1/4 + (5/4)e^(-2t)
    """


# ============================================================================
# EDO DE SEGUNDO ORDEN
# ============================================================================

def edo_segundo_orden():
    """
    EDO: d²y/dt² + y = 0, y(0) = 0, y'(0) = 1
    
    Solución analítica: y(t) = sin(t)
    
    Retorna:
    --------
    g : función g(t, y, yp) para d²y/dt² = g(t, y, yp)
    f_sistema : función del sistema de primer orden
    solucion_analitica : función con solución exacta
    t0, y0, yp0 : condiciones iniciales
    """
    def g(t, y, yp):
        """d²y/dt² = -y"""
        return -y
    
    def f_sistema(t, Y):
        """
        Sistema de primer orden:
        y₁' = y₂
        y₂' = -y₁
        """
        y1, y2 = Y
        return np.array([y2, -y1])
    
    def solucion_analitica(t):
        """
        Retorna [y(t), y'(t)] = [sin(t), cos(t)]
        """
        y = np.sin(t)
        yp = np.cos(t)
        return np.column_stack([y, yp]) if isinstance(t, np.ndarray) else np.array([y, yp])
    
    t0 = 0.0
    y0 = 0.0
    yp0 = 1.0
    
    return g, f_sistema, solucion_analitica, t0, y0, yp0


def derivar_solucion_segundo_orden():
    """
    Derivación de la solución analítica para:
    d²y/dt² + y = 0, y(0) = 0, y'(0) = 1
    
    Pasos:
    1. Ecuación característica: r² + 1 = 0
    2. Raíces: r = ±i
    3. Solución general: y(t) = c₁cos(t) + c₂sin(t)
    4. Aplicar y(0) = 0: c₁ = 0
    5. y'(t) = -c₁sin(t) + c₂cos(t)
    6. Aplicar y'(0) = 1: c₂ = 1
    7. Solución: y(t) = sin(t)
    """
    return """
    EDO: d²y/dt² + y = 0, y(0) = 0, y'(0) = 1
    
    Ecuación característica: r² + 1 = 0
    Raíces: r = ±i (complejas conjugadas)
    
    Solución general: y(t) = c₁cos(t) + c₂sin(t)
    
    Con y(0) = 0: c₁ = 0
    Con y'(0) = 1: c₂ = 1
    
    Solución particular: y(t) = sin(t)
    """


# ============================================================================
# SISTEMA 2×2 LINEAL
# ============================================================================

def sistema_2x2_lineal():
    """
    Sistema desacoplado:
    dx/dt = 2x, x(0) = 1
    dy/dt = -y, y(0) = 1
    
    Solución analítica:
    x(t) = e^(2t)
    y(t) = e^(-t)
    
    Retorna:
    --------
    f : función del sistema
    solucion_analitica : función con solución exacta
    t0, X0 : condiciones iniciales
    """
    def f(t, X):
        x, y = X
        return np.array([2*x, -y])
    
    def solucion_analitica(t):
        x = np.exp(2*t)
        y = np.exp(-t)
        return np.column_stack([x, y]) if isinstance(t, np.ndarray) else np.array([x, y])
    
    t0 = 0.0
    X0 = np.array([1.0, 1.0])
    
    return f, solucion_analitica, t0, X0


def derivar_solucion_sistema_lineal():
    """
    Derivación de la solución analítica para:
    dx/dt = 2x, x(0) = 1
    dy/dt = -y, y(0) = 1
    
    Sistema desacoplado - cada ecuación se resuelve independientemente.
    """
    return """
    Sistema desacoplado:
    dx/dt = 2x, x(0) = 1
    dy/dt = -y, y(0) = 1
    
    Primera ecuación:
    dx/x = 2dt
    ln|x| = 2t + C₁
    x(t) = Ae^(2t)
    Con x(0) = 1: A = 1
    Solución: x(t) = e^(2t)
    
    Segunda ecuación:
    dy/y = -dt
    ln|y| = -t + C₂
    y(t) = Be^(-t)
    Con y(0) = 1: B = 1
    Solución: y(t) = e^(-t)
    """


# ============================================================================
# FUNCIONES AUXILIARES PARA VALIDACIÓN
# ============================================================================

def ejecutar_validacion_completa(metodo, nombre_metodo: str, h: float = 0.01):
    """
    Ejecuta validación completa de un método numérico.
    
    Parámetros:
    -----------
    metodo : Callable
        Método numérico (euler o rk4)
    nombre_metodo : str
        Nombre del método para reportes
    h : float
        Tamaño de paso
    """
    print("="*70)
    print(f"VALIDACIÓN DEL MÉTODO: {nombre_metodo}")
    print("="*70)
    
    # 1. EDO de primer orden
    print("\n1. EDO DE PRIMER ORDEN: dy/dt = -2y + t, y(0) = 1")
    print("-" * 70)
    f1, sol1, t0_1, y0_1 = edo_primer_orden()
    tf_1 = 2.0
    
    t1, y1_num = metodo(f1, t0_1, y0_1, tf_1, h)
    y1_exact = sol1(t1)
    error1 = np.max(np.abs(y1_num - y1_exact))
    
    print(f"Tiempo final: t = {tf_1}")
    print(f"Solución numérica: y({tf_1}) = {y1_num[-1]:.8f}")
    print(f"Solución analítica: y({tf_1}) = {y1_exact[-1]:.8f}")
    print(f"Error máximo: {error1:.2e}")
    
    # 2. EDO de segundo orden
    print("\n2. EDO DE SEGUNDO ORDEN: d²y/dt² + y = 0, y(0) = 0, y'(0) = 1")
    print("-" * 70)
    g2, f2, sol2, t0_2, y0_2, yp0_2 = edo_segundo_orden()
    tf_2 = 2*np.pi
    Y0_2 = np.array([y0_2, yp0_2])
    
    t2, Y2_num = metodo(f2, t0_2, Y0_2, tf_2, h)
    Y2_exact = sol2(t2)
    error2_y = np.max(np.abs(Y2_num[:, 0] - Y2_exact[:, 0]))
    error2_yp = np.max(np.abs(Y2_num[:, 1] - Y2_exact[:, 1]))
    
    print(f"Tiempo final: t = {tf_2:.4f}")
    print(f"Solución numérica: y({tf_2:.4f}) = {Y2_num[-1, 0]:.8f}")
    print(f"Solución analítica: y({tf_2:.4f}) = {Y2_exact[-1, 0]:.8f}")
    print(f"Error máximo en y: {error2_y:.2e}")
    print(f"Error máximo en y': {error2_yp:.2e}")
    
    # 3. Sistema 2×2
    print("\n3. SISTEMA 2×2 LINEAL: dx/dt = 2x, dy/dt = -y")
    print("-" * 70)
    f3, sol3, t0_3, X0_3 = sistema_2x2_lineal()
    tf_3 = 2.0
    
    t3, X3_num = metodo(f3, t0_3, X0_3, tf_3, h)
    X3_exact = sol3(t3)
    error3_x = np.max(np.abs(X3_num[:, 0] - X3_exact[:, 0]))
    error3_y = np.max(np.abs(X3_num[:, 1] - X3_exact[:, 1]))
    
    print(f"Tiempo final: t = {tf_3}")
    print(f"Solución numérica: x({tf_3}) = {X3_num[-1, 0]:.8f}, y({tf_3}) = {X3_num[-1, 1]:.8f}")
    print(f"Solución analítica: x({tf_3}) = {X3_exact[-1, 0]:.8f}, y({tf_3}) = {X3_exact[-1, 1]:.8f}")
    print(f"Error máximo en x: {error3_x:.2e}")
    print(f"Error máximo en y: {error3_y:.2e}")
    
    print("\n" + "="*70)
    print("VALIDACIÓN COMPLETA ✓")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Mostrar derivaciones
    print("DERIVACIONES DE SOLUCIONES ANALÍTICAS")
    print("="*70)
    
    print("\n" + derivar_solucion_primer_orden())
    print("\n" + derivar_solucion_segundo_orden())
    print("\n" + derivar_solucion_sistema_lineal())
    
    print("\n✓ Módulo de validación cargado exitosamente")