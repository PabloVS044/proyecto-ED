"""
Módulo de Visualización
Proyecto Final - Ecuaciones Diferenciales 1

Genera todos los gráficos necesarios para el informe.
"""

import numpy as np
import matplotlib.pyplot as plt
from metodos_numericos import euler, rk4
from validacion import edo_primer_orden, edo_segundo_orden, sistema_2x2_lineal
from sistema_no_lineal import resolver_sistema_no_lineal

# Configuración global de matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.linewidth'] = 2.5


def graficar_validacion_primer_orden(guardar=False):
    """Gráfico comparativo para EDO de primer orden"""
    f, sol_exact, t0, y0 = edo_primer_orden()
    tf = 2.0
    h = 0.05
    
    # Resolver con ambos métodos
    t_euler, y_euler = euler(f, t0, y0, tf, h)
    t_rk4, y_rk4 = rk4(f, t0, y0, tf, h)
    
    # Solución analítica
    t_exact = np.linspace(t0, tf, 500)
    y_exact = sol_exact(t_exact)
    
    # Crear figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfico 1: Comparación de soluciones
    ax1.plot(t_exact, y_exact, 'k-', label='Solución Analítica', linewidth=3)
    ax1.plot(t_euler, y_euler, 'ro--', label='Euler', markersize=6, markevery=5)
    ax1.plot(t_rk4, y_rk4, 'bs--', label='RK4', markersize=6, markevery=5)
    ax1.set_xlabel('Tiempo (t)')
    ax1.set_ylabel('y(t)')
    ax1.set_title('EDO de Primer Orden: dy/dt = -2y + t, y(0) = 1')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Error absoluto
    error_euler = np.abs(y_euler - sol_exact(t_euler))
    error_rk4 = np.abs(y_rk4 - sol_exact(t_rk4))
    
    ax2.semilogy(t_euler, error_euler, 'ro-', label='Error Euler', markersize=5)
    ax2.semilogy(t_rk4, error_rk4, 'bs-', label='Error RK4', markersize=5)
    ax2.set_xlabel('Tiempo (t)')
    ax2.set_ylabel('Error Absoluto (escala log)')
    ax2.set_title('Error Absoluto vs Tiempo')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    if guardar:
        plt.savefig('validacion_primer_orden.png', dpi=300, bbox_inches='tight')
    plt.show()


def graficar_validacion_segundo_orden(guardar=False):
    """Gráfico comparativo para EDO de segundo orden"""
    g, f, sol_exact, t0, y0, yp0 = edo_segundo_orden()
    Y0 = np.array([y0, yp0])
    tf = 2 * np.pi
    h = 0.05
    
    # Resolver con ambos métodos
    t_euler, Y_euler = euler(f, t0, Y0, tf, h)
    t_rk4, Y_rk4 = rk4(f, t0, Y0, tf, h)
    
    # Solución analítica
    t_exact = np.linspace(t0, tf, 500)
    Y_exact = sol_exact(t_exact)
    
    # Crear figura
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gráfico 1: y(t)
    axes[0, 0].plot(t_exact, Y_exact[:, 0], 'k-', label='Analítica', linewidth=3)
    axes[0, 0].plot(t_euler, Y_euler[:, 0], 'ro--', label='Euler', markersize=5, markevery=8)
    axes[0, 0].plot(t_rk4, Y_rk4[:, 0], 'bs--', label='RK4', markersize=5, markevery=8)
    axes[0, 0].set_xlabel('Tiempo (t)')
    axes[0, 0].set_ylabel('y(t)')
    axes[0, 0].set_title('Posición: y(t) = sin(t)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gráfico 2: y'(t)
    axes[0, 1].plot(t_exact, Y_exact[:, 1], 'k-', label='Analítica', linewidth=3)
    axes[0, 1].plot(t_euler, Y_euler[:, 1], 'ro--', label='Euler', markersize=5, markevery=8)
    axes[0, 1].plot(t_rk4, Y_rk4[:, 1], 'bs--', label='RK4', markersize=5, markevery=8)
    axes[0, 1].set_xlabel('Tiempo (t)')
    axes[0, 1].set_ylabel("y'(t)")
    axes[0, 1].set_title("Velocidad: y'(t) = cos(t)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gráfico 3: Error en y(t)
    error_y_euler = np.abs(Y_euler[:, 0] - sol_exact(t_euler)[:, 0])
    error_y_rk4 = np.abs(Y_rk4[:, 0] - sol_exact(t_rk4)[:, 0])
    
    axes[1, 0].semilogy(t_euler, error_y_euler, 'ro-', label='Euler', markersize=4)
    axes[1, 0].semilogy(t_rk4, error_y_rk4, 'bs-', label='RK4', markersize=4)
    axes[1, 0].set_xlabel('Tiempo (t)')
    axes[1, 0].set_ylabel('Error Absoluto en y(t)')
    axes[1, 0].set_title('Error en Posición')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, which='both')
    
    # Gráfico 4: Plano fase
    axes[1, 1].plot(Y_exact[:, 0], Y_exact[:, 1], 'k-', label='Analítica', linewidth=3)
    axes[1, 1].plot(Y_euler[:, 0], Y_euler[:, 1], 'ro--', label='Euler', markersize=4, markevery=10)
    axes[1, 1].plot(Y_rk4[:, 0], Y_rk4[:, 1], 'bs--', label='RK4', markersize=4, markevery=10)
    axes[1, 1].set_xlabel('y(t)')
    axes[1, 1].set_ylabel("y'(t)")
    axes[1, 1].set_title('Plano Fase')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axis('equal')
    
    plt.suptitle('EDO de Segundo Orden: d²y/dt² + y = 0', fontsize=16, y=1.00)
    plt.tight_layout()
    if guardar:
        plt.savefig('validacion_segundo_orden.png', dpi=300, bbox_inches='tight')
    plt.show()


def graficar_validacion_sistema_2x2(guardar=False):
    """Gráfico comparativo para sistema 2×2"""
    f, sol_exact, t0, X0 = sistema_2x2_lineal()
    tf = 2.0
    h = 0.02
    
    # Resolver con ambos métodos
    t_euler, X_euler = euler(f, t0, X0, tf, h)
    t_rk4, X_rk4 = rk4(f, t0, X0, tf, h)
    
    # Solución analítica
    t_exact = np.linspace(t0, tf, 500)
    X_exact = sol_exact(t_exact)
    
    # Crear figura
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gráfico 1: x(t)
    axes[0, 0].plot(t_exact, X_exact[:, 0], 'k-', label='Analítica', linewidth=3)
    axes[0, 0].plot(t_euler, X_euler[:, 0], 'ro--', label='Euler', markersize=5, markevery=5)
    axes[0, 0].plot(t_rk4, X_rk4[:, 0], 'bs--', label='RK4', markersize=5, markevery=5)
    axes[0, 0].set_xlabel('Tiempo (t)')
    axes[0, 0].set_ylabel('x(t)')
    axes[0, 0].set_title('Primera Variable: x(t) = e^(2t)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gráfico 2: y(t)
    axes[0, 1].plot(t_exact, X_exact[:, 1], 'k-', label='Analítica', linewidth=3)
    axes[0, 1].plot(t_euler, X_euler[:, 1], 'ro--', label='Euler', markersize=5, markevery=5)
    axes[0, 1].plot(t_rk4, X_rk4[:, 1], 'bs--', label='RK4', markersize=5, markevery=5)
    axes[0, 1].set_xlabel('Tiempo (t)')
    axes[0, 1].set_ylabel('y(t)')
    axes[0, 1].set_title('Segunda Variable: y(t) = e^(-t)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gráfico 3: Error en x(t)
    error_x_euler = np.abs(X_euler[:, 0] - sol_exact(t_euler)[:, 0])
    error_x_rk4 = np.abs(X_rk4[:, 0] - sol_exact(t_rk4)[:, 0])
    
    axes[1, 0].semilogy(t_euler, error_x_euler, 'ro-', label='Euler', markersize=4)
    axes[1, 0].semilogy(t_rk4, error_x_rk4, 'bs-', label='RK4', markersize=4)
    axes[1, 0].set_xlabel('Tiempo (t)')
    axes[1, 0].set_ylabel('Error Absoluto en x(t)')
    axes[1, 0].set_title('Error en x(t)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, which='both')
    
    # Gráfico 4: Plano fase
    axes[1, 1].plot(X_exact[:, 0], X_exact[:, 1], 'k-', label='Analítica', linewidth=3)
    axes[1, 1].plot(X_euler[:, 0], X_euler[:, 1], 'ro--', label='Euler', markersize=4, markevery=10)
    axes[1, 1].plot(X_rk4[:, 0], X_rk4[:, 1], 'bs--', label='RK4', markersize=4, markevery=10)
    axes[1, 1].set_xlabel('x(t)')
    axes[1, 1].set_ylabel('y(t)')
    axes[1, 1].set_title('Plano Fase')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Sistema 2×2 Lineal: dx/dt = 2x, dy/dt = -y', fontsize=16, y=1.00)
    plt.tight_layout()
    if guardar:
        plt.savefig('validacion_sistema_2x2.png', dpi=300, bbox_inches='tight')
    plt.show()


def graficar_convergencia(guardar=False):
    """Gráfico de convergencia para los tres casos"""
    from metodos_numericos import estudio_convergencia
    
    tamaños_paso = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    
    # EDO primer orden
    f1, sol1, t0_1, y0_1 = edo_primer_orden()
    conv_euler_1 = estudio_convergencia(f1, euler, t0_1, y0_1, 2.0, tamaños_paso, sol1)
    conv_rk4_1 = estudio_convergencia(f1, rk4, t0_1, y0_1, 2.0, tamaños_paso, sol1)
    
    # EDO segundo orden
    g2, f2, sol2, t0_2, y0_2, yp0_2 = edo_segundo_orden()
    Y0_2 = np.array([y0_2, yp0_2])
    conv_euler_2 = estudio_convergencia(f2, euler, t0_2, Y0_2, 2*np.pi, tamaños_paso, sol2)
    conv_rk4_2 = estudio_convergencia(f2, rk4, t0_2, Y0_2, 2*np.pi, tamaños_paso, sol2)
    
    # Sistema 2x2
    f3, sol3, t0_3, X0_3 = sistema_2x2_lineal()
    conv_euler_3 = estudio_convergencia(f3, euler, t0_3, X0_3, 2.0, tamaños_paso, sol3)
    conv_rk4_3 = estudio_convergencia(f3, rk4, t0_3, X0_3, 2.0, tamaños_paso, sol3)
    
    # Crear figura
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Gráfico 1: EDO primer orden
    axes[0].loglog(conv_euler_1['pasos'], conv_euler_1['errores_max'], 
                   'ro-', label='Euler', markersize=8, linewidth=2)
    axes[0].loglog(conv_rk4_1['pasos'], conv_rk4_1['errores_max'], 
                   'bs-', label='RK4', markersize=8, linewidth=2)
    axes[0].set_xlabel('Tamaño de Paso (h)')
    axes[0].set_ylabel('Error Máximo')
    axes[0].set_title('EDO de Primer Orden')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, which='both')
    
    # Gráfico 2: EDO segundo orden
    axes[1].loglog(conv_euler_2['pasos'], conv_euler_2['errores_max'], 
                   'ro-', label='Euler', markersize=8, linewidth=2)
    axes[1].loglog(conv_rk4_2['pasos'], conv_rk4_2['errores_max'], 
                   'bs-', label='RK4', markersize=8, linewidth=2)
    axes[1].set_xlabel('Tamaño de Paso (h)')
    axes[1].set_ylabel('Error Máximo')
    axes[1].set_title('EDO de Segundo Orden')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which='both')
    
    # Gráfico 3: Sistema 2x2
    axes[2].loglog(conv_euler_3['pasos'], conv_euler_3['errores_max'], 
                   'ro-', label='Euler', markersize=8, linewidth=2)
    axes[2].loglog(conv_rk4_3['pasos'], conv_rk4_3['errores_max'], 
                   'bs-', label='RK4', markersize=8, linewidth=2)
    axes[2].set_xlabel('Tamaño de Paso (h)')
    axes[2].set_ylabel('Error Máximo')
    axes[2].set_title('Sistema 2×2 Lineal')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, which='both')
    
    plt.suptitle('Estudio de Convergencia - Error vs Tamaño de Paso', fontsize=16, y=1.02)
    plt.tight_layout()
    if guardar:
        plt.savefig('convergencia.png', dpi=300, bbox_inches='tight')
    plt.show()


def graficar_lotka_volterra(guardar=False):
    """Gráficos para modelo de Lotka-Volterra"""
    h = 0.01
    tf = 50.0
    
    # Resolver con ambos métodos
    lv_euler = resolver_sistema_no_lineal('lotka-volterra', euler, 'Euler', h, tf)
    lv_rk4 = resolver_sistema_no_lineal('lotka-volterra', rk4, 'RK4', h, tf)
    
    # Crear figura
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gráfico 1: Presas vs Tiempo
    axes[0, 0].plot(lv_euler['tiempo'], lv_euler['solucion'][:, 0], 
                    'r-', label='Euler', linewidth=2)
    axes[0, 0].plot(lv_rk4['tiempo'], lv_rk4['solucion'][:, 0], 
                    'b--', label='RK4', linewidth=2)
    axes[0, 0].set_xlabel('Tiempo')
    axes[0, 0].set_ylabel('Población de Presas')
    axes[0, 0].set_title('Dinámica de Presas')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gráfico 2: Predadores vs Tiempo
    axes[0, 1].plot(lv_euler['tiempo'], lv_euler['solucion'][:, 1], 
                    'r-', label='Euler', linewidth=2)
    axes[0, 1].plot(lv_rk4['tiempo'], lv_rk4['solucion'][:, 1], 
                    'b--', label='RK4', linewidth=2)
    axes[0, 1].set_xlabel('Tiempo')
    axes[0, 1].set_ylabel('Población de Predadores')
    axes[0, 1].set_title('Dinámica de Predadores')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gráfico 3: Plano Fase
    axes[1, 0].plot(lv_euler['solucion'][:, 0], lv_euler['solucion'][:, 1], 
                    'r-', label='Euler', linewidth=2, alpha=0.7)
    axes[1, 0].plot(lv_rk4['solucion'][:, 0], lv_rk4['solucion'][:, 1], 
                    'b--', label='RK4', linewidth=2, alpha=0.7)
    axes[1, 0].plot([1.0], [1.0], 'go', markersize=10, label='Condición Inicial')
    axes[1, 0].set_xlabel('Presas')
    axes[1, 0].set_ylabel('Predadores')
    axes[1, 0].set_title('Plano Fase: Predadores vs Presas')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gráfico 4: Comparación directa
    axes[1, 1].plot(lv_euler['tiempo'], lv_euler['solucion'][:, 0], 
                    'r-', label='Presas (Euler)', linewidth=2)
    axes[1, 1].plot(lv_euler['tiempo'], lv_euler['solucion'][:, 1], 
                    'b-', label='Predadores (Euler)', linewidth=2)
    axes[1, 1].set_xlabel('Tiempo')
    axes[1, 1].set_ylabel('Población')
    axes[1, 1].set_title('Ambas Poblaciones (Método de Euler)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Modelo de Lotka-Volterra: Dinámica Predador-Presa', fontsize=16, y=1.00)
    plt.tight_layout()
    if guardar:
        plt.savefig('lotka_volterra.png', dpi=300, bbox_inches='tight')
    plt.show()


def graficar_sir(guardar=False):
    """Gráficos para modelo SIR"""
    h = 0.01
    tf = 100.0
    
    # Resolver con ambos métodos
    sir_euler = resolver_sistema_no_lineal('sir', euler, 'Euler', h, tf)
    sir_rk4 = resolver_sistema_no_lineal('sir', rk4, 'RK4', h, tf)
    
    # Calcular R (recuperados)
    R_euler = 1 - sir_euler['solucion'][:, 0] - sir_euler['solucion'][:, 1]
    R_rk4 = 1 - sir_rk4['solucion'][:, 0] - sir_rk4['solucion'][:, 1]
    
    # Crear figura
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gráfico 1: Todas las categorías (Euler)
    axes[0, 0].plot(sir_euler['tiempo'], sir_euler['solucion'][:, 0], 
                    'b-', label='Susceptibles (S)', linewidth=2.5)
    axes[0, 0].plot(sir_euler['tiempo'], sir_euler['solucion'][:, 1], 
                    'r-', label='Infectados (I)', linewidth=2.5)
    axes[0, 0].plot(sir_euler['tiempo'], R_euler, 
                    'g-', label='Recuperados (R)', linewidth=2.5)
    axes[0, 0].set_xlabel('Tiempo (días)')
    axes[0, 0].set_ylabel('Proporción de la Población')
    axes[0, 0].set_title('Modelo SIR - Método de Euler')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    # Gráfico 2: Todas las categorías (RK4)
    axes[0, 1].plot(sir_rk4['tiempo'], sir_rk4['solucion'][:, 0], 
                    'b-', label='Susceptibles (S)', linewidth=2.5)
    axes[0, 1].plot(sir_rk4['tiempo'], sir_rk4['solucion'][:, 1], 
                    'r-', label='Infectados (I)', linewidth=2.5)
    axes[0, 1].plot(sir_rk4['tiempo'], R_rk4, 
                    'g-', label='Recuperados (R)', linewidth=2.5)
    axes[0, 1].set_xlabel('Tiempo (días)')
    axes[0, 1].set_ylabel('Proporción de la Población')
    axes[0, 1].set_title('Modelo SIR - Método de RK4')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # Gráfico 3: Comparación de infectados
    axes[1, 0].plot(sir_euler['tiempo'], sir_euler['solucion'][:, 1]*100, 
                    'r-', label='Euler', linewidth=2.5)
    axes[1, 0].plot(sir_rk4['tiempo'], sir_rk4['solucion'][:, 1]*100, 
                    'b--', label='RK4', linewidth=2.5)
    axes[1, 0].axhline(y=sir_euler['analisis']['pico_infectados']['porcentaje'], 
                      color='gray', linestyle=':', label='Pico Euler')
    axes[1, 0].set_xlabel('Tiempo (días)')
    axes[1, 0].set_ylabel('Infectados (%)')
    axes[1, 0].set_title('Comparación: Dinámica de Infectados')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gráfico 4: Plano Fase S-I
    axes[1, 1].plot(sir_euler['solucion'][:, 0], sir_euler['solucion'][:, 1], 
                    'r-', label='Euler', linewidth=2.5, alpha=0.7)
    axes[1, 1].plot(sir_rk4['solucion'][:, 0], sir_rk4['solucion'][:, 1], 
                    'b--', label='RK4', linewidth=2.5, alpha=0.7)
    axes[1, 1].plot([0.99], [0.01], 'go', markersize=10, label='Condición Inicial')
    axes[1, 1].set_xlabel('Susceptibles (S)')
    axes[1, 1].set_ylabel('Infectados (I)')
    axes[1, 1].set_title('Plano Fase: Infectados vs Susceptibles')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Modelo SIR de Epidemias (R₀ = {sir_euler["analisis"]["R0"]:.1f})', 
                 fontsize=16, y=1.00)
    plt.tight_layout()
    if guardar:
        plt.savefig('modelo_sir.png', dpi=300, bbox_inches='tight')
    plt.show()


def generar_todos_los_graficos(guardar=True):
    """Genera todos los gráficos del proyecto"""
    print("Generando todos los gráficos...")
    print("\n1. Validación - EDO de Primer Orden...")
    graficar_validacion_primer_orden(guardar)
    
    print("2. Validación - EDO de Segundo Orden...")
    graficar_validacion_segundo_orden(guardar)
    
    print("3. Validación - Sistema 2×2...")
    graficar_validacion_sistema_2x2(guardar)
    
    print("4. Estudio de Convergencia...")
    graficar_convergencia(guardar)
    
    print("5. Modelo de Lotka-Volterra...")
    graficar_lotka_volterra(guardar)
    
    print("6. Modelo SIR de Epidemias...")
    graficar_sir(guardar)
    
    print("\n✓ Todos los gráficos generados exitosamente!")
    if guardar:
        print("Los gráficos se guardaron en el directorio actual.")


if __name__ == "__main__":
    import sys
    
    guardar = '--guardar' in sys.argv or '-g' in sys.argv
    
    print("="*70)
    print("GENERACIÓN DE GRÁFICOS - PROYECTO FINAL")
    print("="*70)
    
    generar_todos_los_graficos(guardar=guardar)
    
    print("\nUso: python visualizacion.py [--guardar | -g]")
    print("  --guardar, -g: Guarda los gráficos como imágenes PNG")