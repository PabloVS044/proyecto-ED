"""
Script Principal - Proyecto Final
Ecuaciones Diferenciales 1 - Segundo Ciclo 2025

Este script ejecuta todo el proyecto:
1. Validación de métodos con soluciones analíticas
2. Estudio de convergencia
3. Resolución de sistemas no lineales
4. Generación de gráficos y tablas

Autores: [Nombres de los integrantes]
Fecha: Noviembre 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from metodos_numericos import euler, rk4, estudio_convergencia, calcular_error
from validacion import (edo_primer_orden, edo_segundo_orden, sistema_2x2_lineal,
                       ejecutar_validacion_completa)
from sistema_no_lineal import resolver_sistema_no_lineal
import warnings
warnings.filterwarnings('ignore')

# Configuración de matplotlib para mejores gráficos
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['lines.linewidth'] = 2


def imprimir_encabezado(titulo):
    """Imprime un encabezado formateado"""
    print("\n" + "="*80)
    print(titulo.center(80))
    print("="*80 + "\n")


def tabla_convergencia(resultados, nombre_edo):
    """Imprime tabla de convergencia"""
    print(f"\nTabla de Convergencia - {nombre_edo}")
    print("-" * 70)
    print(f"{'Paso (h)':<15} {'Error Máximo':<20} {'Error L2':<20}")
    print("-" * 70)
    for h, err_max, err_l2 in zip(resultados['pasos'], 
                                   resultados['errores_max'],
                                   resultados['errores_l2']):
        print(f"{h:<15.6f} {err_max:<20.8e} {err_l2:<20.8e}")
    print("-" * 70)


# ============================================================================
# PARTE 1: VALIDACIÓN CON SOLUCIONES ANALÍTICAS
# ============================================================================

def parte1_validacion():
    """Ejecuta la validación completa con ambos métodos"""
    imprimir_encabezado("PARTE 1: VALIDACIÓN CON SOLUCIONES ANALÍTICAS")
    
    # Validar Euler
    ejecutar_validacion_completa(euler, "MÉTODO DE EULER", h=0.01)
    
    # Validar RK4
    ejecutar_validacion_completa(rk4, "MÉTODO DE RUNGE-KUTTA 4", h=0.01)


# ============================================================================
# PARTE 2: ESTUDIO DE CONVERGENCIA
# ============================================================================

def parte2_convergencia():
    """Realiza estudio de convergencia variando el tamaño de paso"""
    imprimir_encabezado("PARTE 2: ESTUDIO DE CONVERGENCIA")
    
    # Tamaños de paso a probar
    tamaños_paso = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    
    print("Tamaños de paso a evaluar:", tamaños_paso)
    
    # 1. EDO de primer orden
    print("\n" + "="*80)
    print("1. CONVERGENCIA - EDO DE PRIMER ORDEN")
    print("="*80)
    f1, sol1, t0_1, y0_1 = edo_primer_orden()
    
    print("\nMétodo de Euler:")
    conv_euler_1 = estudio_convergencia(f1, euler, t0_1, y0_1, 2.0, 
                                       tamaños_paso, sol1)
    tabla_convergencia(conv_euler_1, "Euler - Primer Orden")
    
    print("\nMétodo de RK4:")
    conv_rk4_1 = estudio_convergencia(f1, rk4, t0_1, y0_1, 2.0,
                                     tamaños_paso, sol1)
    tabla_convergencia(conv_rk4_1, "RK4 - Primer Orden")
    
    # 2. EDO de segundo orden
    print("\n" + "="*80)
    print("2. CONVERGENCIA - EDO DE SEGUNDO ORDEN")
    print("="*80)
    g2, f2, sol2, t0_2, y0_2, yp0_2 = edo_segundo_orden()
    Y0_2 = np.array([y0_2, yp0_2])
    
    print("\nMétodo de Euler:")
    conv_euler_2 = estudio_convergencia(f2, euler, t0_2, Y0_2, 2*np.pi,
                                       tamaños_paso, sol2)
    tabla_convergencia(conv_euler_2, "Euler - Segundo Orden")
    
    print("\nMétodo de RK4:")
    conv_rk4_2 = estudio_convergencia(f2, rk4, t0_2, Y0_2, 2*np.pi,
                                     tamaños_paso, sol2)
    tabla_convergencia(conv_rk4_2, "RK4 - Segundo Orden")
    
    # 3. Sistema 2x2
    print("\n" + "="*80)
    print("3. CONVERGENCIA - SISTEMA 2×2 LINEAL")
    print("="*80)
    f3, sol3, t0_3, X0_3 = sistema_2x2_lineal()
    
    print("\nMétodo de Euler:")
    conv_euler_3 = estudio_convergencia(f3, euler, t0_3, X0_3, 2.0,
                                       tamaños_paso, sol3)
    tabla_convergencia(conv_euler_3, "Euler - Sistema 2×2")
    
    print("\nMétodo de RK4:")
    conv_rk4_3 = estudio_convergencia(f3, rk4, t0_3, X0_3, 2.0,
                                     tamaños_paso, sol3)
    tabla_convergencia(conv_rk4_3, "RK4 - Sistema 2×2")
    
    return {
        'primer_orden': {'euler': conv_euler_1, 'rk4': conv_rk4_1},
        'segundo_orden': {'euler': conv_euler_2, 'rk4': conv_rk4_2},
        'sistema_2x2': {'euler': conv_euler_3, 'rk4': conv_rk4_3}
    }


# ============================================================================
# PARTE 3: SISTEMAS NO LINEALES
# ============================================================================

def parte3_sistemas_no_lineales():
    """Resuelve los sistemas no lineales con ambos métodos"""
    imprimir_encabezado("PARTE 3: SISTEMAS NO LINEALES")
    
    h = 0.01  # Tamaño de paso
    tf = 50.0  # Tiempo final
    
    # 1. Lotka-Volterra
    print("\n" + "="*80)
    print("1. MODELO DE LOTKA-VOLTERRA (PREDADOR-PRESA)")
    print("="*80)
    
    print("\nResolviendo con Método de Euler...")
    lv_euler = resolver_sistema_no_lineal('lotka-volterra', euler, 'Euler', h, tf)
    
    print("\nResolviendo con Método de RK4...")
    lv_rk4 = resolver_sistema_no_lineal('lotka-volterra', rk4, 'RK4', h, tf)
    
    # Mostrar resultados
    print("\nRESULTADOS - Método de Euler:")
    print("-" * 70)
    print(f"Presas promedio: {lv_euler['analisis']['presas']['promedio']:.4f}")
    print(f"Presas máximo: {lv_euler['analisis']['presas']['maximo']:.4f}")
    print(f"Presas mínimo: {lv_euler['analisis']['presas']['minimo']:.4f}")
    print(f"Predadores promedio: {lv_euler['analisis']['predadores']['promedio']:.4f}")
    print(f"Predadores máximo: {lv_euler['analisis']['predadores']['maximo']:.4f}")
    print(f"Predadores mínimo: {lv_euler['analisis']['predadores']['minimo']:.4f}")
    
    print("\nRESULTADOS - Método de RK4:")
    print("-" * 70)
    print(f"Presas promedio: {lv_rk4['analisis']['presas']['promedio']:.4f}")
    print(f"Presas máximo: {lv_rk4['analisis']['presas']['maximo']:.4f}")
    print(f"Presas mínimo: {lv_rk4['analisis']['presas']['minimo']:.4f}")
    print(f"Predadores promedio: {lv_rk4['analisis']['predadores']['promedio']:.4f}")
    print(f"Predadores máximo: {lv_rk4['analisis']['predadores']['maximo']:.4f}")
    print(f"Predadores mínimo: {lv_rk4['analisis']['predadores']['minimo']:.4f}")
    
    # 2. Modelo SIR
    print("\n" + "="*80)
    print("2. MODELO SIR DE EPIDEMIAS")
    print("="*80)
    
    print("\nResolviendo con Método de Euler...")
    sir_euler = resolver_sistema_no_lineal('sir', euler, 'Euler', h, tf)
    
    print("\nResolviendo con Método de RK4...")
    sir_rk4 = resolver_sistema_no_lineal('sir', rk4, 'RK4', h, tf)
    
    # Mostrar resultados
    print("\nRESULTADOS - Método de Euler:")
    print("-" * 70)
    print(f"R₀: {sir_euler['analisis']['R0']:.2f}")
    print(f"Pico de infectados: {sir_euler['analisis']['pico_infectados']['porcentaje']:.2f}% "
          f"en t = {sir_euler['analisis']['pico_infectados']['tiempo']:.2f}")
    print(f"Total afectados al final: {sir_euler['analisis']['porcentaje_final_afectados']:.2f}%")
    
    print("\nRESULTADOS - Método de RK4:")
    print("-" * 70)
    print(f"R₀: {sir_rk4['analisis']['R0']:.2f}")
    print(f"Pico de infectados: {sir_rk4['analisis']['pico_infectados']['porcentaje']:.2f}% "
          f"en t = {sir_rk4['analisis']['pico_infectados']['tiempo']:.2f}")
    print(f"Total afectados al final: {sir_rk4['analisis']['porcentaje_final_afectados']:.2f}%")
    
    return {
        'lotka_volterra': {'euler': lv_euler, 'rk4': lv_rk4},
        'sir': {'euler': sir_euler, 'rk4': sir_rk4}
    }


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal que ejecuta todo el proyecto"""
    
    imprimir_encabezado("PROYECTO FINAL - ECUACIONES DIFERENCIALES 1")
    print("Implementación de Métodos Numéricos: Euler y Runge-Kutta 4")
    print("Segundo Ciclo 2025")
    
    try:
        # Parte 1: Validación
        print("\n>>> Ejecutando Parte 1: Validación...")
        parte1_validacion()
        
        # Parte 2: Convergencia
        print("\n>>> Ejecutando Parte 2: Estudio de Convergencia...")
        resultados_convergencia = parte2_convergencia()
        
        # Parte 3: Sistemas No Lineales
        print("\n>>> Ejecutando Parte 3: Sistemas No Lineales...")
        resultados_no_lineales = parte3_sistemas_no_lineales()
        
        # Resumen final
        imprimir_encabezado("EJECUCIÓN COMPLETADA EXITOSAMENTE ✓")
        print("Todos los cálculos han sido completados.")
        print("\nPróximos pasos:")
        print("1. Ejecutar visualizacion.py para generar gráficos")
        print("2. Compilar resultados en el informe")
        print("3. Documentar la bitácora de uso de IA")
        
        return resultados_convergencia, resultados_no_lineales
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    # Ejecutar proyecto completo
    conv, no_lin = main()
    
    print("\n" + "="*80)
    print("Para generar gráficos, ejecute: python visualizacion.py")
    print("="*80)