import tkinter as tk
from tkinter import ttk, messagebox
import math
import numpy as np
import traceback

# --- Añadido para la imagen ---
try:
    from PIL import ImageTk, Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Advertencia: Pillow no está instalado. La imagen no se mostrará.")
    print("Instala Pillow con: pip install Pillow")
# ---------------------------

# --- Añadido para gráficos ---
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# ---------------------------

# (Funciones toggle_k_manivela_entry y toggle_k_biela_entry SIN CAMBIOS)
def toggle_k_manivela_entry():
    if es_manivela_uniforme.get():
        label_k_manivela.grid_remove(); entry_k_manivela.grid_remove(); label_unidad_k_manivela.grid_remove()
    else:
        label_k_manivela.grid(row=5, column=0, sticky=tk.W, padx=(20, 5), pady=2)
        entry_k_manivela.grid(row=5, column=1, sticky="ew", padx=5, pady=2)
        label_unidad_k_manivela.grid(row=5, column=2, sticky=tk.W, padx=5, pady=2)

def toggle_k_biela_entry():
    if es_biela_uniforme.get():
        label_k_biela.grid_remove(); entry_k_biela.grid_remove(); label_unidad_k_biela.grid_remove()
    else:
        label_k_biela.grid(row=8, column=0, sticky=tk.W, padx=(20, 5), pady=2)
        entry_k_biela.grid(row=8, column=1, sticky="ew", padx=5, pady=2)
        label_unidad_k_biela.grid(row=8, column=2, sticky=tk.W, padx=5, pady=2)

# --- Variables globales para almacenar los últimos resultados ---
last_results = {
    "theta_m": None, "theta_b": None, "s": None,
    "velocidad_m": None, "velocidad_b": None, "velocidad_s": None,
    "aceleracion_m": None, "aceleracion_b": None, "aceleracion_s": None,
    # --- Añadir aquí variables para resultados dinámicos si se necesitan globalmente ---
    # "F_Ax": None, "F_Ay": None, "F_Bx": None, "F_By": None, ...
    # "Momento_entrada": None, "Fuerza_corredera": None, ...
    # -------------------------------------------------------------------------------
    "calculation_successful": False # Flag para saber si hay datos válidos
}
# ----------------------------------------------------------

#
# Función de Cálculo Principal (Cinemática, Inercias y Dinámica)
#
def calcular():
    global last_results # Indicar que usaremos la variable global

    # Inicialización y Validación
    I_manivela_pivot = None
    I_biela_CM = None
    solved_forces = None
    dynamics_results_text = "" # Inicializar texto de resultados dinámicos
    g=9.81

    # Limpiar resultados anteriores
    results_text_widget.config(state=tk.NORMAL)
    results_text_widget.delete('1.0', tk.END)
    results_text_widget.config(state=tk.DISABLED)

    # --- Resetear resultados globales al inicio de cada cálculo ---
    last_results["calculation_successful"] = False
    for key in last_results:
        if key != "calculation_successful":
            last_results[key] = None
    # ----------------------------------------------------------

    try:
        # Lectura y Validación de Parámetros
        Lm = longitud_manivela.get()
        Lb = longitud_biela.get()
        e = excentricidad.get()
        Mm = masa_manivela.get()
        Mb = masa_biela.get()
        Ms = masa_corredera.get()
        mu_k = coeficiente_friccion.get()

        if Lm <= 0 or Lb <= 0: raise ValueError("Longitudes deben ser positivas.")
        if Mm < 0 or Mb < 0 or Ms < 0: raise ValueError("Masas no pueden ser negativas.")
        if mu_k < 0: raise ValueError("Coef. de fricción no puede ser negativo.")

        #--- Cálculo de Momentos de Inercia
        if es_manivela_uniforme.get():
            I_manivela_pivot = (1.0/3.0) * Mm * Lm**2
        else:
            k_m_val = k_manivela.get()
            if k_m_val <= 0: raise ValueError("k (manivela) debe ser positivo si no es uniforme.")
            I_manivela_pivot = Mm * k_m_val**2

        if es_biela_uniforme.get():
            I_biela_CM = (1.0/12.0) * Mb * Lb**2
        else:
            k_b_val = k_biela.get()
            if k_b_val <= 0: raise ValueError("k (biela) debe ser positivo si no es uniforme.")
            I_biela_CM = Mb * k_b_val**2

        #--- Lectura Variables Cinemáticas de Entrada
        angulo_seleccionado = variable_seleccionada_ang.get()
        velocidad_seleccionada = variable_seleccionada_vel.get()
        aceleracion_seleccionada = variable_seleccionada_acel.get()
        angulo_ing_val = angulo_ingresado.get()
        velocidad_ing = velocidad_ingresada.get()
        aceleracion_ing = aceleracion_ingresada.get()

        #--- Análisis Cinemático (TU CÓDIGO ORIGINAL DEBE ESTAR AQUÍ) ---
        # --- Análisis Cinemático ---
        if angulo_seleccionado == "Ángulo de manivela":
            theta_m = math.radians(angulo_ing_val)
            arg_asin = (Lm * math.sin(theta_m) + e) / Lb if not math.isclose(Lb,0) else 2.0
            if abs(arg_asin) > 1.00001: raise ValueError(f"Config. imposible (asin fuera de rango: {arg_asin:.4f}).")
            arg_asin = max(-1.0, min(1.0, arg_asin))
            theta_b = math.asin(arg_asin)
            s = Lm * math.cos(theta_m) + Lb * math.cos(theta_b)
        elif angulo_seleccionado == "Ángulo de biela":
            theta_b = math.radians(angulo_ing_val)
            arg_asin = (Lb * math.sin(theta_b) - e) / Lm if not math.isclose(Lm,0) else 2.0
            if abs(arg_asin) > 1.00001: raise ValueError(f"Config. imposible (asin fuera de rango: {arg_asin:.4f}).")
            arg_asin = max(-1.0, min(1.0, arg_asin))
            theta_m = math.asin(arg_asin)
            s = Lm * math.cos(theta_m) + Lb * math.cos(theta_b)
        elif angulo_seleccionado == "Posición de corredera":
            s = angulo_ing_val
            if math.isclose(e, 0):
                if math.isclose(s * Lm, 0): raise ValueError("División por cero o config. inválida (s=0 o Lm=0 con e=0).")
                arg_acos = (s**2 + Lm**2 - Lb**2) / (2 * s * Lm)
                if abs(arg_acos) > 1.00001: raise ValueError("Posición de corredera inalcanzable (e=0).")
                arg_acos = max(-1.0, min(1.0, arg_acos))
                theta_m_sol1 = math.acos(arg_acos); theta_m_sol2 = -math.acos(arg_acos)
                found_solution = False
                for theta_m_test in [theta_m_sol1, theta_m_sol2]:
                    if math.isclose(Lb, 0): continue
                    arg_asin_b = (Lm * math.sin(theta_m_test)) / Lb
                    if abs(arg_asin_b) <= 1.00001:
                        arg_asin_b = max(-1.0, min(1.0, arg_asin_b)); theta_b_test = math.asin(arg_asin_b)
                        s_check = Lm * math.cos(theta_m_test) + Lb * math.cos(theta_b_test)
                        if math.isclose(s, s_check, abs_tol=1e-6):
                            theta_m, theta_b, found_solution = theta_m_test, theta_b_test, True; break
                if not found_solution: raise ValueError("No se encontró configuración válida para s (e=0).")
            else: raise NotImplementedError("Cálculo desde posición de corredera con excentricidad no implementado.")
        else: raise ValueError("Selección de variable de ángulo no válida.")
        A_vel = np.array([[-Lm*math.sin(theta_m), -Lb*math.sin(theta_b), -1],[Lm*math.cos(theta_m), -Lb*math.cos(theta_b), 0],[0,0,0]])
        B_vel = np.array([0,0,0])
        if velocidad_seleccionada == "Velocidad de manivela": A_vel[2,0], B_vel[2] = 1, velocidad_ing
        elif velocidad_seleccionada == "Velocidad de biela": A_vel[2,1], B_vel[2] = 1, velocidad_ing
        elif velocidad_seleccionada == "Velocidad de corredera": A_vel[2,2], B_vel[2] = 1, velocidad_ing
        else: raise ValueError("Selección de variable de velocidad no válida.")
        if abs(np.linalg.det(A_vel)) < 1e-9: raise np.linalg.LinAlgError("Singularidad en cálculo de velocidad.")
        velocidades = np.linalg.solve(A_vel, B_vel)
        velocidad_m, velocidad_b, velocidad_s = velocidades
        A_acel = np.copy(A_vel)
        B_acel_base = np.array([Lm*math.cos(theta_m)*velocidad_m**2 + Lb*math.cos(theta_b)*velocidad_b**2, Lm*math.sin(theta_m)*velocidad_m**2 - Lb*math.sin(theta_b)*velocidad_b**2, 0])
        if aceleracion_seleccionada == "Aceleración de manivela": A_acel[2,:], B_acel_base[2] = [1,0,0], aceleracion_ing
        elif aceleracion_seleccionada == "Aceleración de biela": A_acel[2,:], B_acel_base[2] = [0,1,0], aceleracion_ing
        elif aceleracion_seleccionada == "Aceleración de corredera": A_acel[2,:], B_acel_base[2] = [0,0,1], aceleracion_ing
        else: raise ValueError("Selección de variable de aceleración no válida.")
        if abs(np.linalg.det(A_acel)) < 1e-9: raise np.linalg.LinAlgError("Singularidad en cálculo de aceleración.")
        aceleraciones = np.linalg.solve(A_acel, B_acel_base)
        aceleracion_m, aceleracion_b, aceleracion_s = aceleraciones
        #--- FIN DE ANÁLISIS CINEMÁTICO ---

        dynamics_results_text = "\n--- Análisis Dinámico ---\n"
        try:
            a_G_mx=-(Lm/2.)*math.sin(theta_m)*aceleracion_m - (Lm/2.)*math.cos(theta_m)*velocidad_m**2
            a_G_my= (Lm/2.)*math.cos(theta_m)*aceleracion_m - (Lm/2.)*math.sin(theta_m)*velocidad_m**2
            a_Bx = -Lm*math.sin(theta_m)*aceleracion_m - Lm*math.cos(theta_m)*velocidad_m**2
            a_By =  Lm*math.cos(theta_m)*aceleracion_m - Lm*math.sin(theta_m)*velocidad_m**2
            a_G_bx = a_Bx - (Lb/2.)*math.cos(theta_b)*velocidad_b**2 - (Lb/2.)*math.sin(theta_b)*aceleracion_b
            a_G_by = a_By - (Lb/2.)*math.sin(theta_b)*velocidad_b**2 + (Lb/2.)*math.cos(theta_b)*aceleracion_b
            a_sx = aceleracion_s
            mm_amx = Mm * a_G_mx; mm_amy = Mm * a_G_my
            mb_abx = Mb * a_G_bx; mb_aby = Mb * a_G_by
            mc_sdd = Ms * a_sx
            sin_t = math.sin(theta_m); cos_t = math.cos(theta_m)
            sin_p = math.sin(theta_b); cos_p = math.cos(theta_b)
            mu_eff = mu_k if abs(velocidad_s) > 1e-6 else 0

            A = np.zeros((8, 8)); b = np.zeros(8)
            A[0,0]=1; A[0,2]=1; b[0]=mm_amx
            A[1,1]=1; A[1,3]=1; b[1]=mm_amy+Mm*g
            A[2,2]=-Lm*sin_t; A[2,3]=Lm*cos_t; b[2]=-(Lm/2)*sin_t*mm_amx+(Lm/2)*cos_t*mm_amy+(Lm/2)*cos_t*Mm*g+I_manivela_pivot*aceleracion_m
            A[3,2]=-1; A[3,4]=1; b[3]=mb_abx
            A[4,3]=-1; A[4,5]=1; b[4]=mb_aby+Mb*g
            A[5,2]=Lb*sin_p; A[5,3]=Lb*cos_p; b[5]=-(Lb/2)*sin_p*mb_abx-(Lb/2)*cos_p*mb_aby-(Lb/2)*cos_p*Mb*g+I_biela_CM*aceleracion_b
            A[6,4]=-1; A[6,6]=mu_eff; A[6,7]=1; b[6]=mc_sdd
            A[7,5]=-1; A[7,6]=1; b[7]=Ms*g

            tipo_carga = variable_seleccionada_fuerza_momento.get()
            valor_carga = valor_fuerza_momento.get()

            if tipo_carga == "Momento en manivela":
                M_input = valor_carga
                A_final = A; b_final = np.copy(b) # Usar copia para no modificar b base
                b_final[2] = b[2] - M_input
                unknown_labels = ['Ax','Ay','Bx','By','Cx','Cy','N','F (req.)']
                if abs(np.linalg.det(A_final)) < 1e-9: raise np.linalg.LinAlgError("Singularidad (Caso Momento).")
                solved_forces = np.linalg.solve(A_final, b_final)
            elif tipo_carga == "Fuerza en corredera":
                F_input = valor_carga
                A_mod = np.copy(A); b_mod = np.copy(b)
                A_mod[2, 7] = -1.0
                A_mod[6, 7] = 0.0
                b_mod[6] = b[6] - 1.0 * F_input
                unknown_labels = ['Ax','Ay','Bx','By','Cx','Cy','N','M (req.)']
                if abs(np.linalg.det(A_mod)) < 1e-9: raise np.linalg.LinAlgError("Singularidad (Caso Fuerza).")
                solved_forces = np.linalg.solve(A_mod, b_mod)

            if solved_forces is not None:
                dynamics_results_text = "\n--- Fuerzas y Par/Fuerza Requerido(a) ---\n"
                for label, value in zip(unknown_labels, solved_forces):
                    unit = "N";
                    if 'M (' in label: unit = "N·m"
                    elif 'F (' in label: unit = "N"
                    elif label == 'N': unit = "N (Normal)"
                    dynamics_results_text += f" {label}: {value:+.3f} {unit}\n"
             #   dynamics_results_text += "(Nota: Fricción calculada con μk directo, signo no ajustado a vs)\n"
            else:
                dynamics_results_text = "\n--- Análisis Dinámico no realizado ---"

          
            if not dynamics_results_text.strip(): # Si no se generó texto dinámico
                 dynamics_results_text = "\n--- Análisis Dinámico (Lógica no implementada) ---\n"
                 dynamics_results_text += "  (Inserta aquí tu código de cálculo de fuerzas/momentos)\n"


        except np.linalg.LinAlgError as dyn_error:
            dynamics_results_text = f"\n--- Error en Análisis Dinámico ---\n  Singularidad al resolver fuerzas: {dyn_error}\n"
        except Exception as dyn_e:
            dynamics_results_text = f"\n--- Error Inesperado en Dinámica ---\n  {dyn_e}\n{traceback.format_exc(limit=1)}\n"

        # --- *** FIN DE LA SECCIÓN DE DINÁMICA *** ---

        # --- Mostrar Resultados ---
        kinematics_results_text = (
            f"--- Geometría, Masas e Inercias ---\n"
            f" Lm: {Lm:.4f} m, Lb: {Lb:.4f} m, e: {e:.4f} m\n"
            f" Mm: {Mm:.3f} kg, Mb: {Mb:.3f} kg, Ms: {Ms:.3f} kg\n"
            f" I_manivela (pivote): {I_manivela_pivot:.4e} kg·m²\n"
            f" I_biela (CM): {I_biela_CM:.4e} kg·m²\n\n"
            f"--- Cinemática ---\n"
            f" θm: {math.degrees(theta_m):.2f}°, θb: {math.degrees(theta_b):.2f}°\n"
            f" s: {s:.4f} m\n"
            f" ωm: {velocidad_m:+.4f} rad/s, ωb: {velocidad_b:+.4f} rad/s, vs: {velocidad_s:+.4f} m/s\n"
            f" αm: {aceleracion_m:+.4f} rad/s², αb: {aceleracion_b:+.4f} rad/s², as: {aceleracion_s:+.4f} m/s²"
        )

        # Combinar cinemática y dinámica
        final_results_string = kinematics_results_text + "\n" + dynamics_results_text

        results_text_widget.config(state=tk.NORMAL)
        results_text_widget.insert(tk.END, final_results_string)
        results_text_widget.config(state=tk.DISABLED)

        # --- Guardar resultados para graficar ---
        last_results["theta_m"] = theta_m
        last_results["theta_b"] = theta_b
        last_results["s"] = s
        last_results["velocidad_m"] = velocidad_m
        last_results["velocidad_b"] = velocidad_b
        last_results["velocidad_s"] = velocidad_s
        last_results["aceleracion_m"] = aceleracion_m
        last_results["aceleracion_b"] = aceleracion_b
        last_results["aceleracion_s"] = aceleracion_s
        # --- Añadir aquí el guardado de resultados dinámicos si los necesitas en otro lugar ---
        # last_results["F_Ax"] = F_Ax
        # ... etc ...
        # ----------------------------------------------------------------------------------
        last_results["calculation_successful"] = True


    except (ValueError, np.linalg.LinAlgError, NotImplementedError) as e:
        results_text_widget.config(state=tk.NORMAL)
        results_text_widget.delete('1.0', tk.END)
        results_text_widget.insert(tk.END, f"\n--- ERROR DE CÁLCULO ---\n{type(e).__name__}: {e}")
        results_text_widget.config(state=tk.DISABLED)
        last_results["calculation_successful"] = False
        print(f"Error Detallado: {traceback.format_exc(limit=2)}") # Más detalle en consola
    except Exception as e:
        results_text_widget.config(state=tk.NORMAL)
        results_text_widget.delete('1.0', tk.END)
        results_text_widget.insert(tk.END, f"\n--- ERROR INESPERADO ---\n{e}\n{traceback.format_exc()}")
        results_text_widget.config(state=tk.DISABLED)
        last_results["calculation_successful"] = False

# --- Funciones de Gráficos (SIN CAMBIOS RESPECTO A LA VERSIÓN ANTERIOR) ---
plot_window = None
plot_canvas = None
fig = None
var_elemento_plot = None # Definir globalmente
var_tipo_movimiento_plot = None # Definir globalmente

def crear_ventana_graficos():
    global plot_window, plot_canvas, fig, var_elemento_plot, var_tipo_movimiento_plot

    if plot_window is not None and plot_window.winfo_exists():
        plot_window.lift()
        return

    if not last_results["calculation_successful"]:
        messagebox.showwarning("Sin Datos", "Realiza un cálculo exitoso antes de generar gráficos.")
        return

    plot_window = tk.Toplevel(root)
    plot_window.title("Gráficos Cinemáticos")
    plot_window.geometry("500x700")

    control_frame = ttk.Frame(plot_window, padding="10")
    control_frame.pack(side=tk.TOP, fill=tk.X)

    ttk.Label(control_frame, text="Elemento:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    var_elemento_plot = tk.StringVar(value="Manivela") # Ahora global
    combo_elemento = ttk.Combobox(control_frame, textvariable=var_elemento_plot,
                                  values=["Manivela", "Biela", "Corredera"], state="readonly")
    combo_elemento.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

    ttk.Label(control_frame, text="Simular como:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
    var_tipo_movimiento_plot = tk.StringVar(value="Velocidad Constante") # Ahora global
    combo_tipo_mov = ttk.Combobox(control_frame, textvariable=var_tipo_movimiento_plot,
                                   values=["Velocidad Constante", "Aceleración Constante"], state="readonly")
    combo_tipo_mov.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)

    update_button = ttk.Button(control_frame, text="Actualizar Gráficos", command=actualizar_graficos)
    update_button.grid(row=2, column=0, columnspan=2, pady=10)

    control_frame.columnconfigure(1, weight=1)

    plot_frame = ttk.Frame(plot_window)
    plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    # Crear figura y ejes si no existen
    if fig is None:
        fig, _ = plt.subplots(3, 1, figsize=(5, 6), sharex=True)

    # Crear o reusar el canvas
    if plot_canvas is None or not plot_canvas.get_tk_widget().winfo_exists():
        plot_canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        plot_canvas_widget = plot_canvas.get_tk_widget()
        plot_canvas_widget.pack(fill=tk.BOTH, expand=True)
    else:
        # Si el canvas existe pero la ventana se recreó, re-empacar
        plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


    actualizar_graficos() # Dibujar gráficos iniciales

def actualizar_graficos():
    global fig # Necesitamos modificar la figura global

    if plot_window is None or not plot_window.winfo_exists() or fig is None or plot_canvas is None:
        print("Ventana de gráficos no lista.")
        return # No hacer nada si la ventana o la figura no existen

    if not last_results["calculation_successful"]:
        for ax in fig.get_axes(): ax.cla()
        fig.suptitle("Realiza un cálculo para ver los gráficos", fontsize=10)
        try:
            plot_canvas.draw()
        except Exception as draw_e:
            print(f"Error al limpiar gráficos: {draw_e}")
        return

    elemento = var_elemento_plot.get()
    tipo_mov = var_tipo_movimiento_plot.get()

    t = np.linspace(0, 2, 200) # Simular por 2 segundos
    pos_data, vel_data, acc_data = None, None, None
    pos_label, vel_label, acc_label = "", "", ""
    eje_y_pos, eje_y_vel, eje_y_acc = "", "", ""
    titulo_elemento = ""

    # Chequeo extra por si acaso los resultados no están aunque el flag sea True
    required_keys = []
    if elemento == "Manivela": required_keys = ["velocidad_m", "aceleracion_m"]
    elif elemento == "Biela": required_keys = ["velocidad_b", "aceleracion_b"]
    else: required_keys = ["velocidad_s", "aceleracion_s"]

    for key in required_keys:
        if last_results.get(key) is None:
             messagebox.showerror("Error de Datos", f"Falta el resultado calculado para '{key}'. No se puede graficar.")
             # Limpiar gráficos como en el caso de no cálculo
             for ax in fig.get_axes(): ax.cla()
             fig.suptitle("Error: Faltan datos calculados", fontsize=10, color='red')
             try:
                 plot_canvas.draw()
             except Exception as draw_e:
                 print(f"Error al limpiar gráficos tras error de datos: {draw_e}")
             return


    if elemento == "Manivela":
        v_base = last_results["velocidad_m"]
        a_base = last_results["aceleracion_m"]
        eje_y_pos, eje_y_vel, eje_y_acc = 'Ángulo (rad)', 'Vel. Angular (rad/s)', 'Acel. Angular (rad/s²)'
        titulo_elemento = "Manivela"
    elif elemento == "Biela":
        v_base = last_results["velocidad_b"]
        a_base = last_results["aceleracion_b"]
        eje_y_pos, eje_y_vel, eje_y_acc = 'Ángulo (rad)', 'Vel. Angular (rad/s)', 'Acel. Angular (rad/s²)'
        titulo_elemento = "Biela"
    else: # Corredera
        v_base = last_results["velocidad_s"]
        a_base = last_results["aceleracion_s"]
        eje_y_pos, eje_y_vel, eje_y_acc = 'Posición (m)', 'Velocidad (m/s)', 'Aceleración (m/s²)'
        titulo_elemento = "Corredera"

    if tipo_mov == "Velocidad Constante":
        vel_const = v_base if v_base is not None else 0
        pos_data = vel_const * t
        vel_data = np.full_like(t, vel_const)
        acc_data = np.zeros_like(t)
        titulo_mov = "Velocidad Constante"
    else: # Aceleración Constante
        acc_const = a_base if a_base is not None else 0
        # v_inicial = v_base if v_base is not None else 0 # Podríamos usar la vel inicial también
        pos_data = 0.5 * acc_const * t**2 # + v_inicial * t (Opcional)
        vel_data = acc_const * t          # + v_inicial     (Opcional)
        acc_data = np.full_like(t, acc_const)
        titulo_mov = "Aceleración Constante"

    axs = fig.get_axes()
    if len(axs) != 3:
        print("Error: No se encontraron los 3 ejes esperados.")
        # Intentar recrear la figura si se perdió
        plt.close(fig) # Cerrar figura potencialmente corrupta
        fig, axs = plt.subplots(3, 1, figsize=(5, 6), sharex=True)
        # Volver a asociar con el canvas (esto puede ser complicado si la ventana aún existe)
        # Es mejor prevenir que los ejes se pierdan.
        # Por ahora, solo informamos.
        messagebox.showerror("Error Interno", "Se perdió la referencia a los ejes del gráfico.")
        return


    for ax in axs: ax.cla() # Limpiar ejes

    axs[0].plot(t, pos_data, label=f'Posición ({titulo_elemento})')
    axs[0].set_ylabel(eje_y_pos)
    axs[0].set_title(f'Simulación: {titulo_elemento} ({titulo_mov})')
    axs[0].grid(True); axs[0].legend(loc='best')

    axs[1].plot(t, vel_data, label=f'Velocidad ({titulo_elemento})', color='orange')
    axs[1].set_ylabel(eje_y_vel)
    axs[1].grid(True); axs[1].legend(loc='best')

    axs[2].plot(t, acc_data, label=f'Aceleración ({titulo_elemento})', color='green')
    axs[2].set_ylabel(eje_y_acc)
    axs[2].set_xlabel('Tiempo (s)')
    axs[2].grid(True); axs[2].legend(loc='best')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    try:
        plot_canvas.draw()
    except Exception as draw_e:
        print(f"Error al dibujar canvas: {draw_e}")
        messagebox.showerror("Error de Gráfico", f"No se pudo dibujar el gráfico: {draw_e}")

#
# Interfaz Gráfica (Tkinter)
#
root = tk.Tk()
root.title("Análisis Cinemático/Dinámico de Biela-Manivela")
# Ajustar tamaño si es necesario, considerar espacio para imagen si es grande
root.geometry("800x900") # Un poco más ancho y alto

# --- Variables Tkinter ---
# (SIN CAMBIOS, igual que antes)
longitud_manivela = tk.DoubleVar(value=0.5)
longitud_biela = tk.DoubleVar(value=0.75)
excentricidad = tk.DoubleVar(value=0.0) # Poner 0.0 si la imagen es sin excentricidad
masa_manivela = tk.DoubleVar(value=1.0)
masa_biela = tk.DoubleVar(value=2.0)
masa_corredera = tk.DoubleVar(value=5.0)
es_manivela_uniforme = tk.BooleanVar(value=True)
es_biela_uniforme = tk.BooleanVar(value=True)
k_manivela = tk.DoubleVar(value=0.0)
k_biela = tk.DoubleVar(value=0.0)
variable_seleccionada_ang = tk.StringVar(value="Ángulo de manivela")
angulo_ingresado = tk.DoubleVar(value=120)
variable_seleccionada_vel = tk.StringVar(value="Velocidad de manivela")
velocidad_ingresada = tk.DoubleVar(value=-20)
variable_seleccionada_acel = tk.StringVar(value="Aceleración de manivela")
aceleracion_ingresada = tk.DoubleVar(value=500)
variable_seleccionada_fuerza_momento = tk.StringVar(value="Momento en manivela") # Opciones: "Momento en manivela", "Fuerza en corredera"
valor_fuerza_momento = tk.DoubleVar(value=0.0) # Valor de la fuerza o momento CONOCIDO para calcular el otro
coeficiente_friccion = tk.DoubleVar(value=0.1)

# --- Estilo ---
style = ttk.Style()
style.configure("TLabel", padding=3)
style.configure("TEntry", padding=3)
style.configure("TCombobox", padding=3)
style.configure("TButton", padding=5)
style.configure("TCheckbutton", padding=(10, 2))
style.configure("TLabelframe.Label", font=("Helvetica", 10, "bold"))

# --- Frame principal ---
main_frame = ttk.Frame(root, padding="10")
main_frame.grid(row=0, column=0, sticky="nsew")
root.columnconfigure(0, weight=1) # Permitir que main_frame se expanda horizontalmente
root.rowconfigure(0, weight=1) # Permitir que main_frame se expanda verticalmente

# Configurar columnas del main_frame (ej. una para imagen, otra para controles)
main_frame.columnconfigure(0, weight=0) # Columna para imagen (ancho fijo)
main_frame.columnconfigure(1, weight=1) # Columna para controles (expandible)
# Configurar filas del main_frame para expansión
main_frame.rowconfigure(4, weight=1) # Fila para resultados (expandible)

# --- *** RESTAURADO: Cargar y Mostrar Imagen *** ---
image_label = None
img_object = None # Guardar referencia
image_frame = ttk.Frame(main_frame) # Frame dedicado para la imagen
image_frame.grid(row=0, column=0, rowspan=4, padx=10, pady=10, sticky="nw") # Colocar a la izquierda

if PIL_AVAILABLE:
    try:
        # --- CAMBIA ESTE NOMBRE DE ARCHIVO ---
        image_path = "/home/fed/Documentos/7semestre/Mecanismos y Vibraciones/bm_final/bm.png"
        # ------------------------------------
        img = Image.open(image_path)
        # Redimensionar si es necesario (opcional)
        base_width = 150
        w_percent = (base_width / float(img.size[0]))
        h_size = int((float(img.size[1]) * float(w_percent)))
        img = img.resize((base_width, h_size), Image.Resampling.LANCZOS)

        img_object = ImageTk.PhotoImage(img)
        image_label = ttk.Label(image_frame, image=img_object)
        image_label.pack(padx=5, pady=5)
    except FileNotFoundError:
        image_label = ttk.Label(image_frame, text=f"Imagen '{image_path}' no encontrada.")
        image_label.pack(padx=5, pady=5)
    except Exception as img_e:
        image_label = ttk.Label(image_frame, text=f"Error al cargar imagen:\n{img_e}")
        image_label.pack(padx=5, pady=5)
        print(f"Error detallado al cargar imagen: {traceback.format_exc()}")
else:
    image_label = ttk.Label(image_frame, text="Pillow no instalado.\nNo se puede mostrar imagen.")
    image_label.pack(padx=5, pady=5)
# --- Fin Sección Imagen ---

# --- Contenedor para los frames de parámetros y entradas (a la derecha de la imagen) ---
controls_container = ttk.Frame(main_frame)
controls_container.grid(row=0, column=1, rowspan=4, sticky="nsew", padx=5)
controls_container.columnconfigure(0, weight=1) # Permitir que los frames internos se expandan

# --- Sección de Parámetros Geométricos y de Masa ---
params_frame = ttk.LabelFrame(controls_container, text="Parámetros", padding="10")
params_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
params_frame.columnconfigure(1, weight=1) # Columna de entradas expandible

# (Widgets de parámetros - SIN CAMBIOS INTERNOS, solo su frame padre)
ttk.Label(params_frame, text="Longitud Manivela (Lm):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
ttk.Entry(params_frame, textvariable=longitud_manivela, width=10).grid(row=0, column=1, sticky="ew", padx=5, pady=2)
ttk.Label(params_frame, text="m").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
# ... (resto de widgets de Lb, e, Mm, Mb, Ms, k, mu_k) ...
ttk.Label(params_frame, text="Longitud Biela (Lb):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
ttk.Entry(params_frame, textvariable=longitud_biela, width=10).grid(row=1, column=1, sticky="ew", padx=5, pady=2)
ttk.Label(params_frame, text="m").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)

ttk.Label(params_frame, text="Excentricidad (e):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
ttk.Entry(params_frame, textvariable=excentricidad, width=10).grid(row=2, column=1, sticky="ew", padx=5, pady=2)
ttk.Label(params_frame, text="m").grid(row=2, column=2, sticky=tk.W, padx=5, pady=2)

ttk.Label(params_frame, text="Masa Manivela (Mm):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
ttk.Entry(params_frame, textvariable=masa_manivela, width=10).grid(row=3, column=1, sticky="ew", padx=5, pady=2)
ttk.Label(params_frame, text="kg").grid(row=3, column=2, sticky=tk.W, padx=5, pady=2)

ttk.Checkbutton(params_frame, text="Barra Uniforme", variable=es_manivela_uniforme, command=toggle_k_manivela_entry).grid(row=4, column=0, columnspan=3, sticky=tk.W, padx=15, pady=2)
label_k_manivela = ttk.Label(params_frame, text="Radio de Giro k (pivote):")
entry_k_manivela = ttk.Entry(params_frame, textvariable=k_manivela, width=10) # State manejado por toggle
label_unidad_k_manivela = ttk.Label(params_frame, text="m")
# toggle_k_manivela_entry() los colocará/ocultará y habilitará/deshabilitará

ttk.Label(params_frame, text="Masa Biela (Mb):").grid(row=6, column=0, sticky=tk.W, padx=5, pady=2)
ttk.Entry(params_frame, textvariable=masa_biela, width=10).grid(row=6, column=1, sticky="ew", padx=5, pady=2)
ttk.Label(params_frame, text="kg").grid(row=6, column=2, sticky=tk.W, padx=5, pady=2)

ttk.Checkbutton(params_frame, text="Barra Uniforme", variable=es_biela_uniforme, command=toggle_k_biela_entry).grid(row=7, column=0, columnspan=3, sticky=tk.W, padx=15, pady=2)
label_k_biela = ttk.Label(params_frame, text="Radio de Giro k (CM):")
entry_k_biela = ttk.Entry(params_frame, textvariable=k_biela, width=10) # State manejado por toggle
label_unidad_k_biela = ttk.Label(params_frame, text="m")
# toggle_k_biela_entry() los colocará/ocultará y habilitará/deshabilitará

ttk.Label(params_frame, text="Masa Corredera (Ms):").grid(row=9, column=0, sticky=tk.W, padx=5, pady=2)
ttk.Entry(params_frame, textvariable=masa_corredera, width=10).grid(row=9, column=1, sticky="ew", padx=5, pady=2)
ttk.Label(params_frame, text="kg").grid(row=9, column=2, sticky=tk.W, padx=5, pady=2)

ttk.Label(params_frame, text="Coef. Fricción Cinética (μk):").grid(row=10, column=0, sticky=tk.W, padx=5, pady=2)
ttk.Entry(params_frame, textvariable=coeficiente_friccion, width=10).grid(row=10, column=1, sticky="ew", padx=5, pady=2)


# --- Sección de Entradas Cinemáticas ---
kin_frame = ttk.LabelFrame(controls_container, text="Entrada Cinemática", padding="10")
kin_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
kin_frame.columnconfigure(1, weight=1)

# (Widgets de entradas cinemáticas - SIN CAMBIOS INTERNOS)
ttk.Label(kin_frame, text="Variable de Entrada:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
combo_ang = ttk.Combobox(kin_frame, textvariable=variable_seleccionada_ang, values=["Ángulo de manivela", "Ángulo de biela", "Posición de corredera"], state="readonly")
combo_ang.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
ttk.Entry(kin_frame, textvariable=angulo_ingresado, width=10).grid(row=0, column=2, padx=5, pady=2)
ttk.Label(kin_frame, text="° / m").grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)

ttk.Label(kin_frame, text="Velocidad Conocida:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2) # Cambiado texto
combo_vel = ttk.Combobox(kin_frame, textvariable=variable_seleccionada_vel, values=["Velocidad de manivela", "Velocidad de biela", "Velocidad de corredera"], state="readonly")
combo_vel.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
ttk.Entry(kin_frame, textvariable=velocidad_ingresada, width=10).grid(row=1, column=2, padx=5, pady=2)
ttk.Label(kin_frame, text="rad/s / m/s").grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)

ttk.Label(kin_frame, text="Aceleración Conocida:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2) # Cambiado texto
combo_acel = ttk.Combobox(kin_frame, textvariable=variable_seleccionada_acel, values=["Aceleración de manivela", "Aceleración de biela", "Aceleración de corredera"], state="readonly")
combo_acel.grid(row=2, column=1, sticky="ew", padx=5, pady=2)
ttk.Entry(kin_frame, textvariable=aceleracion_ingresada, width=10).grid(row=2, column=2, padx=5, pady=2)
ttk.Label(kin_frame, text="rad/s² / m/s²").grid(row=2, column=3, sticky=tk.W, padx=5, pady=2)


# --- Sección de Carga Externa (para Dinámica) ---
force_frame = ttk.LabelFrame(controls_container, text="Entrada Dinámica (Conocida)", padding="10")
force_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
force_frame.columnconfigure(1, weight=1)

# (Widgets de carga externa - SIN CAMBIOS INTERNOS)
ttk.Label(force_frame, text="Variable Conocida:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2) # Texto ajustado
combo_force = ttk.Combobox(force_frame, textvariable=variable_seleccionada_fuerza_momento, values=["Momento en manivela", "Fuerza en corredera"], state="readonly")
combo_force.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
ttk.Entry(force_frame, textvariable=valor_fuerza_momento, width=10).grid(row=0, column=2, padx=5, pady=2)
ttk.Label(force_frame, text="N·m / N").grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)


# --- Botones ---
button_frame = ttk.Frame(controls_container) # Frame para botones debajo de las entradas
button_frame.grid(row=3, column=0, pady=10, sticky="ew")
button_frame.columnconfigure(0, weight=1) # Centrar botones o distribuirlos
button_frame.columnconfigure(1, weight=1)

calc_button = ttk.Button(button_frame, text="Calcular", command=calcular)
calc_button.grid(row=0, column=0, padx=10, pady=5) # sticky='e' para alinear a la derecha si se quiere

plot_button = ttk.Button(button_frame, text="Mostrar Gráficos", command=crear_ventana_graficos)
plot_button.grid(row=0, column=1, padx=10, pady=5) # sticky='w' para alinear a la izquierda si se quiere


# --- Área de Resultados ---
results_frame = ttk.LabelFrame(main_frame, text="Resultados", padding="10")
# Colocar debajo de la imagen y los controles
results_frame.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
results_frame.rowconfigure(0, weight=1) # Permitir que el Text se expanda verticalmente
results_frame.columnconfigure(0, weight=1) # Permitir que el Text se expanda horizontalmente

results_text_widget = tk.Text(results_frame, height=20, width=90, wrap=tk.WORD, state=tk.DISABLED, font=("Courier New", 9)) # Aumentada altura inicial
results_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=results_text_widget.yview)
results_text_widget.config(yscrollcommand=results_scrollbar.set)

results_text_widget.grid(row=0, column=0, sticky="nsew")
results_scrollbar.grid(row=0, column=1, sticky="ns")

# Llamar a las funciones toggle inicialmente para establecer el estado correcto de k
toggle_k_manivela_entry()
toggle_k_biela_entry()

# Ejecutar la aplicación
root.mainloop()