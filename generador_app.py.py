import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
# Importación de distribuciones y tests estadísticos
from scipy.stats import expon, kstest, chisquare, beta as beta_dist, gamma as gamma_dist, binom, poisson
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

# ===========================
# FUNCIONES DE GENERACIÓN
# Implementación de métodos de simulación.
# ===========================
# Discretas
def generar_bernoulli(p, n):
    # Método de Transformada Inversa: U < p -> 1, U >= p -> 0
    U = np.random.rand(n)
    return (U < p).astype(int)

def generar_binomial(n_val, p, n):
    # Método de Convolución: Suma de n_val variables de Bernoulli (más lento, pero implementa el método)
    return np.sum(np.random.rand(n, n_val) < p, axis=1)

def generar_poisson(lam, n):
    # Algoritmo de Aceptación/Rechazo o optimizado (aproximación rápida para lambda bajo)
    L = np.exp(-lam)
    k = np.zeros(n)
    for i in range(n):
        p = 1.0
        k_i = 0
        while p > L:
            p *= np.random.rand()
            k_i += 1
        k[i] = k_i -1
    return k

# Continuas
def generar_exponencial(lam, n):
    # Método de Transformada Inversa
    U = np.random.rand(n)
    return -np.log(1-U)/lam

def generar_erlang(k, lam, n):
    # Método de Convolución: Suma de k variables Exponenciales
    return np.sum(-np.log(1 - np.random.rand(n, int(k)))/lam, axis=1)

def generar_gamma(k, lam, n):
    # El método de convolución implementado solo funciona para 'k' entero (distribución Erlang)
    k_val = float(k)
    if not k_val.is_integer():
        raise ValueError("Solo k entero para método de convolución (Erlang).")
    return generar_erlang(int(k_val), lam, n)

def generar_beta(alpha, beta, n):
    # Método de Composición (a partir de dos variables Gamma/Erlang)
    alpha_val = float(alpha)
    beta_val = float(beta)
    if not alpha_val.is_integer() or not beta_val.is_integer():
        raise ValueError("Solo α y β enteros para método de composición (requiere Gamma/Erlang).")
    Y1 = generar_gamma(int(alpha_val), 1, n)
    Y2 = generar_gamma(int(beta_val), 1, n)
    return Y1 / (Y1 + Y2)

# ===========================
# PRUEBAS ESTADÍSTICAS
# Implementación de bondad de ajuste (KS para continuas, Chi^2 para discretas).
# ===========================
def prueba_ks_continua(datos, dist_name, params):
    # Prueba de Kolmogórov-Smirnov para distribuciones continuas.
    if dist_name == "exponencial":
        stat, p_val = kstest(datos, 'expon', args=(0, 1/params['λ']))
        desc = "Si p-value>0.05, los datos siguen la distribución exponencial."
    elif dist_name == "beta":
        stat, p_val = kstest(datos, 'beta', args=(params['α'], params['β']))
        desc = "Si p-value>0.05, los datos siguen la distribución beta."
    elif dist_name in ["gamma","erlang"]:
        # Gamma/Erlang: scipy usa forma k y escala 1/lambda
        stat, p_val = kstest(datos, 'gamma', args=(params['k'],0,1/params['λ']))
        desc = "Si p-value>0.05, los datos siguen la distribución gamma/erlang."
    return stat, p_val, desc

def prueba_chi_discreta(datos, dist_name, params):
    # Prueba Chi-cuadrado para distribuciones discretas.
    valores_observados, frec_observada = np.unique(datos, return_counts=True)
    n = len(datos)

    if dist_name == "bernoulli":
        p = params['p']
        valores_teoricos = np.array([0,1])
        frec_esperada = n * np.array([1-p,p])
        descripcion = "Si p-value>0.05, los datos siguen la distribución Bernoulli."
    elif dist_name == "binomial":
        n_val = int(params['n'])
        p = params['p']
        min_obs, max_obs = int(np.min(datos)), int(np.max(datos))
        valores_teoricos = np.arange(min_obs, max_obs + 1)
        frec_esperada = np.array([binom.pmf(k, n_val, p)*n for k in valores_teoricos])
        descripcion = "Si p-value>0.05, los datos siguen la distribución Binomial."
    elif dist_name == "poisson":
        lam = params['λ']
        min_obs, max_obs = int(np.min(datos)), int(np.max(datos))
        valores_teoricos = np.arange(min_obs, max_obs + 1)
        frec_esperada = np.array([poisson.pmf(k, lam)*n for k in valores_teoricos])
        descripcion = "Si p-value>0.05, los datos siguen la distribución Poisson."

    frec_observada_full = np.zeros(len(valores_teoricos))
    # Mapeo de frecuencias observadas a los bins teóricos
    for i, val in enumerate(valores_teoricos):
        if val in valores_observados:
            frec_observada_full[i] = frec_observada[np.where(valores_observados==val)[0][0]]

    # Normalización de frecuencias esperadas (ajuste menor por posibles errores de precisión flotante)
    if dist_name in ["binomial", "poisson"]:
        if np.sum(frec_esperada) > 0:
            factor_ajuste = n / np.sum(frec_esperada)
            frec_esperada = frec_esperada * factor_ajuste
        else:
            messagebox.showwarning("Advertencia", "No se pueden calcular frecuencias esperadas significativas para la prueba Chi-cuadrado.")
            return 0, 1.0, valores_teoricos, frec_esperada, descripcion

    # Evita divisiones por cero en el cálculo del estadístico Chi-cuadrado.
    frec_esperada[frec_esperada == 0] = 1e-9

    if len(frec_observada_full) <= 1:
        messagebox.showwarning("Advertencia", "Demasiados pocos bins para la prueba Chi-cuadrado. Intente con más muestras o parámetros diferentes.")
        return 0, 1.0, valores_teoricos, frec_esperada, descripcion

    stat, p_val = chisquare(f_obs=frec_observada_full, f_exp=frec_esperada)
    return stat, p_val, valores_teoricos, frec_esperada, descripcion

# ===========================
# CLASE PRINCIPAL DE LA APLICACIÓN
# ===========================

class GeneradorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Generador de variables aleatorias")

        # Intenta maximizar la ventana en diferentes entornos
        try:
            self.state('zoomed')
        except tk.TclError:
            self.attributes('-fullscreen', False)
            self.state('iconic')
            self.state('normal')
            self.wm_attributes('-zoomed', True)

        # Contenedor principal para la gestión de Frames
        container = tk.Frame(self)
        container.pack(fill="both", expand=True)
        self.frames = {}

        # Definición de todos los Frames de la aplicación
        for F in (InicioFrame, DiscretaFrame, ContinuaFrame,
                  BernoulliFrame, BinomialFrame, PoissonFrame,
                  ExponencialFrame, ErlangFrame, GammaFrame, BetaFrame):
            frame = F(container, self)
            self.frames[F] = frame
            # Todos los frames se apilan en la misma celda (0,0)
            frame.grid(row=0, column=0, sticky="nsew")

        self.mostrar_frame(InicioFrame)

    def mostrar_frame(self, cont):
        # Función utilitaria para cambiar de vista (frame)
        frame = self.frames[cont]
        frame.tkraise()

# --------------------------------------------------------------------------------

# ===========================
# FRAMES DE NAVEGACIÓN
# ===========================

class InicioFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        # Configuración del grid para centrado vertical y horizontal
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0) # Fila del contenido
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Frame auxiliar para contener el contenido y aplicar el centrado
        center_frame = tk.Frame(self)
        center_frame.grid(row=1, column=0, sticky="")

        # Lógica de carga y redimensionamiento de imagen
        try:
            image_path = "box.png"
            pil_img = Image.open(image_path)
            pil_img = pil_img.resize((300, 200), Image.Resampling.LANCZOS)
            self.tk_img = ImageTk.PhotoImage(pil_img)

            image_label = tk.Label(center_frame, image=self.tk_img)
            image_label.pack(pady=10)

        except FileNotFoundError:
            tk.Label(center_frame, text="[ERROR: Imagen 'box.png' no encontrada]", fg="red", font=("Arial", 16)).pack(pady=10)
            self.tk_img = None
        except Exception as e:
            tk.Label(center_frame, text=f"[ERROR al cargar imagen: {e}]", fg="red", font=("Arial", 16)).pack(pady=10)
            self.tk_img = None

        # Botones de navegación
        tk.Label(center_frame, text="Generador de variables aleatorias", font=("Arial", 24)).pack(pady=20)
        tk.Label(center_frame, text="Seleccione su tipo de distribución", font=("Arial", 16)).pack(pady=10)
        tk.Button(center_frame, text="Discreta", width=20, command=lambda: controller.mostrar_frame(DiscretaFrame)).pack(pady=5)
        tk.Button(center_frame, text="Continua", width=20, command=lambda: controller.mostrar_frame(ContinuaFrame)).pack(pady=5)


class DiscretaFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        # Configuración del grid para centrado
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        center_frame = tk.Frame(self)
        center_frame.grid(row=1, column=0, sticky="")

        tk.Label(center_frame, text="Distribuciones discretas", font=("Arial", 20)).pack(pady=20)
        # Botones de selección de distribuciones discretas
        tk.Button(center_frame, text="Bernoulli", width=20, command=lambda: controller.mostrar_frame(BernoulliFrame)).pack(pady=5)
        tk.Button(center_frame, text="Binomial", width=20, command=lambda: controller.mostrar_frame(BinomialFrame)).pack(pady=5)
        tk.Button(center_frame, text="Poisson", width=20, command=lambda: controller.mostrar_frame(PoissonFrame)).pack(pady=5)
        tk.Button(center_frame, text="Atrás", width=20, command=lambda: controller.mostrar_frame(InicioFrame)).pack(pady=20)

class ContinuaFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        # Configuración del grid para centrado
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        center_frame = tk.Frame(self)
        center_frame.grid(row=1, column=0, sticky="")

        tk.Label(center_frame, text="Distribuciones continuas", font=("Arial", 20)).pack(pady=20)
        # Botones de selección de distribuciones continuas
        tk.Button(center_frame, text="Exponencial", width=20, command=lambda: controller.mostrar_frame(ExponencialFrame)).pack(pady=5)
        tk.Button(center_frame, text="Erlang", width=20, command=lambda: controller.mostrar_frame(ErlangFrame)).pack(pady=5)
        tk.Button(center_frame, text="Gamma", width=20, command=lambda: controller.mostrar_frame(GammaFrame)).pack(pady=5)
        tk.Button(center_frame, text="Beta", width=20, command=lambda: controller.mostrar_frame(BetaFrame)).pack(pady=5)
        tk.Button(center_frame, text="Atrás", width=20, command=lambda: controller.mostrar_frame(InicioFrame)).pack(pady=20)

# --------------------------------------------------------------------------------

# ===========================
# PANEL GENÉRICO DE DISTRIBUCIÓN
# Clase base para todos los frames de generación de variables.
# ===========================

class DistribucionFrame(tk.Frame):
    def crear_panel(self, controller, nombre, params_list, generar_func, tipo):

        # Contenedor para manejar la expansión y centrado del contenido principal
        center_container = tk.Frame(self)
        center_container.pack(expand=True, padx=20, pady=20)

        # Frame para los inputs (arriba, fijo)
        config_frame = tk.Frame(center_container)
        config_frame.pack(pady=5)

        # Frame para la gráfica y resultados (abajo, expandible)
        plot_frame = tk.Frame(center_container)
        plot_frame.pack(expand=True, fill="both")

        # 1. Configuración de parámetros
        tk.Label(config_frame, text=f"Distribución {nombre}", font=("Arial", 20)).pack(pady=5)
        # Determinación del método de generación para fines informativos
        metodo = 'Transformada inversa' if nombre in ['Bernoulli','Poisson','Exponencial'] else 'Convolución' if nombre in ['Binomial','Erlang','Gamma'] else 'Composición'
        tk.Label(config_frame, text=f"Método: {metodo}", font=("Arial", 14)).pack(pady=2)

        self.entries = {}
        param_frame = tk.Frame(config_frame)
        param_frame.pack(pady=5)
        row = 0
        # Creación dinámica de campos de entrada para cada parámetro
        for p in params_list:
            help_text = self.get_help_text(nombre, p)
            tk.Label(param_frame, text=f"{p}:").grid(row=row, column=0, padx=5, pady=2, sticky="e")
            entry = tk.Entry(param_frame)
            entry.grid(row=row, column=1, padx=5, pady=2)
            self.entries[p] = entry

            tk.Label(param_frame, text=help_text, fg="blue").grid(row=row, column=2, padx=10, pady=2, sticky="w")
            row +=1

        # Campo para el número de muestras
        tk.Label(param_frame, text="Número de muestras:").grid(row=row, column=0, padx=5, pady=2, sticky="e")
        self.n_entry = tk.Entry(param_frame)
        self.n_entry.grid(row=row, column=1, padx=5, pady=2)
        tk.Label(param_frame, text="Entero positivo (> 100)").grid(row=row, column=2, padx=10, pady=2, sticky="w")

        tk.Button(config_frame, text="Generar", command=lambda: self.generar_datos(generar_func, nombre, tipo)).pack(pady=5)

        # 2. Gráfica y Resultados
        self.resultado_label = tk.Label(plot_frame, text="", font=("Arial", 10))
        self.resultado_label.pack(pady=2)

        # Inicialización del objeto Matplotlib Figure y el Canvas de Tkinter
        self.figura = plt.Figure(figsize=(7, 4))
        self.canvas = FigureCanvasTkAgg(self.figura, plot_frame)
        self.canvas.get_tk_widget().pack(pady=5, expand=True, fill="both")

        # Botón para volver al menú anterior (Discreta o Continua)
        back_frame = ContinuaFrame if tipo=="continua" else DiscretaFrame
        tk.Button(config_frame, text="Atrás", command=lambda: controller.mostrar_frame(back_frame)).pack(pady=5)

    def get_help_text(self, nombre, param):
        # Texto de ayuda conciso para los parámetros de la distribución
        if nombre == "Bernoulli" and param == "p": return "Probabilidad de éxito (0 < p < 1)"
        if nombre == "Binomial" and param == "n": return "Número de pruebas (Entero positivo)"
        if nombre == "Binomial" and param == "p": return "Probabilidad de éxito (0 < p < 1)"
        if nombre == "Poisson" and param == "λ": return "Tasa promedio (λ > 0)"
        if nombre == "Exponencial" and param == "λ": return "Tasa promedio (λ > 0)"
        if nombre == "Erlang" and param == "k": return "Parámetro de forma (Entero positivo)"
        if nombre == "Erlang" and param == "λ": return "Parámetro de tasa (λ > 0)"
        if nombre == "Gamma" and param == "k": return "Parámetro de forma (Entero positivo)"
        if nombre == "Gamma" and param == "λ": return "Parámetro de tasa (λ > 0)"
        if nombre == "Beta" and param == "α": return "Parámetro α (Entero positivo)"
        if nombre == "Beta" and param == "β": return "Parámetro β (Entero positivo)"
        return ""


    def validar_parametros(self, nombre, params):
        # Validación de rangos y tipos de datos para los parámetros estocásticos
        if nombre == "Bernoulli" or nombre == "Binomial":
            p = params.get('p')
            if p is not None and (p <= 0 or p >= 1):
                return False, f"El parámetro 'p' debe ser una probabilidad válida (0 < p < 1)."
        if nombre in ["Poisson", "Exponencial", "Erlang", "Gamma"]:
            lam = params.get('λ')
            if lam is not None and lam <= 0:
                return False, f"El parámetro 'λ' (lambda) debe ser positivo (λ > 0)."

        if nombre == "Binomial":
            n_val = params.get('n')
            if n_val is not None and (n_val <= 0 or not float(n_val).is_integer()):
                return False, f"El parámetro 'n' debe ser un entero positivo."
        if nombre in ["Erlang", "Gamma"]:
            k = params.get('k')
            if k is not None and (k <= 0 or not float(k).is_integer()):
                return False, f"El parámetro 'k' debe ser un entero positivo."
        if nombre in ["Beta"]:
            alpha, beta = params.get('α'), params.get('β')
            if alpha is not None and (alpha <= 0 or not float(alpha).is_integer()):
                return False, f"El parámetro 'α' debe ser un entero positivo."
            if beta is not None and (beta <= 0 or not float(beta).is_integer()):
                return False, f"El parámetro 'β' debe ser un entero positivo."

        # Validación del número de muestras
        try:
            n = int(self.n_entry.get())
            if n <= 0:
                return False, "El 'Número de muestras' debe ser un entero positivo."
        except ValueError:
            return False, "El 'Número de muestras' debe ser un número entero."

        return True, ""


    def generar_datos(self, generar_func, nombre, tipo):
        # Lógica principal de generación, validación, prueba y ploteo
        try:
            # Cast a float para los parámetros y a int para n
            params = {p: float(self.entries[p].get()) for p in self.entries}
            n = int(self.n_entry.get())
        except ValueError:
            messagebox.showerror("Error de Entrada", "Asegúrese de ingresar solo números en todos los campos.")
            return

        valido, msg = self.validar_parametros(nombre, params)
        if not valido:
            messagebox.showerror("Error de Parámetro", msg)
            return

        try:
            # Llamada a la función de generación con el casting de parámetros específico (int para shape/n)
            if nombre=="Beta":
                datos = generar_func(int(params['α']), int(params['β']), n)
            elif nombre=="Gamma":
                datos = generar_func(int(params['k']), params['λ'], n)
            elif nombre=="Erlang":
                datos = generar_func(int(params['k']), params['λ'], n)
            elif nombre=="Binomial":
                datos = generar_func(int(params['n']), params['p'], n)
            elif nombre=="Bernoulli":
                datos = generar_func(params['p'], n)
            elif nombre=="Poisson":
                datos = generar_func(params['λ'], n)
            else: # Exponencial
                datos = generar_func(params['λ'], n)
        except Exception as e:
            messagebox.showerror("Error de Generación", str(e))
            return

        self.figura.clf()
        ax = self.figura.add_subplot(111)

        # Gráfico y prueba para distribuciones discretas
        if tipo=="discreta":
            stat, p_val, valores_teoricos, frec_teorica, desc = prueba_chi_discreta(datos, nombre.lower(), params)

            # Ajuste de bins para que coincidan con los valores discretos
            num_bins = max(len(valores_teoricos), 2)

            # Histograma (Simulado) y gráfico de barras (Teórico)
            ax.hist(datos, bins=num_bins, alpha=0.6, color='purple', density=True, label="Simulados", align='left')
            ax.bar(valores_teoricos, frec_teorica/n, width=1.0, alpha=0.4, color='red', label="Teórica", align='center')

            texto = f"Prueba Chi-cuadrado: X2 = {stat:.4f}, p-value = {p_val:.4f}\n{desc}"
            self.resultado_label.config(text=texto)
        # Gráfico y prueba para distribuciones continuas
        else:
            ax.hist(datos, bins=30, density=True, alpha=0.6, color='purple', label="Simulados")
            x = np.linspace(min(datos), max(datos), 100)
            # Dibujo de la función de densidad de probabilidad (PDF) teórica
            if nombre=="Exponencial":
                ax.plot(x, expon.pdf(x, scale=1/params['λ']), 'r', lw=2, label="Teórica")
            elif nombre=="Gamma":
                ax.plot(x, gamma_dist.pdf(x, params['k'], scale=1/params['λ']), 'r', lw=2, label="Teórica")
            elif nombre=="Erlang":
                ax.plot(x, gamma_dist.pdf(x, params['k'], scale=1/params['λ']), 'r', lw=2, label="Teórica")
            elif nombre=="Beta":
                ax.plot(x, beta_dist.pdf(x, params['α'], params['β']), 'r', lw=2, label="Teórica")

            stat, p_val, desc = prueba_ks_continua(datos, nombre.lower(), params)
            texto = f"Prueba KS: D = {stat:.4f}, p-value = {p_val:.4f}\n{desc}"
            self.resultado_label.config(text=texto)

        ax.set_title(nombre)
        ax.legend()
        self.figura.tight_layout(pad=1.0)
        self.canvas.draw()


# ===========================
# IMPLEMENTACIONES ESPECÍFICAS DE DISTRIBUCIÓN
# Subclases que inicializan el panel genérico con parámetros específicos.
# ===========================

class BernoulliFrame(DistribucionFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.crear_panel(controller, "Bernoulli", ["p"], generar_bernoulli, "discreta")

class BinomialFrame(DistribucionFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.crear_panel(controller, "Binomial", ["n","p"], generar_binomial, "discreta")

class PoissonFrame(DistribucionFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.crear_panel(controller, "Poisson", ["λ"], generar_poisson, "discreta")

class ExponencialFrame(DistribucionFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.crear_panel(controller, "Exponencial", ["λ"], generar_exponencial, "continua")

class ErlangFrame(DistribucionFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.crear_panel(controller, "Erlang", ["k","λ"], generar_erlang, "continua")

class GammaFrame(DistribucionFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.crear_panel(controller, "Gamma", ["k","λ"], generar_gamma, "continua")

class BetaFrame(DistribucionFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.crear_panel(controller, "Beta", ["α","β"], generar_beta, "continua")

# ===========================
# EJECUTAR APP
# ===========================

if __name__ == "__main__":
    app = GeneradorApp()
    app.mainloop()