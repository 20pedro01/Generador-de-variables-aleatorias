# Generador de variables aleatorias 📊

Este proyecto es una aplicación de escritorio desarrollada en Python que simula la generación de variables aleatorias discretas (Bernoulli, Binomial, Poisson) y continuas (Exponencial, Erlang, Gamma, Beta). Utiliza métodos estocásticos clásicos (Transformada Inversa, Convolución, Composición) y valida la calidad de los datos generados mediante pruebas de bondad de ajuste (Chi-cuadrado y Kolmogórov-Smirnov).

## 🚀 Instalación y requisitos

Para ejecutar esta aplicación, necesita tener instalado **Python 3.x** y las librerías científicas y de interfaz gráfica especificadas.

### 1\. Librerías requeridas

Este proyecto utiliza librerías clave que deben ser instaladas. El único requisito que no es estándar es `Pillow` (`PIL`), usado para manejar la imagen de inicio.

**Debe instalar las siguientes librerías:**

```bash
pip install numpy scipy matplotlib pillow
```

| Librería | Propósito |
| :--- | :--- |
| **`numpy`** | Operaciones vectoriales eficientes y generación de números pseudoaleatorios base. |
| **`scipy`** | Funciones de distribución, pruebas de bondad de ajuste (`kstest`, `chisquare`) y funciones de densidad teóricas. |
| **`matplotlib`** | Generación de gráficas (histogramas y curvas de densidad) dentro de Tkinter. |
| **`pillow`** (PIL) | Carga y manejo de la imagen `box.png` en la interfaz. |
| **`tkinter`** | Biblioteca estándar de Python para la Interfaz Gráfica de Usuario (GUI). |

### 2\. Estructura del proyecto

Asegúrese de que el archivo principal de Python y la imagen de inicio estén en la misma carpeta:

```
/Generador_Variables_Aleatorias/
├── generador_app.py  <-- Archivo de código principal
├── box.png           <-- Imagen utilizada en el frame de inicio
└── README.md         <-- Este archivo
```

## 💻 Uso de la aplicación

Para iniciar la aplicación, navegue hasta el directorio raíz del proyecto (`/Generador_Variables_Aleatorias/`) en su terminal o línea de comandos y ejecute el archivo Python:

```bash
python generador_app.py
```

### Flujo de trabajo

1.  **Inicio:** Seleccione **"Discreta"** o **"Continua"**.
2.  **Selección:** Elija la distribución (e.g., Binomial, Erlang).
3.  **Configuración:** Ingrese los parámetros requeridos (e.g., $p$, $\lambda$, $k$, $\alpha$) y el **Número de Muestras** ($N$).
      * *Nota:* Para Gamma, Erlang y Beta, los parámetros de forma deben ser **enteros positivos** debido al método de simulación implementado (Convolución/Composición).
4.  **Generar:** La aplicación mostrará un histograma, la curva/barras teóricas, y el resultado de la prueba de bondad de ajuste (p-value).

## 💡 Métodos de simulación implementados

El programa valida la muestra generada con el método correspondiente a la naturaleza de la distribución, asegurando la precisión de los resultados.

| Distribución | Método de generación | Prueba de validación |
| :--- | :--- | :--- |
| **Bernoulli, Exponencial, Poisson** | Transformada Inversa | Chi-cuadrado / Kolmogórov-Smirnov |
| **Binomial, Erlang, Gamma** (k entero) | Convolución | Chi-cuadrado / Kolmogórov-Smirnov |
| **Beta** ($\alpha, \beta$ enteros) | Composición (vía Gamma) | Kolmogórov-Smirnov |

## ⚙️ Desarrollo y estructura del código

El código está organizado en clases de Python siguiendo el principio de Programación Orientada a Objetos (POO):

  * **`GeneradorApp`:** Clase principal que maneja la ventana y la navegación entre Frames.
  * **`generar_*` funciones:** Funciones independientes que implementan los algoritmos estocásticos.
  * **`DistribucionFrame`:** Clase base abstracta que centraliza la lógica de la GUI, validación de inputs y el ploteo de resultados de Matplotlib.
  * **`*Frame` (Específicas):** Clases que heredan de `DistribucionFrame` y configuran los parámetros para una distribución particular.
