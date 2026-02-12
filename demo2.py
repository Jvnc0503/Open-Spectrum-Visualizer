import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter1d
import matplotlib.ticker as ticker

# --- CONFIGURACIÓN DE AUDIO ---
SAMPLE_RATE = 48000
BLOCK_SIZE = 4096  # Mayor tamaño para mejor resolución en bajos
CHANNELS = 1
DOWNSAMPLE = 1     # 1 = usar todo el sample rate

# --- CONFIGURACIÓN ESTÉTICA (LOOK PRO) ---
COLOR_BG = '#121212'       # Fondo casi negro
COLOR_PLOT_BG = '#1e1e1e'  # Fondo del gráfico
COLOR_GRID = '#333333'     # Color de la rejilla (sutil)
COLOR_TRACE = '#00f0ff'    # Cian neón para la señal principal
COLOR_TEXT = '#b0b0b0'     # Texto gris claro

class RealTimeAnalyzer:
    def __init__(self):
        self.stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            callback=self.audio_callback
        )
        self.audio_data = np.zeros(BLOCK_SIZE)
        
        # Preparar Gráfico
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.fig.patch.set_facecolor(COLOR_BG)
        self.ax.set_facecolor(COLOR_PLOT_BG)
        
        # Frecuencias para el eje X
        self.freqs = np.fft.rfftfreq(BLOCK_SIZE, 1/SAMPLE_RATE)
        
        # Línea inicial
        self.line, = self.ax.plot([], [], color=COLOR_TRACE, lw=1.5, alpha=0.9)
        self.fill = None # Para el efecto de relleno opcional
        
        self.setup_plot()

    def setup_plot(self):
        """Configura los ejes y la rejilla estilo software de audio"""
        self.ax.set_xscale('log')
        self.ax.set_xlim(20, 20000)
        self.ax.set_ylim(-90, 0) # Rango en dBFS (ajustar si tienes calibración)
        
        # --- EJE X (Frecuencias ISO Estándar) ---
        # Etiquetas principales (Octavas)
        major_ticks = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        major_labels = ['31', '63', '125', '250', '500', '1k', '2k', '4k', '8k', '16k']
        
        self.ax.set_xticks(major_ticks)
        self.ax.set_xticklabels(major_labels, fontsize=10, color=COLOR_TEXT, fontfamily='monospace')
        
        # --- REJILLA (GRID) ---
        # Rejilla principal (Líneas sólidas tenues)
        self.ax.grid(which='major', color=COLOR_GRID, linestyle='-', linewidth=1, alpha=0.6)
        # Rejilla menor (Puntos muy sutiles para logaritmo)
        self.ax.grid(which='minor', color=COLOR_GRID, linestyle=':', linewidth=0.5, alpha=0.3)
        
        # Etiquetas y Títulos
        self.ax.set_ylabel("Amplitude (dBFS)", color=COLOR_TEXT, fontsize=10)
        self.ax.set_xlabel("Frequency (Hz)", color=COLOR_TEXT, fontsize=10)
        self.ax.set_title("REAL TIME ANALYZER | 1/3 OCT SMOOTHING", color='white', fontsize=12, pad=10, loc='left', fontweight='bold')
        
        # Bordes limpios
        for spine in self.ax.spines.values():
            spine.set_edgecolor('#444444')

    def audio_callback(self, indata, frames, time, status):
        """Callback que recibe el audio crudo"""
        if status:
            print(status)
        # Tomamos el primer canal y copiamos
        self.audio_data = indata[:, 0].copy()

    def process_data(self):
        """Calcula FFT y convierte a dB"""
        # Ventana Hanning para reducir fugas espectrales
        windowed_data = self.audio_data * np.hanning(len(self.audio_data))
        
        # FFT
        fft_data = np.fft.rfft(windowed_data)
        magnitude = np.abs(fft_data)
        
        # Convertir a dB con protección contra log(0)
        with np.errstate(divide='ignore'):
            magnitude_db = 20 * np.log10(magnitude / np.max(magnitude + 1e-9))
        
        # Normalizar visualmente (offset simple para que se vea en pantalla)
        magnitude_db = magnitude_db - 10 
        
        return magnitude_db

    def smooth_curve(self, y_data, sigma=3):
        """
        Suavizado estilo RTA. 
        Usamos filtro gaussiano como aproximación rápida al suavizado por octavas.
        Para un suavizado de 1/3 octava real se requiere re-muestreo, 
        pero esto visualmente funciona muy bien.
        """
        return gaussian_filter1d(y_data, sigma=sigma)

    def update(self, frame):
        """Función de actualización para la animación"""
        mag_db = self.process_data()
        
        # Aplicamos suavizado para que parezca profesional y no "nervioso"
        # Cuanto mayor el sigma, más suave la curva.
        smoothed_db = self.smooth_curve(mag_db, sigma=5) # Ajusta sigma a gusto (2-10)
        
        self.line.set_data(self.freqs, smoothed_db)
        
        return self.line,

    def start(self):
        # Usamos blit=True para mayor rendimiento (FPS)
        self.ani = FuncAnimation(self.fig, self.update, interval=30, blit=True)
        with self.stream:
            plt.show()

# --- EJECUTAR ---
if __name__ == "__main__":
    rta = RealTimeAnalyzer()
    rta.start()