import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter1d
import matplotlib.ticker as ticker
import threading

# --- CONFIGURACIÓN DE AUDIO ---
SAMPLE_RATE = 48000
# FFT_SIZE controla la resolución en frecuencia; HOP_SIZE controla la “respuesta/fluidez”.
FFT_SIZE = 4096
HOP_SIZE = 1024

# Tamaño de bloque real del stream (mantener pequeño para actualizar más seguido).
BLOCK_SIZE = HOP_SIZE
CHANNELS = 1
DOWNSAMPLE = 1     # 1 = usar todo el sample rate

# Refresco del gráfico (ms): dibuja fluido; el espectro se actualiza cuando hay datos nuevos.
ANIM_INTERVAL_MS = 33.33

# Si quieres ver el status de PortAudio (p.ej. overflow), ponlo en True.
PRINT_AUDIO_STATUS = False

# Procesar FFT cada N bloques de audio (1 = cada bloque). Subirlo reduce CPU pero también “respuesta”.
PROCESS_EVERY_N_BLOCKS = 1

# --- CONFIGURACIÓN ESTÉTICA (LOOK PRO) ---
COLOR_BG = '#121212'       # Fondo casi negro
COLOR_PLOT_BG = '#1e1e1e'  # Fondo del gráfico
COLOR_GRID = '#333333'     # Color de la rejilla (sutil)
COLOR_TRACE = '#00f0ff'    # Cian neón para la señal principal
COLOR_TEXT = '#b0b0b0'     # Texto gris claro

class RealTimeAnalyzer:
    def __init__(self):
        self._audio_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._audio_status_printed = False
        self._overflow_count = 0

        # Ring buffer “doble”: ventana contigua en _ring[pos:pos+FFT_SIZE]
        self._ring = np.zeros(FFT_SIZE * 2, dtype=np.float32)
        self._write_pos = 0  # dentro de [0, FFT_SIZE)
        self._filled = 0

        self._window = np.hanning(FFT_SIZE).astype(np.float32)
        self._new_data_event = threading.Event()

        # Último espectro calculado (dB)
        self._latest_db = np.full(FFT_SIZE // 2 + 1, -90.0, dtype=np.float32)
        self._latest_seq = 0

        self.stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype='float32',
            latency='high',
            callback=self.audio_callback,
        )

        self._worker = threading.Thread(target=self._processing_loop, daemon=True)
        self._worker.start()
        
        # Preparar Gráfico
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.fig.patch.set_facecolor(COLOR_BG)
        self.ax.set_facecolor(COLOR_PLOT_BG)
        
        # Frecuencias para el eje X
        self.freqs = np.fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
        
        # Línea inicial
        self.line, = self.ax.plot([], [], color=COLOR_TRACE, lw=1.5, alpha=0.9)
        self.fill = None # Para el efecto de relleno opcional
        
        self.setup_plot()

    def setup_plot(self):
        """Configura los ejes y la rejilla estilo software de audio"""
        self.ax.set_xscale('log')
        self.ax.set_xlim(50, 20000)
        self.ax.set_ylim(-90, 0) # Rango en dBFS (ajustar si tienes calibración)
        
        # --- EJE X (Frecuencias ISO Estándar) ---
        # Etiquetas principales (Octavas)
        major_ticks = [50, 100, 200, 500, 1000, 2000, 4000, 8000, 16000]
        major_labels = ['50', '100', '200', '500', '1k', '2k', '4k', '8k', '16k']
        
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
            # Contabiliza overflows sin spamear la consola.
            try:
                if getattr(status, 'input_overflow', False):
                    self._overflow_count += 1
            except Exception:
                pass

            if PRINT_AUDIO_STATUS and not self._audio_status_printed:
                print(f"Audio stream status: {status} (further messages suppressed)")
                self._audio_status_printed = True

        # Push al ring buffer (mínimo trabajo en callback)
        channel_data = indata[:, 0]
        n = int(len(channel_data))
        if n <= 0:
            return

        # Si por algún motivo llega un bloque mayor al FFT_SIZE, nos quedamos con el final.
        if n > FFT_SIZE:
            channel_data = channel_data[-FFT_SIZE:]
            n = FFT_SIZE

        with self._audio_lock:
            end = self._write_pos + n
            self._ring[self._write_pos:end] = channel_data
            self._ring[self._write_pos + FFT_SIZE:end + FFT_SIZE] = channel_data
            self._write_pos = (self._write_pos + n) % FFT_SIZE
            self._filled = min(FFT_SIZE, self._filled + n)
            self._latest_seq += 1
        self._new_data_event.set()

    def _processing_loop(self):
        """Hilo de procesamiento: calcula FFT sin bloquear el callback ni la GUI."""
        snapshot = np.zeros(FFT_SIZE, dtype=np.float32)
        last_processed_seq = 0

        while not self._stop_event.is_set():
            # Espera datos nuevos (con timeout para permitir salir)
            self._new_data_event.wait(timeout=0.25)
            self._new_data_event.clear()

            with self._audio_lock:
                if self._filled < FFT_SIZE:
                    continue
                seq = self._latest_seq
                if seq == last_processed_seq:
                    continue
                if PROCESS_EVERY_N_BLOCKS > 1 and (seq - last_processed_seq) < PROCESS_EVERY_N_BLOCKS:
                    continue
                start = self._write_pos
                np.copyto(snapshot, self._ring[start:start + FFT_SIZE])
                last_processed_seq = seq

            # FFT + dB
            windowed = snapshot * self._window
            fft_data = np.fft.rfft(windowed)
            magnitude = np.abs(fft_data)

            # dBFS relativo (normalizado al máximo del frame)
            max_mag = float(np.max(magnitude) + 1e-12)
            with np.errstate(divide='ignore'):
                magnitude_db = 20.0 * np.log10(magnitude / max_mag)

            magnitude_db = magnitude_db - 10.0

            smoothed_db = self.smooth_curve(magnitude_db, sigma=2)
            smoothed_db = np.asarray(smoothed_db, dtype=np.float32)

            with self._audio_lock:
                np.copyto(self._latest_db, smoothed_db)

    def process_data(self):
        """Devuelve el último espectro ya calculado (la FFT corre en el hilo de fondo)."""
        with self._audio_lock:
            return self._latest_db.copy()

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
        latest = self.process_data()
        self.line.set_data(self.freqs, latest)
        
        return self.line,

    def start(self):
        # Usamos blit=True para mayor rendimiento (FPS)
        self.ani = FuncAnimation(
            self.fig,
            self.update,
            interval=ANIM_INTERVAL_MS,
            blit=True,
            cache_frame_data=False,
        )

        def _on_close(_evt):
            self._stop_event.set()

        self.fig.canvas.mpl_connect('close_event', _on_close)
        with self.stream:
            plt.show()

# --- EJECUTAR ---
if __name__ == "__main__":
    rta = RealTimeAnalyzer()
    rta.start()