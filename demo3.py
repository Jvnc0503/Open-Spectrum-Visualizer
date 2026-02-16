import sys
import numpy as np
import scipy.signal
import sounddevice as sd
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QComboBox
from PySide6.QtCore import QTimer, Qt

# ==============================================================================
# 1. MOTOR DE PROCESAMIENTO (Basado en Sección 8.2 del PDF)
# ==============================================================================
class SpectralAnalyzer:
    """
    Motor DSP que implementa las correcciones metrológicas descritas en el PDF.
    Incluye: Ventanado Flat-Top/Hann, Corrección de Amplitud (ACF), y Ponderación A.
    """
    def __init__(self, sample_rate=48000, fft_size=2048, window_type='hann'):
        self.fs = sample_rate
        self.fft_size = fft_size
        self.freqs = np.fft.rfftfreq(fft_size, 1/sample_rate)
        
        # Estado del promediado exponencial
        self.prev_spectrum = np.zeros(len(self.freqs))
        self.alpha = 0.2  # Factor de suavizado (0.0 - 1.0) 

        # Configuración de Ventana y Factor de Corrección de Amplitud (ACF) 
        self.update_window(window_type)

        # Pre-cálculo de la curva de Ponderación A (A-Weighting) 
        self.a_curve = self._compute_a_weighting(self.freqs)

    def update_window(self, window_type):
        """Configura la ventana y calcula el ACF para recuperar la energía perdida."""
        if window_type == 'flattop':
            # Flat Top: Mejor para precisión de amplitud (tipo voltímetro) 
            self.window = scipy.signal.windows.flattop(self.fft_size)
            self.acf = 1 / np.mean(self.window) # Aprox 4.18 
        else:
            # Hann: Estándar general 
            self.window = scipy.signal.windows.hann(self.fft_size)
            self.acf = 1 / np.mean(self.window) # Aprox 2.0 

    def _compute_a_weighting(self, f):
        """
        Calcula la curva de ponderación psicoacústica A según IEC 61672:2003.
        Fórmula exacta extraída de y .
        """
        # Evitar división por cero
        f = np.maximum(f, 1e-5)
        f2 = f**2
        
        # Constantes de la fórmula estándar
        const = 12194**2 * f**4
        denom = ((f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194**2))
        
        ra = const / denom
        
        # Normalización a 0dB en 1kHz (1000 Hz) 
        ra_1k_num = 12194**2 * 1000**4
        ra_1k_den = ((1000**2 + 20.6**2) * np.sqrt((1000**2 + 107.7**2) * (1000**2 + 737.9**2)) * (1000**2 + 12194**2))
        ra_1k = ra_1k_num / ra_1k_den
        
        return ra / ra_1k

    def process(self, audio_frame, weighting=True):
        """
        Procesa un bloque de audio crudo y retorna el espectro en dB.
        Sigue el flujo: DC -> Window -> FFT -> Scale/ACF -> Weight -> Average
        """
        # 1. Eliminación de componente DC (Offset) 
        audio_frame = audio_frame - np.mean(audio_frame)

        # 2. Aplicación de Ventana 
        windowed_frame = audio_frame * self.window

        # 3. FFT (Real) y Escalamiento 
        # Multiplicamos por 2/N (espectro unilateral) y por el ACF (corrección de energía)
        spectrum_mag = np.abs(np.fft.rfft(windowed_frame)) * (2 / self.fft_size) * self.acf

        # 4. Ponderación A (Opcional) 
        if weighting:
            spectrum_mag *= self.a_curve

        # 5. Promediado Exponencial (Video Filtering) 
        # Y_new = alpha * X_curr + (1 - alpha) * Y_old
        self.prev_spectrum = self.alpha * spectrum_mag + (1 - self.alpha) * self.prev_spectrum
        result = self.prev_spectrum

        # 6. Conversión a dB con piso de ruido (-120dB) 
        return 20 * np.log10(np.maximum(result, 1e-6))

# ==============================================================================
# 2. ARQUITECTURA EN TIEMPO REAL (GUI + Ring Buffer)
# ==============================================================================
class RealTimeVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # --- Configuración del Rango de Frecuencias ---
        self.MIN_FREQ = 100    # 100 Hz
        self.MAX_FREQ = 18000  # 18 kHz
        
        # Configuración de Audio
        self.SAMPLE_RATE = 48000
        self.FFT_SIZE = 4096      
        self.BUFFER_SIZE = 4096   
        
        # Inicializar Motor DSP
        self.analyzer = SpectralAnalyzer(self.SAMPLE_RATE, self.FFT_SIZE, 'hann')

        # --- Optimización: Pre-calcular la máscara de frecuencias ---
        # Esto evita tener que buscar qué índices corresponden a 100Hz-18kHz en cada frame (60 veces por segundo).
        # Creamos un array de Verdadero/Falso que indica qué bins están dentro del rango.
        self.freq_mask = (self.analyzer.freqs >= self.MIN_FREQ) & (self.analyzer.freqs <= self.MAX_FREQ)

        # --- Interfaz Gráfica (PyQt6) ---
        self.setWindowTitle("Analizador de Espectro FFT (100Hz - 18kHz)")
        self.resize(1000, 600)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Controles e Información
        self.lbl_info = QLabel(f"Rango: {self.MIN_FREQ}Hz - {self.MAX_FREQ/1000}kHz | Piso: -120dB")
        layout.addWidget(self.lbl_info)

        # Gráfico (PyQtGraph)
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)
        
        # --- Configuración de Ejes ---
        self.plot_widget.setLogMode(x=True, y=False)
        self.plot_widget.setYRange(-120, 5) 
        
        # Fijar el rango X visualmente
        # IMPORTANTE: En modo Log(x=True), setXRange espera log10(frecuencia).
        # log10(100) = 2.0
        # log10(18000) ≈ 4.25
        self.plot_widget.setXRange(np.log10(self.MIN_FREQ), np.log10(self.MAX_FREQ))
        
        self.plot_widget.setLabel('left', 'Amplitud', units='dB')
        self.plot_widget.setLabel('bottom', 'Frecuencia', units='Hz')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        self.curve = self.plot_widget.plot(pen=pg.mkPen('c', width=2))

        # --- Búfer Circular ---
        self.audio_buffer = np.zeros(self.BUFFER_SIZE)
        
        # Iniciar Stream de Audio
        try:
            self.stream = sd.InputStream(
                channels=1,
                samplerate=self.SAMPLE_RATE,
                callback=self.audio_callback,
                blocksize=512 
            )
            self.stream.start()
        except Exception as e:
            self.lbl_info.setText(f"Error de micrófono: {str(e)}")

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(16) 

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_buffer = np.roll(self.audio_buffer, -frames)
        self.audio_buffer[-frames:] = indata[:, 0]

    def update_plot(self):
        # 1. Obtener datos
        data = self.audio_buffer
        
        # (Opcional: Ganancia Digital si tu micro es muy bajo)
        data = data * 10.0 

        # 2. Procesar FFT
        spectrum_db = self.analyzer.process(data, weighting=True)
        
        # 3. Filtrar datos usando la máscara pre-calculada
        # Aplicamos la máscara tanto a las frecuencias como a las magnitudes (dB)
        # Esto elimina matemáticamente cualquier dato fuera del rango 100Hz-18kHz
        freqs_limited = self.analyzer.freqs[self.freq_mask]
        spectrum_limited = spectrum_db[self.freq_mask]
        
        # 4. Actualizar gráfico
        self.curve.setData(freqs_limited, spectrum_limited)

    def closeEvent(self, event):
        self.stream.stop()
        self.stream.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RealTimeVisualizer()
    window.show()
    sys.exit(app.exec())