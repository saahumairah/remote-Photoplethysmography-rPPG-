import cv2
import numpy as np
from scipy.signal import butter, filtfilt, detrend
from collections import deque
import time

# Main Parameters
FPS = 30     
Buffer_Duration = 5 
MIN_BPM = 60 
MAX_BPM = 120 

# BPM Stabilization Parameters
BPM_MEDIAN_SIZE = 15  
BPM_WARMUP_SIZE = 3      
BPM_JUMP_GATE = 12.0
HARMONIC_TOL = 0.12  
SNR_THRESHOLD = 1.5 
SNR_WARMUP = 1.2 
PROMINENCE_RATIO = 1.3 
CONFIDENCE_DECAY = 15 
MIN_HZ = MIN_BPM / 60.0 
MAX_HZ = MAX_BPM / 60.0

# Adaptive FPS
class AdaptiveFPS:
    def __init__(self, window: int = 60):
        self.timestamps: deque = deque(maxlen=window)

    def tick(self) -> None:
        self.timestamps.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self.timestamps) < 2:
            return FPS
        elapsed = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / elapsed if elapsed > 0 else FPS

# Plane Orthogonal to Skin
def extract_pos_signal(rgb_buffer: np.ndarray):
    mean_rgb = np.mean(rgb_buffer, axis=0)
    mean_rgb[mean_rgb == 0] = 1.0 
    
    normalized = rgb_buffer / mean_rgb
    Rn = normalized[:, 0]
    Gn = normalized[:, 1]
    Bn = normalized[:, 2]

    X = 3 * Rn - 2 * Gn
    Y = 1.5 * Rn + Gn - 1.5 * Bn

    X_detrend = detrend(X, type='linear')
    Y_detrend = detrend(Y, type='linear')
    
    std_x = np.std(X_detrend)
    std_y = np.std(Y_detrend)
    
    alpha = std_x / std_y if std_y > 1e-6 else 0.0

    H = X_detrend + alpha * Y_detrend
    return H

# Bandpass Filter
def butter_bandpass(data: np.ndarray, lowcut: float, highcut: float, 
                    fs: float, order: int = 5):
    nyq = 0.5 * fs
    low = np.clip(lowcut  / nyq, 1e-4, 0.9999)
    high = np.clip(highcut / nyq, 1e-4, 0.9999)
    if low >= high:
        raise ValueError(f"Cutoff tidak valid: low={low:.4f} >= high={high:.4f}")
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Harmonic guard
def resolve_harmonic(candidate_bpm: float, reference_bpm: float, tol: float = HARMONIC_TOL):
    if reference_bpm <= 0:
        return candidate_bpm
    is_subharmonic = abs(candidate_bpm - reference_bpm * 0.5) / reference_bpm < tol
    is_harmonic = abs(candidate_bpm - reference_bpm * 2.0) / reference_bpm < tol
    return reference_bpm if (is_subharmonic or is_harmonic) else candidate_bpm

# Peak (comparing 5 peaks on the left and right with threshold ratio = 1.3)
def peak_is_prominent(fft_mag: np.ndarray, peak_idx: int, neighborhood: int = 5, ratio: float = PROMINENCE_RATIO):
    n = len(fft_mag)
    lo = max(0, peak_idx - neighborhood)
    hi = min(n, peak_idx + neighborhood + 1)
    neighbors = np.concatenate([fft_mag[lo:peak_idx], fft_mag[peak_idx+1:hi]])
    if len(neighbors) == 0:
        return True
    mean_neighbor = np.mean(neighbors)
    return bool((mean_neighbor > 0) and (fft_mag[peak_idx] / mean_neighbor >= ratio))

# Estimate BPM
def estimate_bpm(signal: np.ndarray, fps: float):
    n = len(signal)
    sig = detrend(signal.copy(), type='linear')
    windowed = sig * np.hanning(n)

    n_fft = n * 4
    fft_mag = np.abs(np.fft.rfft(windowed, n=n_fft))
    fft_freq = np.fft.rfftfreq(n_fft, d=1.0 / fps)

    mask = (fft_freq >= MIN_HZ) & (fft_freq <= MAX_HZ)
    if not np.any(mask):
        return 0.0, 0.0

    masked_mag = fft_mag[mask]
    masked_freq = fft_freq[mask]

    # FFT 
    # Probability center at 78 BPM (1.3 Hz) to dampen low frequency dominance
    center_hz = 78.0 / 60.0
    std_hz = 30.0 / 60.0
    weights = np.exp(-0.5 * ((masked_freq - center_hz) / std_hz)**2)
    weighted_mag = masked_mag * (0.5 + 0.5 * weights) 
    
    peak_idx = int(np.argmax(weighted_mag))
    peak_freq = masked_freq[peak_idx]
    peak_amp = masked_mag[peak_idx]

    mean_amp = np.mean(masked_mag)
    snr = float(peak_amp / mean_amp) if mean_amp > 0 else 0.0

    if not peak_is_prominent(masked_mag, peak_idx):
        return float(peak_freq * 60.0), 0.0

    return float(peak_freq * 60.0), snr

# BPM Stabilizer
class BPMStabilizer:
    def __init__(self):
        self._history: deque = deque(maxlen = BPM_MEDIAN_SIZE)
        self._low_snr_streak: int = 0
        self.bpm_display: float = 0.0
        self.frozen: bool = False
    @property
    def _is_warmup(self):
        return self.bpm_display == 0.0 or len(self._history) < BPM_WARMUP_SIZE
    def update(self, bpm_raw: float, snr: float):
        snr_thr = SNR_WARMUP if self._is_warmup else SNR_THRESHOLD

        if snr < snr_thr:
            self._low_snr_streak += 1
            self.frozen = (self._low_snr_streak >= CONFIDENCE_DECAY)
            return self.bpm_display
        else:
            self._low_snr_streak = 0
            self.frozen = False

        if not (MIN_BPM <= bpm_raw <= MAX_BPM):
            return self.bpm_display

        if not self._is_warmup:
            guarded = resolve_harmonic(bpm_raw, self.bpm_display)
            if guarded != bpm_raw:
                return self.bpm_display

        if not self._is_warmup and abs(bpm_raw - self.bpm_display) > BPM_JUMP_GATE:
            return self.bpm_display
        self._history.append(bpm_raw)
        med_window = BPM_WARMUP_SIZE if self._is_warmup else BPM_MEDIAN_SIZE
        recent = list(self._history)[-med_window:]
        self.bpm_display = float(np.median(recent))
        
        return self.bpm_display

# ROI FOREHEAD
def get_forehead_roi(frame: np.ndarray, x: int, y: int, w: int, h: int):
    fh_x  = x + int(w * 0.25)
    fh_y  = y + int(h * 0.10)
    fh_x2 = fh_x + int(w * 0.50)
    fh_y2 = fh_y + int(h * 0.20)

    fH, fW = frame.shape[:2]
    fh_x  = max(0, fh_x);   fh_y  = max(0, fh_y)
    fh_x2 = min(fW, fh_x2); fh_y2 = min(fH, fh_y2)

    if fh_x2 <= fh_x or fh_y2 <= fh_y:
        return None, None

    return frame[fh_y:fh_y2, fh_x:fh_x2], (fh_x, fh_y, fh_x2 - fh_x, fh_y2 - fh_y)

def draw_signal_graph(filtered_signal: np.ndarray, bpm: float, snr: float, fps: float, canvas_w: int = 500, canvas_h: int = 250):
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    sig_range = filtered_signal.max() - filtered_signal.min()
    if sig_range < 1e-8:
        return canvas

    pad_left = 45
    pad_bottom = 30
    pad_top = 30
    pad_right = 15

    draw_w = canvas_w - pad_left - pad_right
    draw_h = canvas_h - pad_top - pad_bottom

    norm = (filtered_signal - filtered_signal.min()) / sig_range * draw_h
    n = len(norm)
    
    xs = pad_left + (np.arange(n) * draw_w / n).astype(int)
    ys = canvas_h - pad_bottom - norm.astype(int)

    # Axes
    cv2.line(canvas, (pad_left, pad_top), (pad_left, canvas_h - pad_bottom), (200, 200, 200), 2)
    cv2.line(canvas, (pad_left, canvas_h - pad_bottom), (canvas_w - pad_right, canvas_h - pad_bottom), (200, 200, 200), 2)

    # Zero-line (Baseline)
    center_y = canvas_h - pad_bottom - (draw_h // 2)
    cv2.line(canvas, (pad_left, center_y), (canvas_w - pad_right, center_y), (40, 40, 40), 1)

    # rPPG Signal
    for i in range(1, n):
        cv2.line(canvas, (int(xs[i-1]), int(ys[i-1])), (int(xs[i]), int(ys[i])), (255, 150, 0), 2)

    # X-Axis (Time)
    buffer_sec = n / fps if fps > 0 else 5.0
    cv2.putText(canvas, f"-{buffer_sec:.1f}s", (pad_left, canvas_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(canvas, "0s", (canvas_w - pad_right - 20, canvas_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(canvas, "Waktu", (canvas_w // 2 - 20, canvas_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Y-Axis (Amplitude)
    cv2.putText(canvas, "Amp", (5, pad_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(canvas, "Max", (10, pad_top + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    cv2.putText(canvas, "Min", (10, canvas_h - pad_bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    # Overlay Text Info
    snr_color = (0, 255, 0) if snr >= SNR_THRESHOLD else (0, 100, 255)
    cv2.putText(canvas, f"FPS: {fps:.1f}   SNR: {snr:.2f}   BPM: {bpm:.1f}",
                (pad_left, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, snr_color, 1)
                
    return canvas

# MAIN LOOP
# Detect Face
def main() -> None:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Tidak dapat membuka kamera.")
        return
    cap.set(cv2.CAP_PROP_FPS, FPS)

    fps_est = AdaptiveFPS(window=60)
    stabilizer = BPMStabilizer()
    rgb_buffer: deque = deque()  
    last_snr = 0.0
    
    # Variabel Smooth Frame Face Detection : EMA (Exponential Moving Average)
    smooth_box = None
    alpha_box = 0.15  

    print("Kamera aktif. Arahkan wajah ke kamera, pastikan pencahayaan cukup.")
    print("Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame tidak terbaca.")
            break

        fps_est.tick()
        fps = fps_est.fps
        buffer_size = max(30, int(fps * Buffer_Duration))

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor = 1.3, minNeighbors = 5, minSize=(100, 100)
        )
        face_detected = len(faces) > 0

        if face_detected:
            x, y, w, h = faces[0]
            
            # Stabilizer ROI using EMA (Exponential Moving Average)
            if smooth_box is None:
                smooth_box = np.array([x, y, w, h], dtype=np.float32)
            else:
                curr_box = np.array([x, y, w, h], dtype=np.float32)
                smooth_box = alpha_box * curr_box + (1.0 - alpha_box) * smooth_box
            
            s_x, s_y, s_w, s_h = smooth_box.astype(int)
            cv2.rectangle(frame, (s_x, s_y), (s_x+s_w, s_y+s_h), (0, 255, 0), 2)

            roi, coords = get_forehead_roi(frame, s_x, s_y, s_w, s_h)
            
            if roi is not None and roi.size > 0:
                fx, fy, fw, fh = coords
                cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)

                b, g, r = cv2.mean(roi)[:3]
                rgb_buffer.append([r, g, b])
                
                while len(rgb_buffer) > buffer_size:
                    rgb_buffer.popleft()

                if len(rgb_buffer) >= buffer_size:
                    arr_rgb = np.array(rgb_buffer, dtype=np.float64)
                    try:
                        pos_signal = extract_pos_signal(arr_rgb)
                        filtered = butter_bandpass(pos_signal, MIN_HZ, MAX_HZ, fps)
                        bpm_raw, snr = estimate_bpm(filtered, fps)
                        
                        last_snr = snr
                        bpm_display = stabilizer.update(bpm_raw, snr)

                        graph = draw_signal_graph(filtered, bpm_display, snr, fps)
                        cv2.imshow('Grafik Sinyal rPPG', graph)

                    except (ValueError, np.linalg.LinAlgError):
                        bpm_display = stabilizer.bpm_display
                else:
                    bpm_display = stabilizer.bpm_display

                # Overlay Text
                if bpm_display > 0:
                    bpm_color = (0, 0, 255) if not stabilizer.frozen else (0, 140, 255)
                    snr_color = (0, 255, 0) if last_snr >= SNR_THRESHOLD else (0, 100, 255)
                    frozen_tag = " [FREEZE]" if stabilizer.frozen else ""
                    cv2.putText(frame, f"BPM: {bpm_display:.1f}{frozen_tag}", (s_x, s_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bpm_color, 2)
                    cv2.putText(frame, f"SNR: {last_snr:.2f}", (s_x, s_y - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.52, snr_color, 1)
                else:
                    pct = int(len(rgb_buffer) / buffer_size * 100)
                    cv2.putText(frame, f"Kalibrasi... {pct}%", (s_x, s_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            smooth_box = None

        # Status Bar
        fH = frame.shape[0]
        status = "Wajah Terdeteksi" if face_detected else "Cari Wajah..."
        status_color = (0, 255, 0) if face_detected else (0, 100, 255)
        cv2.putText(frame, f"FPS: {fps:.1f}  |  {status}", (10, fH - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

        cv2.imshow('Deteksi Detak Jantung rPPG', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Program selesai.")

if __name__ == '__main__':
    main()