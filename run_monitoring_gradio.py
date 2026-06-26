import os
import ctypes
from ctypes import wintypes
import threading
from pathlib import Path
from time import time, sleep
from PIL import Image as PILImage
from dbd.AI_model import AI_model
from dbd.geometric_detector import GeometricSkillCheckDetector
from dbd.utils.directkeys import PressKey, ReleaseKey, SPACE

# ── Windows performance ────────────────────────────────────────────────────────
_winmm    = ctypes.WinDLL('winmm')
_kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
_avrt     = ctypes.WinDLL('avrt')

_kernel32.SetProcessInformation.argtypes = [
    wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
]
_kernel32.SetProcessInformation.restype = wintypes.BOOL


class _THROTTLING(ctypes.Structure):
    _fields_ = [("Version", wintypes.ULONG),
                ("ControlMask", wintypes.ULONG),
                ("StateMask", wintypes.ULONG)]


def _boost_this_thread():
    """Desativa EcoQoS + TIME_CRITICAL + MMCSS Pro Audio na thread chamante."""
    handle = _kernel32.GetCurrentProcess()
    try:
        _kernel32.SetPriorityClass(handle, 0x00000080)          # HIGH_PRIORITY_CLASS
    except Exception:
        pass
    try:
        state = _THROTTLING(Version=1, ControlMask=0x5, StateMask=0)
        _kernel32.SetProcessInformation(handle, 4, ctypes.byref(state), ctypes.sizeof(state))
    except Exception:
        pass
    try:
        _kernel32.SetThreadPriority(_kernel32.GetCurrentThread(), 15)  # TIME_CRITICAL
    except Exception:
        pass
    try:
        _avrt.AvSetMmThreadCharacteristicsW.restype  = wintypes.HANDLE
        _avrt.AvSetMmThreadCharacteristicsW.argtypes = [
            wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD),
        ]
        task_index = wintypes.DWORD(0)
        _avrt.AvSetMmThreadCharacteristicsW("Pro Audio", ctypes.byref(task_index))
    except Exception:
        pass


_boost_this_thread()

# ── Constantes de UI ───────────────────────────────────────────────────────────
PICTURES_DIR = Path(os.path.expanduser("~")) / "Pictures" / "dbd_hits"

debug_options = [
    "Nenhum (padrão)",
    "Exibir frame monitorado ao vivo",
    "Salvar frames dos acertos",
    "Salvar todos os frames de skill check",
]

# ── Gradio imports ─────────────────────────────────────────────────────────────
from gradio import (
    Dropdown, Radio, Number, Image, Label, Button, Slider,
    skip, Error, Blocks, Row, Column, Markdown,
)


# ── Helpers de I/O assíncronos ─────────────────────────────────────────────────
def _release_space_later():
    sleep(0.005)
    ReleaseKey(SPACE)


def _save_frame_async(frame_array, path):
    PILImage.fromarray(frame_array).save(str(path))


# ── Loop principal ─────────────────────────────────────────────────────────────
def monitor(ai_model_path, device, debug_option,
            confidence_threshold, delay_compensation_ms, hit_ante_ms):

    if ai_model_path is None or not os.path.exists(ai_model_path):
        raise Error("Arquivo do modelo IA não encontrado", duration=0)

    # Aplica boost na thread do worker do Gradio (MMCSS é por thread)
    _boost_this_thread()

    try:
        use_gpu = (device == devices[1])
        ai_model = AI_model(ai_model_path, use_gpu, nb_cpu_threads=None)
        provider  = ai_model.check_provider()
    except Exception as e:
        raise Error(f"Erro ao carregar modelo IA: {e}", duration=0)

    print(f"[INFO] Provider: {provider}")

    geo = GeometricSkillCheckDetector(pipeline_ms=float(delay_compensation_ms))

    debug_folder = "saved_images"
    if debug_option in (debug_options[2], debug_options[3]):
        Path(debug_folder).mkdir(exist_ok=True)
        for i in range(len(ai_model.pred_dict)):
            Path(os.path.join(debug_folder, str(i))).mkdir(exist_ok=True)

    PICTURES_DIR.mkdir(parents=True, exist_ok=True)

    display_live  = (debug_option == debug_options[1])
    nb_frames     = 0
    nb_hits       = 0
    last_hit_time = 0.0
    MIN_HIT_DELAY = 0.35
    last_update   = time()
    non_sc_streak = 999  # força reset do geo na primeira detecção

    _winmm.timeBeginPeriod(1)
    try:
        while True:
            frame = ai_model.grab_frame_numpy()
            if frame is None:
                sleep(0.001)
                continue

            nb_frames    += 1
            current_time  = time()

            frame_input = ai_model.numpy_to_model_input(frame)
            pred, desc, probs_dict, ai_should_hit, confidence = ai_model.predict(
                frame_input,
                confidence_threshold=float(confidence_threshold),
            )

            # Detector geométrico: só roda quando IA confirma skill check ativo.
            # Evita falsos positivos de HSV (sangue, fogo, cabelo vermelho, UI branca).
            if pred != 0:
                if non_sc_streak >= 3:
                    geo.reset()
                non_sc_streak = 0
                geo_should_hit, _ = geo.process(frame)
            else:
                non_sc_streak += 1
                geo_should_hit = False

            # Geo dispara cedo via predição de velocidade; IA é fallback no warm-up.
            should_hit = geo_should_hit or ai_should_hit

            if display_live and (current_time - last_update) > 1.0:
                yield skip(), PILImage.fromarray(frame), skip()

            if debug_option == debug_options[3] and pred != 0:
                path = os.path.join(debug_folder, str(pred), f"{nb_hits}.png")
                threading.Thread(target=_save_frame_async, args=(frame, path), daemon=True).start()

            if should_hit and (current_time - last_hit_time) > MIN_HIT_DELAY:
                last_hit_time = current_time
                nb_hits      += 1

                # Geo já compensou o delay de pipeline; ante-frontier via IA ainda precisa de delay.
                delay_ms = float(hit_ante_ms) if (not geo_should_hit and pred in (2, 9)) else 0.0

                def _do_hit(d=delay_ms):
                    if d > 0:
                        sleep(d / 1000.0)
                    PressKey(SPACE)
                    sleep(0.005)
                    ReleaseKey(SPACE)

                threading.Thread(target=_do_hit, daemon=True).start()

                if debug_option == debug_options[2]:
                    hit_path = PICTURES_DIR / f"hit_{nb_hits:04d}.png"
                    threading.Thread(
                        target=_save_frame_async, args=(frame, hit_path), daemon=True
                    ).start()

                yield skip(), PILImage.fromarray(frame), probs_dict
                continue

            t_diff = current_time - last_update
            if t_diff > 0.5:
                fps = round(nb_frames / t_diff, 1)
                yield fps, skip(), skip()
                last_update = current_time
                nb_frames   = 0

    finally:
        _winmm.timeEndPeriod(1)


# ── Interface Gradio ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    devices     = ["CPU (padrão)", "GPU"]
    model_files = [f for f in os.listdir() if f.endswith(".onnx") or f.endswith(".engine")]
    if not model_files:
        model_files = ["model.onnx"]

    with Blocks(title="Recruta") as webui:
        Markdown("<h1 style='text-align: center;'>Recruta</h1>")

        with Row():
            with Column(variant="panel"):

                with Column(variant="panel"):
                    Markdown("### Modelo")
                    ai_model_path = Dropdown(
                        choices=model_files,
                        value=model_files[0],
                        label="Arquivo do modelo IA (ONNX ou TensorRT)",
                    )
                    device = Radio(
                        choices=devices,
                        value=devices[0],
                        label="Dispositivo",
                    )

                with Column(variant="panel"):
                    Markdown("### Configurações de timing")
                    delay_compensation = Slider(
                        minimum=0, maximum=200, step=5, value=80,
                        label="Compensação de delay (ms)",
                        info=(
                            "Controle principal de timing do detector geométrico. "
                            "Errou DEPOIS da great → AUMENTE. "
                            "Errou ANTES → diminua. "
                            "Típico: 50–120 ms."
                        ),
                    )
                    confidence_threshold = Slider(
                        minimum=0.25, maximum=0.95, step=0.05, value=0.65,
                        label="Confiança mínima (IA)",
                        info="Diminua se não clicar. Aumente se clicar errado.",
                    )
                    hit_ante = Slider(
                        minimum=0, maximum=100, step=5, value=25,
                        label="Delay ante-frontier (ms)",
                        info="Delay extra ao detectar ante-frontier pela IA (fallback).",
                    )

                with Column(variant="panel"):
                    Markdown("### Debug")
                    debug_option = Dropdown(
                        choices=debug_options,
                        value=debug_options[0],
                        label="Modo debug",
                    )

                with Column():
                    run_button  = Button("INICIAR", variant="primary")
                    stop_button = Button("PARAR",   variant="stop")

            with Column(variant="panel"):
                fps_out   = Number(label="FPS", interactive=False)
                image_out = Image(label="Último acerto", height=224, interactive=False)
                probs_out = Label(label="Última detecção")

        mon = run_button.click(
            fn=monitor,
            inputs=[
                ai_model_path, device, debug_option,
                confidence_threshold, delay_compensation, hit_ante,
            ],
            outputs=[fps_out, image_out, probs_out],
        )
        stop_button.click(fn=None, inputs=None, outputs=None, cancels=[mon])

    webui.launch()
