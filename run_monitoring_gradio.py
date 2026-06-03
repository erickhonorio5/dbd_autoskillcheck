import os
import ctypes
import threading
from pathlib import Path
from time import time, sleep
from dbd.AI_model import AI_model
from dbd.utils.directkeys import PressKey, ReleaseKey, SPACE

# Windows timer resolution: default is ~15ms, this sets it to ~1ms.
# Critical for hit_ante precision — without this, a 25ms sleep can fire anywhere between 10-40ms.
_winmm = ctypes.WinDLL('winmm')


def _perform_hit(delay_ms: float):
    """Executa o key press em background para não bloquear o loop principal durante o delay."""
    if delay_ms > 0:
        sleep(delay_ms / 1000.0)
    PressKey(SPACE)
    sleep(0.005)
    ReleaseKey(SPACE)

from gradio import (
    Dropdown, Radio, Number, Image, Label, Button, Slider,
    skip, Info, Warning, Error, Blocks, Row, Column, Markdown
)


def monitor(ai_model_path, device, debug_option, hit_ante, cpu_stress, confidence_threshold):
    if ai_model_path is None or not os.path.exists(ai_model_path):
        raise Error("Invalid AI model file", duration=0)

    if device is None:
        raise Error("Invalid device option")

    if debug_option is None:
        raise Error("Invalid debug option")

    # Configurações otimizadas de CPU: apenas low, normal e high
    CPU_CONFIGS = {
        "low": {
            "nb_cpu_threads": 2,
            "target_fps": 30,
            "description": "Economia de CPU - 2 threads, ~30 FPS"
        },
        "normal": {
            "nb_cpu_threads": 4,
            "target_fps": 60,
            "description": "Balanceado - 4 threads, ~60 FPS"
        },
        "high": {
            "nb_cpu_threads": 8,
            "target_fps": 120,
            "description": "Máxima performance - 8 threads, ~120 FPS"
        }
    }
    
    config = CPU_CONFIGS.get(cpu_stress, CPU_CONFIGS["normal"])
    nb_cpu_threads = config["nb_cpu_threads"]
    target_fps = config["target_fps"]

    try:
        use_gpu = (device == devices[1])
        ai_model = AI_model(ai_model_path, use_gpu, nb_cpu_threads)
        execution_provider = ai_model.check_provider()
    except Exception as e:
        raise Error("Error when loading AI model: {}".format(e), duration=0)

    print(f"[INFO] CPU Config: {config['description']}")
    
    if execution_provider == "CUDAExecutionProvider":
        print("Running AI model on GPU (CUDA)")
    elif execution_provider == "DmlExecutionProvider":
        print("Running AI model on GPU (DirectML)")
    elif execution_provider == "TensorRT":
        print("Running AI model on GPU (TensorRT)")
    else:
        print(f"Running AI model on CPU ({nb_cpu_threads} threads)")
        if use_gpu:
            print("WARNING: Could not run on GPU. Using CPU.")

    # Create debug folders
    if debug_option == debug_options[2] or debug_option == debug_options[3]:
        Path(debug_folder).mkdir(exist_ok=True)
        for folder_idx in range(len(ai_model.pred_dict)):
            Path(os.path.join(debug_folder, str(folder_idx))).mkdir(exist_ok=True)

    # Variables
    t0 = time()
    nb_frames = 0
    nb_hits = 0
    last_hit_time = 0
    MIN_HIT_DELAY = 0.3

    # Controle de FPS baseado na configuração de CPU
    frame_time = 1.0 / target_fps
    last_fps_update = time()

    # Aumenta resolução do timer do Windows: 15ms → ~1ms
    # Sem isso, hit_ante=25ms pode variar entre 10-40ms na prática
    _winmm.timeBeginPeriod(1)
    try:
        while True:
            frame_start = time()

            screenshot = ai_model.grab_screenshot()
            image_pil = ai_model.screenshot_to_pil(screenshot)
            image_np = ai_model.pil_to_numpy(image_pil)
            nb_frames += 1

            # Predição simples e rápida
            pred, desc, probs, should_hit, confidence = ai_model.predict(
                image_np,
                confidence_threshold,
                require_consecutive=1
            )

            current_time = time()

            if pred != 0 and debug_option == debug_options[3]:
                path = os.path.join(debug_folder, str(pred), "{}.png".format(nb_hits))
                image_pil.save(path)
                nb_hits += 1

            if should_hit and (current_time - last_hit_time) > MIN_HIT_DELAY:
                delay_ms = hit_ante if pred in (2, 9) else 0
                last_hit_time = current_time

                # Key press em thread separada: loop principal não bloqueia durante o delay
                threading.Thread(target=_perform_hit, args=(delay_ms,), daemon=True).start()

                if debug_option == debug_options[2]:
                    path = os.path.join(debug_folder, str(pred), "hit_{}.png".format(nb_hits))
                    image_pil.save(path)
                    nb_hits += 1

                yield skip(), image_pil, probs
                continue

            # Compute fps - atualiza a cada 0.5s para responsividade
            t_diff = current_time - last_fps_update
            if t_diff > 0.5:
                fps = round(nb_frames / t_diff, 1)
                yield fps, image_pil, skip()
                last_fps_update = current_time
                nb_frames = 0

            # Controle de FPS - sleep ultra-mínimo para máxima responsividade
            sleep_time = frame_time - (current_time - frame_start)
            if sleep_time > 0.001:
                sleep(sleep_time)
    finally:
        _winmm.timeEndPeriod(1)


if __name__ == "__main__":
    debug_folder = "saved_images"

    debug_options = [
        "None (default)",
        "Display the monitored frame (a 224x224 center-cropped image, displayed at 1fps) instead of last hit skill check frame. Useful to check the monitored screen",
        "Save hit skill check frames in {}/".format(debug_folder),
        "Save all skill check frames in {}/ (will impact fps)".format(debug_folder)
    ]

    fps_info = "Number of frames per second the AI model analyses the monitored frame. Check The GitHub FAQ for more details and requirements."
    devices = ["CPU (default)", "GPU"]

    model_files = [f for f in os.listdir() if f.endswith(".onnx") or f.endswith(".engine")]
    if not model_files:
        model_files = ["model.onnx"]  

    with (Blocks(title="Recruta") as webui):
        Markdown("<h1 style='text-align: center;'>Recruta</h1>", elem_id="title")

        with Row():
            with Column(variant="panel"):
                with Column(variant="panel"):
                    Markdown("AI inference settings")
                    ai_model_path = Dropdown(
                        choices=model_files,  
                        value=model_files[0], 
                        label="Filepath of the AI model (ONNX or TensorRT Engine)"
                    )
                    device = Radio(choices=devices, value=devices[0], label="Device the AI model will use")
                with Column(variant="panel"):
                    Markdown("Debug options - for debugging or analytics")
                    debug_option = Dropdown(choices=debug_options, value=debug_options[0], label="Debugging selection")
                with Column(variant="panel"):
                    Markdown("Features options")
                    hit_ante = Slider(
                        minimum=0, maximum=100, step=5, value=25,
                        label="Ante-frontier delay (ms)",
                        info="Delay ao detectar ante-frontier para acertar na zona perfeita (teste: 20-35ms)"
                    )
                    confidence_threshold = Slider(
                        minimum=0.50, maximum=0.95, step=0.05, value=0.65,
                        label="Confidence Threshold",
                        info="Confiança mínima para detectar skill check (padrão: 0.65)"
                    )
                    cpu_stress = Radio(
                        label="CPU Performance Mode",
                        choices=["low", "normal", "high"],
                        value="normal",
                        info="Low: ~30 FPS, economiza CPU | Normal: ~60 FPS | High: ~120 FPS, máximo desempenho"
                    )
                with Column():
                    run_button = Button("RUN", variant="primary")
                    stop_button = Button("STOP", variant="stop")

            with Column(variant="panel"):
                fps = Number(label="AI model FPS", info=fps_info, interactive=False)
                image_pil = Image(label="Last hit skill check frame", height=224, interactive=False)
                probs = Label(label="Skill check recognition")

        monitoring = run_button.click(
            fn=monitor, 
            inputs=[ai_model_path, device, debug_option, hit_ante, cpu_stress, confidence_threshold], 
            outputs=[fps, image_pil, probs]
        )
        stop_button.click(fn=None, inputs=None, outputs=None, cancels=[monitoring])

    webui.launch()
