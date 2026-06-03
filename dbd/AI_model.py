import numpy as np
from PIL import Image
from mss import mss
import onnxruntime as ort
import atexit
import sys
import threading
from time import time
from pyautogui import size as pyautogui_size
from collections import deque

# MSS uses threading.local() internally — one instance per thread required
_mss_local = threading.local()

try:
    import torch
    import tensorrt as trt
except ImportError as e:
    print(e)


# Cache monitor attributes globally for better performance
_MONITOR_CACHE = None

def get_monitor_attributes():
    """Get monitor attributes with caching for better performance."""
    global _MONITOR_CACHE
    
    if _MONITOR_CACHE is not None:
        return _MONITOR_CACHE
    
    width, height = pyautogui_size()
    object_size_h_ratio = 224 / 1080  # Modelo espera 224x224
    object_size = int(object_size_h_ratio * height)

    _MONITOR_CACHE = {
        "top": height // 2 - object_size // 2,
        "left": width // 2 - object_size // 2,
        "width": object_size,
        "height": object_size
    }
    return _MONITOR_CACHE

class AI_model:
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    # Opening animation is ~2-3 frames (33-50ms at 60fps). 50ms covers it without eating into the hit zone.
    # 120ms was too long — it suppressed the entire hit zone on fast/perk-triggered skill checks.
    _ENTRY_COOLDOWN = 0.05

    pred_dict = {
        0: {"desc": "None", "hit": False},
        1: {"desc": "repair-heal (great)", "hit": True},
        2: {"desc": "repair-heal (ante-frontier)", "hit": True},  # Aperta com delay ajustável
        3: {"desc": "repair-heal (out)", "hit": False},
        4: {"desc": "full white (great)", "hit": True},
        5: {"desc": "full white (out)", "hit": False},
        6: {"desc": "full black (great)", "hit": True},
        7: {"desc": "full black (out)", "hit": False},
        8: {"desc": "wiggle (great)", "hit": True},
        9: {"desc": "wiggle (frontier)", "hit": True},   # fires with ante-frontier delay, same as class 2
        10: {"desc": "wiggle (out)", "hit": False}
    }

    def __init__(self, model_path="model.onnx", use_gpu=False, nb_cpu_threads=None):
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.nb_cpu_threads = nb_cpu_threads
        self.monitor = get_monitor_attributes()
        self.crop_size = 224  # Modelo espera 224x224
        
        self.context = None
        self.engine = None
        
        # Pre-allocate arrays for faster processing
        self.prealloc_array = None
        self._logits_buffer = None
        
        # Sistema de validação simples - lógica original que funcionava
        self.prediction_history = deque(maxlen=2)
        self.last_valid_pred = 0

        # Entry cooldown: evita hits falsos durante a animação de abertura do círculo
        self._consecutive_none = 0   # frames consecutivos com pred==0
        self._skill_check_start_time = None

        if model_path.endswith(".engine"):
            assert self.use_gpu, "TensorRT engine model requires GPU mode"
            assert "torch" in sys.modules, "TensorRT engine model requires torch lib"
            assert "tensorrt" in sys.modules, "TensorRT engine model requires tensorrt lib"
            self.load_tensorrt()
        else:
            self.load_onnx()

        atexit.register(self.cleanup)

    def cleanup(self):
        if getattr(self, 'is_tensorrt', False):
            del self.context
            del self.engine
            torch.cuda.empty_cache()

    def grab_screenshot(self):
        if not hasattr(_mss_local, 'sct'):
            _mss_local.sct = mss()
        return _mss_local.sct.grab(self.monitor)

    def screenshot_to_pil(self, screenshot):
        """Converte screenshot para PIL com otimizações de performance."""
        # Conversão otimizada diretamente para RGB
        pil_image = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        
        # Resize apenas se necessário, usando BILINEAR (mais rápido)
        if pil_image.width != self.crop_size or pil_image.height != self.crop_size:
            pil_image = pil_image.resize((self.crop_size, self.crop_size), Image.Resampling.BILINEAR)
        
        return pil_image

    def pil_to_numpy(self, image_pil):
        """Converte PIL para numpy com performance otimizada e buffers pré-alocados."""
        # Pre-allocate arrays for faster processing if not already done
        if self.prealloc_array is None or self.prealloc_array.shape[2:] != (self.crop_size, self.crop_size):
            self.prealloc_array = np.zeros((1, 3, self.crop_size, self.crop_size), dtype=np.float32)
        
        # Conversão otimizada usando numpy array direto
        img = np.asarray(image_pil, dtype=np.float32, order='C') / 255.0
        
        # Transposição otimizada (HWC para CHW) e normalização
        img = np.transpose(img, (2, 0, 1))
        img = (img - self.MEAN[:, None, None]) / self.STD[:, None, None]
        
        self.prealloc_array[0] = img
        
        return self.prealloc_array

    def softmax(self, x):
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        sum_exp = np.sum(exp_x)
        return exp_x / sum_exp

    def load_onnx(self):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.optimized_model_filepath = self.model_path + ".optimized"

        if not self.use_gpu and self.nb_cpu_threads is not None:
            sess_options.intra_op_num_threads = self.nb_cpu_threads
            sess_options.inter_op_num_threads = self.nb_cpu_threads

        if self.use_gpu:
            assert "torch" in sys.modules, "GPU mode requires torch lib"
            available_providers = ort.get_available_providers()
            preferred_execution_providers = ['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
            execution_providers = [p for p in preferred_execution_providers if p in available_providers]
        else:
            execution_providers = ["CPUExecutionProvider"]

        self.ort_session = ort.InferenceSession(
            self.model_path, providers=execution_providers, sess_options=sess_options
        )

        self.input_name = self.ort_session.get_inputs()[0].name
        self.input_dtype = self.ort_session.get_inputs()[0].type
        self.is_tensorrt = False

    def load_tensorrt(self):
        self.is_tensorrt = True
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(self.model_path, "rb") as f:
            engine_data = f.read()
            self.engine = runtime.deserialize_cuda_engine(engine_data)

        self.stream = torch.cuda.Stream()
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = self.allocate_buffers(self.engine)

    def allocate_buffers(self, engine):
        inputs, outputs, bindings = [], [], []

        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            tensor_shape = engine.get_tensor_shape(tensor_name)
            tensor_dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

            if -1 in tensor_shape:
                raise ValueError(f"Tensor '{tensor_name}' has a dynamic shape {tensor_shape}. Set static dimensions before inference!")

            size = trt.volume(tensor_shape)
            device_mem = torch.empty(size, dtype=torch.float32, device="cuda")
            host_mem = np.empty(size, dtype=tensor_dtype)

            bindings.append(device_mem.data_ptr())

            tensor_mode = engine.get_tensor_mode(tensor_name)
            tensor_info = {'host': host_mem, 'device': device_mem, 'name': tensor_name}

            if tensor_mode == trt.TensorIOMode.INPUT:
                inputs.append(tensor_info)
            else:
                outputs.append(tensor_info)

        return inputs, outputs, bindings

    def predict(self, image, confidence_threshold=0.65, require_consecutive=1):
        """
        Predição otimizada com validação de predições consecutivas.
        
        Args:
            image: Imagem PIL ou numpy array
            confidence_threshold: Limiar de confiança mínimo (0.65-0.75 recomendado para responsividade)
            require_consecutive: Número de predições consecutivas necessárias (1=rápido, 2=conservador)
        """
        if isinstance(image, np.ndarray):
            img_np = image
        else:
            img_np = self.pil_to_numpy(image)
            
        # Pre-allocate memory for results if needed
        if self._logits_buffer is None:
            self._logits_buffer = np.zeros((len(self.pred_dict),), dtype=np.float32)

        if self.is_tensorrt:
            torch.cuda.synchronize()
            torch.cuda.current_stream().wait_stream(self.stream)

            np.copyto(self.inputs[0]['host'], img_np.ravel())
            self.inputs[0]['device'].copy_(torch.tensor(self.inputs[0]['host'], dtype=torch.float32, device="cuda"))

            self.context.execute_v2(bindings=self.bindings)

            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream): 
                output_tensor = self.outputs[0]['device'].to("cpu", non_blocking=True)

            torch.cuda.current_stream().wait_stream(stream)

            self.outputs[0]['host'][:] = output_tensor.numpy()

            torch.cuda.synchronize()
            logits = np.squeeze(self.outputs[0]['host'])
        else:
            if self.input_dtype == "tensor(float)":
                img_np = img_np.astype(np.float32)
            elif self.input_dtype == "tensor(float16)":
                img_np = img_np.astype(np.float16)

            # Only convert dtype when necessary — prealloc_array is already float32
            if self.input_dtype == "tensor(float16)":
                img_np = img_np.astype(np.float16)

            ort_inputs = {self.input_name: img_np}
            logits = np.squeeze(self.ort_session.run(None, ort_inputs))

            # Use buffer pré-alocado para resultados
            np.copyto(self._logits_buffer, logits)
            logits = self._logits_buffer

        pred = int(np.argmax(logits))
        probs = self.softmax(logits)
        probs_dict = {self.pred_dict[i]["desc"]: probs[i] for i in range(len(probs))}

        # Get confidence of predicted class
        confidence = probs[pred]

        # --- Entry cooldown: bloqueia hits durante animação de abertura do círculo ---
        current_time = time()

        if pred == 0:
            self._consecutive_none += 1
            # Só confirma "sem skill check" após 2 frames consecutivos de classe 0
            if self._consecutive_none >= 2:
                self._skill_check_start_time = None
        else:
            self._consecutive_none = 0
            # Inicia cooldown na PRIMEIRA detecção de classe hit (sem depender de _last_none_time)
            if self.pred_dict[pred]["hit"] and self._skill_check_start_time is None:
                self._skill_check_start_time = current_time

        in_entry_cooldown = (
            self._skill_check_start_time is not None and
            (current_time - self._skill_check_start_time) < self._ENTRY_COOLDOWN
        )
        # -------------------------------------------------------------------------

        # Adicionar predição ao histórico
        self.prediction_history.append((pred, confidence))

        # Lógica simples e direta - o que funcionava antes
        should_hit = False

        if pred != 0 and confidence >= confidence_threshold and not in_entry_cooldown:
            # Validar apenas que a predição está acima do threshold
            if len(self.prediction_history) >= require_consecutive:
                recent_preds = [p[0] for p in list(self.prediction_history)[-require_consecutive:]]
                recent_confs = [p[1] for p in list(self.prediction_history)[-require_consecutive:]]

                # Predições devem ser similares e com boa confiança
                if (all(p == pred for p in recent_preds) and
                    all(c >= confidence_threshold for c in recent_confs)):
                    should_hit = self.pred_dict[pred]["hit"]
                    self.last_valid_pred = pred

        return pred, self.pred_dict[pred]["desc"], probs_dict, should_hit, confidence

    def check_provider(self):
        return "TensorRT" if self.is_tensorrt else self.ort_session.get_providers()[0]
