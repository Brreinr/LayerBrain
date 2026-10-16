"""Datasets vectoriales y codificadores sensoriales para redes de impulsos."""

import importlib
import wave
from pathlib import Path

import numpy as np

from .errors import DatasetError, IncompatibleDimensionsError, UnknownLabelError
from .neurons import (
    CochlearAuditoryLayer,
    GustatoryLayer,
    OlfactoryLayer,
    RetinalVisualLayer,
    SomatosensoryLayer,
    TextSensoryLayer,
)


class SpikeDataset:
    """Pares de vectores de entrada y etiquetas, preparados para ``fit``.

    Args:
        samples: secuencia de pares ``(vector_de_entrada, etiqueta)``.
        classes: etiquetas permitidas, en el orden de las neuronas de salida.
    """

    def __init__(self, samples, classes, prediction_samples=()):
        if isinstance(classes, str):
            raise DatasetError("`classes` debe ser una secuencia de etiquetas, no un texto.")
        try:
            self.classes = tuple(classes)
        except TypeError as exc:
            raise DatasetError("`classes` debe ser una secuencia de etiquetas.") from exc
        if len(self.classes) < 2:
            raise DatasetError("Declara al menos dos clases en `classes`.")
        if any(not isinstance(label, str) or not label for label in self.classes):
            raise DatasetError("Cada nombre de clase debe ser texto no vacío.")
        if len(set(self.classes)) != len(self.classes):
            raise DatasetError("Las clases deben ser únicas.")

        if samples is None:
            raise DatasetError("`samples` no puede ser None.")
        try:
            sample_list = list(samples)
        except TypeError as exc:
            raise DatasetError("`samples` debe ser una secuencia de pares.") from exc
        if not sample_list:
            raise DatasetError("El dataset está vacío; añade al menos un ejemplo.")

        self.class_to_index = {label: index for index, label in enumerate(self.classes)}
        self.features = []
        self.targets = []
        expected_size = None

        for sample_index, sample in enumerate(sample_list):
            if not isinstance(sample, (tuple, list)) or len(sample) != 2:
                raise DatasetError(
                    f"El ejemplo {sample_index} debe tener formato (entrada, etiqueta)."
                )
            raw_features, label = sample
            try:
                features = np.asarray(raw_features, dtype=float)
            except (TypeError, ValueError) as exc:
                raise DatasetError(
                    f"La entrada del ejemplo {sample_index} debe contener números."
                ) from exc
            if features.ndim != 1 or features.size == 0:
                raise DatasetError(
                    f"La entrada del ejemplo {sample_index} debe ser un vector 1D no vacío."
                )
            if not np.all(np.isfinite(features)):
                raise DatasetError(
                    f"La entrada del ejemplo {sample_index} contiene NaN o infinito."
                )
            if expected_size is None:
                expected_size = features.size
            elif features.size != expected_size:
                raise IncompatibleDimensionsError(
                    f"El ejemplo {sample_index} tiene {features.size} valores; "
                    f"los anteriores tienen {expected_size}."
                )
            if not isinstance(label, str) or label not in self.class_to_index:
                raise UnknownLabelError(
                    f"Etiqueta desconocida {label!r} en el ejemplo {sample_index}. "
                    f"Clases válidas: {', '.join(self.classes)}."
                )
            self.features.append(features.copy())
            self.targets.append(self.class_to_index[label])

        self.features = np.stack(self.features)
        self.targets = np.asarray(self.targets, dtype=int)
        self.input_size = int(expected_size)
        self.sensor_signature = None
        self.prediction_inputs = self._validate_prediction_samples(
            prediction_samples, self.input_size
        )

    @staticmethod
    def _validate_prediction_samples(samples, expected_size):
        try:
            items = list(samples)
        except TypeError as exc:
            raise DatasetError("`prediction_samples` debe ser una secuencia de entradas.") from exc
        inputs = []
        for index, item in enumerate(items):
            try:
                values = np.asarray(item, dtype=float)
            except (TypeError, ValueError) as exc:
                raise DatasetError(
                    f"La entrada de predicción {index} debe contener números."
                ) from exc
            if values.shape != (expected_size,) or not np.all(np.isfinite(values)):
                raise IncompatibleDimensionsError(
                    f"La entrada de predicción {index} debe ser un vector finito "
                    f"de tamaño {expected_size}."
                )
            inputs.append(values.copy())
        return tuple(inputs)

    def __len__(self):
        return len(self.targets)


class SensoryDataset(SpikeDataset):
    """Base para datos crudos que se codifican con su capa sensorial."""

    sensor_layer_type = None
    modality = "sensorial"

    def __init__(
        self,
        samples,
        classes,
        sensor_parameters=None,
        encoding_parameters=None,
        vocabulary=None,
        prediction_samples=(),
        file_parameters=None,
    ):
        if self.sensor_layer_type is None:
            raise DatasetError(
                "SensoryDataset es una clase base; usa TextDataset, AudioDataset, "
                "VisionDataset, TasteDataset, SmellDataset o TouchDataset."
            )
        self.sensor_parameters = self._validate_parameters(
            sensor_parameters, "sensor_parameters"
        )
        self.encoding_parameters = self._validate_parameters(
            encoding_parameters, "encoding_parameters"
        )
        self.file_parameters = self._validate_parameters(
            file_parameters, "file_parameters"
        )

        try:
            raw_samples = list(samples)
        except TypeError as exc:
            raise DatasetError("`samples` debe ser una secuencia de pares.") from exc
        if not raw_samples:
            raise DatasetError("El dataset está vacío; añade al menos un ejemplo.")
        if any(not isinstance(sample, (tuple, list)) or len(sample) != 2 for sample in raw_samples):
            raise DatasetError("Cada ejemplo debe tener formato (entrada, etiqueta).")

        raw_inputs = [
            self._load_sample_input(sample[0], f"ejemplo {index}")
            for index, sample in enumerate(raw_samples)
        ]
        labels = [sample[1] for sample in raw_samples]
        if isinstance(prediction_samples, (str, Path)):
            prediction_samples = (prediction_samples,)
        try:
            prediction_samples = list(prediction_samples)
        except TypeError as exc:
            raise DatasetError(
                "`prediction_samples` debe ser una secuencia de entradas."
            ) from exc
        try:
            raw_prediction_inputs = [
                self._load_sample_input(item, f"predicción {index}")
                for index, item in enumerate(prediction_samples)
            ]
        except TypeError as exc:
            raise DatasetError(
                "`prediction_samples` debe ser una secuencia de entradas."
            ) from exc
        self.vocabulary = self._make_vocabulary(raw_inputs, vocabulary)
        shape = self._get_input_shape(raw_inputs[0])
        self.sensor_shape = shape
        try:
            self.encoder_layer = self._build_sensor_layer()
        except (TypeError, ValueError) as exc:
            raise DatasetError(
                f"Parámetros inválidos para la capa sensorial de {self.modality}: {exc}"
            ) from exc

        encoded_samples = []
        for index, (raw_input, label) in enumerate(zip(raw_inputs, labels)):
            if self._get_input_shape(raw_input) != shape:
                raise IncompatibleDimensionsError(
                    f"La entrada {index} tiene forma {self._get_input_shape(raw_input)}; "
                    f"se esperaba {shape}."
                )
            try:
                encoded = self.encoder_layer.encode(
                    raw_input, **self.encoding_parameters
                )
            except (TypeError, ValueError) as exc:
                raise DatasetError(
                    f"No se pudo codificar la entrada {index} como {self.modality}: {exc}"
                ) from exc
            encoded = np.asarray(encoded, dtype=float)
            if encoded.ndim != 1 or encoded.size == 0 or not np.all(np.isfinite(encoded)):
                raise DatasetError(
                    f"La codificación de la entrada {index} debe ser un vector "
                    "finito y no vacío."
                )
            encoded_samples.append((encoded, label))

        encoded_prediction_samples = []
        for index, raw_input in enumerate(raw_prediction_inputs):
            if self._get_input_shape(raw_input) != shape:
                raise IncompatibleDimensionsError(
                    f"La entrada de predicción {index} tiene forma "
                    f"{self._get_input_shape(raw_input)}; se esperaba {shape}."
                )
            try:
                encoded = self.encoder_layer.encode(
                    raw_input, **self.encoding_parameters
                )
            except (TypeError, ValueError) as exc:
                raise DatasetError(
                    f"No se pudo codificar la entrada de predicción {index} "
                    f"como {self.modality}: {exc}"
                ) from exc
            encoded_prediction_samples.append(np.asarray(encoded, dtype=float))

        super().__init__(encoded_samples, classes, encoded_prediction_samples)
        self.raw_inputs = tuple(raw_inputs)
        self.prediction_inputs = tuple(raw_prediction_inputs)
        self.sensor_signature = (
            type(self).__name__,
            self.sensor_shape,
            repr(sorted(self.sensor_parameters.items())),
            repr(sorted(self.encoding_parameters.items())),
            repr(sorted(self.file_parameters.items())),
            self.vocabulary,
        )
        if isinstance(self.encoder_layer, SomatosensoryLayer):
            self.encoder_layer.prev_pressure.fill(0.0)

    @staticmethod
    def _validate_parameters(parameters, name):
        if parameters is None:
            return {}
        if not isinstance(parameters, dict):
            raise DatasetError(f"`{name}` debe ser un diccionario.")
        return dict(parameters)

    def _make_vocabulary(self, raw_inputs, vocabulary):
        if vocabulary is not None:
            raise DatasetError("`vocabulary` solo se admite en `TextDataset`.")
        if not raw_inputs:
            raise DatasetError("El dataset debe incluir al menos una entrada.")
        return None

    def _load_input(self, raw_input):
        if not isinstance(raw_input, (str, Path)):
            return raw_input
        path = Path(raw_input)
        extension = path.suffix.lower()
        if not path.is_file():
            if (
                self.sensor_layer_type is TextSensoryLayer
                and extension not in (".txt", ".pdf", ".doc", ".docx")
            ):
                return str(raw_input)
            raise DatasetError(f"No existe el archivo de entrada: {path}")
        if self.sensor_layer_type is TextSensoryLayer:
            if extension == ".txt":
                return path.read_text(encoding="utf-8-sig")
            if extension == ".pdf":
                try:
                    reader = importlib.import_module("pypdf").PdfReader(str(path))
                except ModuleNotFoundError as exc:
                    raise DatasetError(
                        "Leer PDF requiere la dependencia opcional `pypdf` "
                        "(instálala con `python -m pip install pypdf`)."
                    ) from exc
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            if extension == ".docx":
                try:
                    document = importlib.import_module("docx").Document(str(path))
                except ModuleNotFoundError as exc:
                    raise DatasetError(
                        "Leer DOCX requiere la dependencia opcional `python-docx` "
                        "(instálala con `python -m pip install python-docx`)."
                    ) from exc
                return "\n".join(paragraph.text for paragraph in document.paragraphs)
            if extension == ".doc":
                raise DatasetError(
                    "El formato `.doc` antiguo no es compatible directamente; "
                    "conviértelo a `.docx` o `.txt`."
                )
            raise DatasetError("Texto admite archivos `.txt`, `.pdf` y `.docx`.")

        if self.sensor_layer_type is RetinalVisualLayer:
            if extension in (".jpg", ".jpeg", ".png"):
                try:
                    image_module = importlib.import_module("PIL.Image")
                except ModuleNotFoundError as exc:
                    raise DatasetError(
                        "Leer imágenes requiere Pillow (`python -m pip install Pillow`)."
                    ) from exc
                image_size = self.file_parameters.get("image_size", (32, 32))
                if (
                    not isinstance(image_size, (tuple, list))
                    or len(image_size) != 2
                    or any(not isinstance(size, int) or size <= 0 for size in image_size)
                ):
                    raise DatasetError(
                        "`file_parameters['image_size']` debe ser (alto, ancho) positivo."
                    )
                image = image_module.open(path).convert("L")
                image = image.resize((image_size[1], image_size[0]))
                return np.asarray(image, dtype=float) / 255.0
            if extension == ".mp4":
                try:
                    cv2 = importlib.import_module("cv2")
                except ModuleNotFoundError as exc:
                    raise DatasetError(
                        "Leer MP4 requiere OpenCV (`python -m pip install opencv-python`)."
                    ) from exc
                capture = cv2.VideoCapture(str(path))
                if not capture.isOpened():
                    raise DatasetError(f"No se pudo abrir el vídeo MP4: {path}")
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_index = max(0, frame_count // 2)
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = capture.read()
                capture.release()
                if not success:
                    raise DatasetError(f"No se pudo leer un fotograma de: {path}")
                image_size = self.file_parameters.get("image_size", (32, 32))
                if (
                    not isinstance(image_size, (tuple, list))
                    or len(image_size) != 2
                    or any(not isinstance(size, int) or size <= 0 for size in image_size)
                ):
                    raise DatasetError(
                        "`file_parameters['image_size']` debe ser (alto, ancho) positivo."
                    )
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (image_size[1], image_size[0]))
                return resized.astype(float) / 255.0
            raise DatasetError("Visión admite `.jpg`, `.jpeg`, `.png` y `.mp4`.")

        if self.sensor_layer_type is CochlearAuditoryLayer:
            if extension == ".wav":
                with wave.open(str(path), "rb") as audio_file:
                    channels = audio_file.getnchannels()
                    sample_width = audio_file.getsampwidth()
                    frames = audio_file.readframes(audio_file.getnframes())
                if sample_width == 1:
                    samples = np.frombuffer(frames, dtype=np.uint8).astype(float) - 128.0
                elif sample_width == 2:
                    samples = np.frombuffer(frames, dtype="<i2").astype(float)
                elif sample_width == 4:
                    samples = np.frombuffer(frames, dtype="<i4").astype(float)
                elif sample_width == 3:
                    raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
                    values = (
                        raw[:, 0].astype(np.int32)
                        | (raw[:, 1].astype(np.int32) << 8)
                        | (raw[:, 2].astype(np.int32) << 16)
                    )
                    samples = ((values ^ 0x800000) - 0x800000).astype(float)
                else:
                    raise DatasetError(f"El WAV usa una profundidad no compatible: {sample_width * 8} bits.")
                if channels > 1:
                    samples = samples.reshape(-1, channels).mean(axis=1)
                return self._audio_spectrum(samples)
            if extension == ".mp3":
                try:
                    librosa = importlib.import_module("librosa")
                except ModuleNotFoundError as exc:
                    raise DatasetError(
                        "Leer MP3 requiere la dependencia opcional `librosa` "
                        "(instálala con `python -m pip install librosa`)."
                    ) from exc
                samples, _ = librosa.load(str(path), sr=None, mono=True)
                return self._audio_spectrum(samples)
            raise DatasetError("Audio admite archivos `.wav` y `.mp3`.")

        if extension in (".npy", ".csv"):
            if extension == ".npy":
                return np.load(path, allow_pickle=False)
            return np.loadtxt(path, delimiter=",")
        raise DatasetError(
            f"{self.modality.capitalize()} acepta vectores numéricos o archivos `.npy`/`.csv`."
        )

    def _load_sample_input(self, raw_input, description):
        try:
            return self._load_input(raw_input)
        except DatasetError:
            raise
        except (OSError, EOFError, wave.Error) as exc:
            raise DatasetError(f"No se pudo leer el archivo de {description}: {exc}") from exc

    def _audio_spectrum(self, samples):
        bands = self.file_parameters.get("audio_bands", 64)
        if not isinstance(bands, int) or bands <= 0:
            raise DatasetError("`file_parameters['audio_bands']` debe ser entero positivo.")
        if samples.size == 0:
            raise DatasetError("El archivo de audio no contiene muestras.")
        spectrum = np.abs(np.fft.rfft(samples.astype(float)))
        bins = np.array_split(spectrum[1:], bands)
        band_energy = np.asarray(
            [float(np.mean(values)) if values.size else 0.0 for values in bins]
        )
        peak = float(np.max(band_energy))
        return band_energy / peak if peak > 0.0 else band_energy

    def _get_input_shape(self, raw_input):
        try:
            values = np.asarray(raw_input, dtype=float)
        except (TypeError, ValueError) as exc:
            raise DatasetError(
                f"Las entradas de {self.modality} deben ser numéricas."
            ) from exc
        if values.ndim != self.expected_dimensions or values.size == 0:
            raise DatasetError(
                f"Cada entrada de {self.modality} debe tener "
                f"{self.expected_dimensions} dimensiones y no estar vacía."
            )
        if not np.all(np.isfinite(values)):
            raise DatasetError(f"Una entrada de {self.modality} contiene NaN o infinito.")
        return values.shape

    @property
    def expected_dimensions(self):
        return 1

    def _build_sensor_layer(self, parameters=None):
        config = dict(self.sensor_parameters)
        if parameters:
            config.update(parameters)
        if self.sensor_layer_type is TextSensoryLayer:
            return self.sensor_layer_type(
                len(self.vocabulary), vocabulary=self.vocabulary, **config
            )
        if self.sensor_layer_type is RetinalVisualLayer:
            height, width = self.sensor_shape
            return self.sensor_layer_type(width=width, height=height, **config)
        return self.sensor_layer_type(self._sensor_size(), **config)

    def _sensor_size(self):
        return int(np.prod(self.sensor_shape))

    def build_sensor_layer(self, parameters=None):
        """Crea una instancia independiente para conectar a la red."""
        try:
            return self._build_sensor_layer(parameters)
        except (TypeError, ValueError) as exc:
            raise DatasetError(
                f"Parámetros inválidos para la capa sensorial de {self.modality}: {exc}"
            ) from exc

    def encode_input(self, raw_input):
        """Transforma una entrada cruda con el mismo codificador usado en entrenamiento."""
        if self._get_input_shape(raw_input) != self.sensor_shape:
            raise IncompatibleDimensionsError(
                f"La entrada de {self.modality} tiene forma "
                f"{self._get_input_shape(raw_input)}; se esperaba {self.sensor_shape}."
            )
        try:
            encoded = np.asarray(
                self.encoder_layer.encode(raw_input, **self.encoding_parameters),
                dtype=float,
            )
        except (TypeError, ValueError) as exc:
            raise DatasetError(
                f"No se pudo codificar la entrada como {self.modality}: {exc}"
            ) from exc
        if encoded.shape != (self.input_size,) or not np.all(np.isfinite(encoded)):
            raise DatasetError(
                f"La codificación de {self.modality} produjo una forma o valores inválidos."
            )
        return encoded


class TextDataset(SensoryDataset):
    """Texto tokenizado por palabras/símbolos y codificado como presencia de tokens."""

    sensor_layer_type = TextSensoryLayer
    modality = "texto"

    def _make_vocabulary(self, raw_inputs, vocabulary):
        if any(not isinstance(text, str) for text in raw_inputs):
            raise DatasetError("Cada entrada de texto debe ser una cadena.")
        if vocabulary is None:
            tokens = {
                token
                for text in raw_inputs
                for token in TextSensoryLayer.tokenize(text)
            }
            tokens.add("<UNK>")
            vocabulary = sorted(tokens)
        elif isinstance(vocabulary, str):
            raise DatasetError("`vocabulary` debe ser una secuencia de tokens.")
        else:
            try:
                vocabulary = tuple(vocabulary)
            except TypeError as exc:
                raise DatasetError("`vocabulary` debe ser una secuencia de tokens.") from exc
            if any(not isinstance(token, str) or not token for token in vocabulary):
                raise DatasetError("Cada token del vocabulario debe ser texto no vacío.")
            if len(set(vocabulary)) != len(vocabulary):
                raise DatasetError("Los tokens de `vocabulary` deben ser únicos.")
            if "<UNK>" not in vocabulary:
                vocabulary = vocabulary + ("<UNK>",)
        if not vocabulary:
            raise DatasetError("El vocabulario de texto no puede estar vacío.")
        return tuple(vocabulary)

    def _get_input_shape(self, raw_input):
        if not isinstance(raw_input, str) or not TextSensoryLayer.tokenize(raw_input):
            raise DatasetError("Cada entrada de texto debe contener al menos un token.")
        return (len(self.vocabulary),)


class AudioDataset(SensoryDataset):
    """Espectros de audio normalizados, codificados por bandas de frecuencia."""

    sensor_layer_type = CochlearAuditoryLayer
    modality = "audio"


class VisionDataset(SensoryDataset):
    """Imágenes 2D normalizadas entre 0 y 1, codificadas por la capa retinal."""

    sensor_layer_type = RetinalVisualLayer
    modality = "visión"

    @property
    def expected_dimensions(self):
        return 2


class TasteDataset(SensoryDataset):
    """Perfiles de intensidad de canales gustativos normalizados entre 0 y 1."""

    sensor_layer_type = GustatoryLayer
    modality = "sabor"


class SmellDataset(SensoryDataset):
    """Perfiles químicos de glomérulos olfativos normalizados entre 0 y 1."""

    sensor_layer_type = OlfactoryLayer
    modality = "olor"


class TouchDataset(SensoryDataset):
    """Mapas de presión normalizados entre 0 y 1, codificados por receptores."""

    sensor_layer_type = SomatosensoryLayer
    modality = "tacto"

    @property
    def expected_dimensions(self):
        return 1
