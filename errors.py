"""Excepciones públicas y entendibles de LayerBrain."""


class LayerBrainError(Exception):
    """Clase base para errores que el usuario puede corregir en su código."""


class DatasetError(LayerBrainError, ValueError):
    """El dataset está vacío, mal formado o tiene etiquetas incompatibles."""


class ArchitectureError(LayerBrainError, ValueError):
    """La arquitectura contiene dimensiones o parámetros incompatibles."""


class IncompatibleDimensionsError(ArchitectureError):
    """El tamaño de una entrada no coincide con el tamaño esperado."""


class UnknownLabelError(DatasetError):
    """Una etiqueta no está declarada entre las clases del modelo."""


class TrainingError(LayerBrainError, ValueError):
    """La configuración o los datos no permiten entrenar el modelo."""


class PredictionError(LayerBrainError, ValueError):
    """La entrada o el estado del modelo no permiten realizar una predicción."""


class PersistenceError(LayerBrainError, OSError):
    """No se pudo guardar o cargar el estado del modelo."""
