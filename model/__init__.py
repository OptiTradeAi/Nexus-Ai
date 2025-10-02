"""
Nexus AI - Módulo de Inteligência Artificial
Sistema de predição para opções binárias usando Transformer.
"""

from .model import NexusAI, NexusTransformer, ModelConfig, FeatureExtractor, create_nexus_model
from .infer import NexusInferenceEngine, InferenceConfig, create_inference_engine

__version__ = "1.0.0"
__author__ = "Manus AI"

__all__ = [
    'NexusAI',
    'NexusTransformer', 
    'ModelConfig',
    'FeatureExtractor',
    'create_nexus_model',
    'NexusInferenceEngine',
    'InferenceConfig', 
    'create_inference_engine'
]

