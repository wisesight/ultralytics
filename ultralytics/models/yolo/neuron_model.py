from ultralytics.engine.model_neuron import Model
from ultralytics.models import yolo


class NeuronYOLO(Model):
    def __init__(self, model="yolov8n.pt", task=None, verbose=False):
        super().__init__(model, task, verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "predictor": yolo.detect.NeuronDetectionPredictor,
            },
        }
