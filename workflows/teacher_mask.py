import fiftyone.zoo as foz
import logging

class MaskTeacher:
    def __init__(self, dataset, dataset_info, model_name, task_type):
        print(dataset, dataset_info, model_name)
        self.dataset = dataset
        self.dataset_info = dataset_info
        self.model_name = model_name
        self.task_type = task_type

    def run_inference(self):
        if self.task_type == "semantic_segmentation":
            self._run_semantic_segmentation()
        elif self.task_type == "depth_estimation":
            self._run_depth_estimation()
        else:
            logging.error(f"Task type {self.task_type} not supported")

    def _run_semantic_segmentation(self):  # loading the model as well
        if self.model_name == "sam2":
            logging.info(f"Loading model {self.model_name}")
            model = foz.load_zoo_model("segment-anything-2.1-hiera-base-plus-image")
            self.dataset.apply_model(model, label_field="auto")

    def _run_depth_estimation(self, model):
        pass