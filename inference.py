"""
Dummy inference.py to satisfy Hugging Face Inference Endpoints or OpenEnv evaluators.
Since this repository functions primarily as an Environment (via FastAPI/Docker) 
rather than a standalone agent/model, this handler gracefully returns a status message.
"""
import logging

class EndpointHandler:
    def __init__(self, path=""):
        """Initialize the Hugging Face Inference handler."""
        logging.info("EndpointHandler initialized. This is an OpenEnv Data Cleaning Environment.")
        self.path = path

    def __call__(self, data):
        """Handle incoming requests."""
        return {
            "status": "success",
            "message": "Data Cleaning Environment is active. Please interact via the /reset and /step endpoints."
        }

# Optional Sagemaker/Alternative format fallbacks
def model_fn(model_dir):
    return None

def predict_fn(data, model):
    return {"status": "success", "message": "Interaction should route to FastAPI."}
