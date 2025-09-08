from sagemaker.pytorch import PyTorchModel
import sagemaker

def deploy_endpoint():
    sagemaker.Session()
    role = "arn:aws:iam::396608810756:role/sentiment-analysis-deploy-endpoint-role"

    model_uri = "s3://video-sentiment-saas/inference/model.tar.gz"

    model = PyTorchModel(
        model_data = model_uri,
        role = role,
        py_version = "py3",
        entry_point = "inference.py",
        source_dir=".",
        name = "sentiment-analysis-endpoint"
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.g5.xlarge",
        endpoint_name="sentiment-analysis-endpoint"
    )


if __name__ == "__main__":
    deploy_endpoint()