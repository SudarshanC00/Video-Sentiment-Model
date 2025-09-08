import logging
import boto3
from sagemaker.session import Session
from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig

# logging.basicConfig(level=logging.DEBUG)

def start_training():

    # sts = boto3.client('sts')
    # print("AWS Caller Identity:", sts.get_caller_identity())
    # boto_sess = boto3.session.Session()  
    # print("Boto3 Default Region:", boto_sess.region_name)
    # boto3.setup_default_session(region_name='us-east-1')  # Set your desired region here
    # boto_sess = boto3.session.Session(region_name="us-east-1")
    # sm_session = Session(boto_session=boto_sess)
    # # If you need to force the region, you can do:
    # # boto_sess = boto3.session.Session(region_name="us-east-2")
    # # sm_session = Session(boto_session=boto_sess)
    # # Otherwise, just let SageMaker pick up boto_sess by default:
    # sm_session = Session(boto_session=boto_sess)

    tensorboard_config = TensorBoardOutputConfig(
        s3_output_path='s3://video-sentiment-saas/tensorboard',
        container_local_output_path='/opt/ml/output/tensorboard'
    )

    estimator = PyTorch(
        entry_point = 'train.py',
        source_dir='training',
        role='arn:aws:iam::396608810756:role/sentiment-analysis-execution-role',
        framework_version='2.5.1',
        py_version='py311',
        instance_count=1,
        instance_type='ml.g5.xlarge',
        hyperparameters={
            'batch_size': 32,
            'epochs': 25
        },
        tensorboard_config=tensorboard_config,
        output_path='s3://video-sentiment-saas/model'
        # sagemaker_session=sm_session
    )

    # Start training
    estimator.fit({
        "training": "s3://video-sentiment-saas/dataset/train",
        "validation": "s3://video-sentiment-saas/dataset/dev",
        "test": "s3://video-sentiment-saas/dataset/test"
    },
    wait = True,
    logs=True) 


if __name__ == "__main__":
    start_training()