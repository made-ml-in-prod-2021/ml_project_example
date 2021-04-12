from boto3 import client

if __name__ == '__main__':
    s3 = client("s3")
    s3.download_file("for-dvc", "train.csv", "../data/raw/train.csv")
    s3.download_file("for-dvc", "test.csv", "../data/raw/test.csv")