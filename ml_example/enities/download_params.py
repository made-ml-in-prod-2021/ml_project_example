from typing import List

from dataclasses import dataclass, field


@dataclass()
class DownloadParams:
    paths: List[str]
    output_folder: str
    s3_bucket: str = field(default="for-dvc")
