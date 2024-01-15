import json
import sys

from models.matrix_factorization import MFJob
from util.job import ConfigurableJob

job_classes: dict[str, type[ConfigurableJob]] = {
    "MFJob": MFJob,
}

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run.py <config_json_file>")
        sys.exit(1)

    json_file = sys.argv[1]
    with open(file=json_file, mode="r") as file:
        data = json.load(file)

    job_type = data.pop("job", None)

    job = job_classes[job_type].from_json(data=data)
    job.run()
