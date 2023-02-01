import argparse
import logging
from servicefoundry import Build, PythonBuild, Service, Resources

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--workspace_fqn", required=True, type=str)
args = parser.parse_args()

image=Build(
	build_spec=PythonBuild(
		command="python app.py",
		requirements_path="requirements.txt",
	)
)

service = Service(
	name="clipseg_deployment",
	image=image,
	ports=[{"port": 8080}],
	resources=Resources(memory_limit=1500, memory_request=1000),
)
service.deploy(workspace_fqn=args.workspace_fqn)
