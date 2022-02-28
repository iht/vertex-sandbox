import argparse

from typing import List

import tfx.v1 as tfx
from tfx.extensions.google_cloud_big_query.example_gen.component import BigQueryExampleGen

from google.cloud import aiplatform


def create_pipeline(query: str,
                    pipeline_name: str,
                    pipeline_root: str,
                    beam_pipeline_args: List[str]) -> tfx.dsl.Pipeline:
    # Read data from BigQuery
    example_gen: BigQueryExampleGen = tfx.extensions.google_cloud_big_query.BigQueryExampleGen(query=query)

    stats_gen = tfx.components.StatisticsGen(examples=example_gen.outputs['examples'])

    schema_gen = tfx.components.SchemaGen(statistics=stats_gen.outputs["statistics"])

    validator = tfx.components.ExampleValidator(statistics=stats_gen.outputs['statistics'],
                                                schema=schema_gen.outputs["schema"])

    components = [example_gen, stats_gen, schema_gen, validator]

    pipeline = tfx.dsl.Pipeline(pipeline_name=pipeline_name,
                                pipeline_root=pipeline_root,
                                components=components,
                                beam_pipeline_args=beam_pipeline_args)

    return pipeline


def main(query: str,
         pipeline_name: str,
         pipeline_root: str,
         project_id: str,
         temp_location: str,
         region: str,
         service_account: str):
    # Beam options: project id and a temp location
    beam_args = [f"--project={project_id}", f"--temp_location={temp_location}"]

    p = create_pipeline(query=query,
                        pipeline_root=pipeline_root,
                        pipeline_name=pipeline_name,
                        beam_pipeline_args=beam_args)

    # Create the runner
    pipeline_definition = pipeline_name + "_pipeline.json"
    runner = tfx.orchestration.experimental.KubeflowV2DagRunner(
        config=tfx.orchestration.experimental.KubeflowV2DagRunnerConfig(),
        output_filename=pipeline_definition)

    runner.run(p)

    aiplatform.init(project=project_id, location=region)

    job = aiplatform.pipeline_jobs.PipelineJob(display_name=pipeline_name,
                                               template_path=pipeline_definition,
                                               enable_caching=True)  # only for convenience during development

    job.submit(service_account=service_account)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--query", required=True)
    parser.add_argument("--pipeline-name", required=True)
    parser.add_argument("--pipeline-root", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--temp-location", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--service-account", required=True)

    args = parser.parse_args()

    main(query=args.query,
         pipeline_name=args.pipeline_name,
         pipeline_root=args.pipeline_root,
         project_id=args.project_id,
         temp_location=args.temp_location,
         region=args.region,
         service_account=args.service_account)
