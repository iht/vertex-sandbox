import argparse

from typing import List

import tfx.v1 as tfx
from tfx.extensions.google_cloud_big_query.example_gen.component import BigQueryExampleGen

import tensorflow_model_analysis as tfma

from google.cloud import aiplatform

BATCH_SIZE = 4096
DATASET_SIZE = 284807


def create_pipeline(query: str,
                    pipeline_name: str,
                    pipeline_root: str,
                    beam_pipeline_args: List[str],
                    transform_file_gcs: str,
                    trainer_file_gcs: str,
                    region: str,
                    project_id: str,
                    temp_location: str) -> tfx.dsl.Pipeline:
    # Read data from BigQuery
    example_gen: BigQueryExampleGen = tfx.extensions.google_cloud_big_query.BigQueryExampleGen(query=query)

    stats_gen = tfx.components.StatisticsGen(examples=example_gen.outputs['examples'])

    schema_gen = tfx.components.SchemaGen(statistics=stats_gen.outputs["statistics"])

    validator = tfx.components.ExampleValidator(statistics=stats_gen.outputs['statistics'],
                                                schema=schema_gen.outputs["schema"])

    transform = tfx.components.Transform(examples=example_gen.outputs['examples'],
                                         schema=schema_gen.outputs['schema'],
                                         module_file=transform_file_gcs)

    vertex_config = {
        'project': project_id,
        'tensorboard': f'projects/237148598933/locations/{region}/tensorboards/3834463238985089024',
        'service_account': 'ihr-tensorboard@ihr-vertex-pipelines.iam.gserviceaccount.com',
        'base_output_directory': {
            'output_uri_prefix': temp_location,  # required for Tensorboard. THIS SHOULD NOT BE PROBABLY A TEMP FILE
        },
        'worker_pool_specs': [{'machine_spec': {'machine_type': 'n1-standard-4'},
                               'replica_count': 1,
                               'container_spec': {'image_uri': f'gcr.io/tfx-oss-public/tfx:{tfx.__version__}'}
                               }]
    }

    trainer = tfx.extensions.google_cloud_ai_platform.Trainer(
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],  # only for schema and other handy stuff
        train_args=tfx.proto.TrainArgs(num_steps=50),
        module_file=trainer_file_gcs,
        custom_config={
            tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY: True,
            tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY: region,
            tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY: vertex_config,
            "batch-size": BATCH_SIZE,
            "dataset-size": DATASET_SIZE
        })

    vertex_endpoint_config = {
        'project': project_id,
        'endpoint_name': "live-workshop",
        'machine_type': 'n1-standard-4'
    }
    container_image_uri = "europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-7:latest"

    pusher = tfx.extensions.google_cloud_ai_platform.Pusher(
        model=trainer.outputs["model"],
        custom_config={
            tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY: True,
            tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY: region,
            tfx.extensions.google_cloud_ai_platform.VERTEX_CONTAINER_IMAGE_URI_KEY: container_image_uri,
            tfx.extensions.google_cloud_ai_platform.SERVING_ARGS_KEY: vertex_endpoint_config
        }
    )

    ## Evaluate model
    model_resolver = tfx.dsl.Resolver(
        strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
        model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
        model_blessing=tfx.dsl.Channel(
            type=tfx.types.standard_artifacts.ModelBlessing)).with_id(
        'latest_blessed_model_resolver')

    # Metrics to be checked
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key='Class')],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(per_slice_thresholds={
                'binary_accuracy':
                    tfma.PerSliceMetricThresholds(thresholds=[
                        tfma.PerSliceMetricThreshold(
                            slicing_specs=[tfma.SlicingSpec()],
                            threshold=tfma.MetricThreshold(
                                value_threshold=tfma.GenericValueThreshold(
                                    lower_bound={'value': 0.6}))
                        )]),
            })])

    evaluator = tfx.components.Evaluator(
        examples=transform.outputs['transformed_examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config)

    components = [example_gen, stats_gen, schema_gen, validator, transform, trainer, pusher, model_resolver, evaluator]

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
         service_account: str,
         transform_file_gcs: str,
         trainer_file_gcs: str):
    # Beam options: project id and a temp location
    beam_args = [f"--project={project_id}", f"--temp_location={temp_location}"]

    p = create_pipeline(query=query,
                        pipeline_root=pipeline_root,
                        pipeline_name=pipeline_name,
                        beam_pipeline_args=beam_args,
                        transform_file_gcs=transform_file_gcs,
                        trainer_file_gcs=trainer_file_gcs,
                        region=region,
                        project_id=project_id,
                        temp_location=temp_location)

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
    parser.add_argument("--transform-file", required=True)
    parser.add_argument("--trainer-file", required=True)

    args = parser.parse_args()

    main(query=args.query,
         pipeline_name=args.pipeline_name,
         pipeline_root=args.pipeline_root,
         project_id=args.project_id,
         temp_location=args.temp_location,
         region=args.region,
         service_account=args.service_account,
         transform_file_gcs=args.transform_file,
         trainer_file_gcs=args.trainer_file)
