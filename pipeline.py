from typing import List

import tfx.v1 as tfx
from tfx.extensions.google_cloud_big_query.example_gen.component import BigQueryExampleGen


def create_pipeline(query: str,
                    pipeline_name: str,
                    pipeline_root: str,
                    beam_pipeline_args: List[str]) -> tfx.dsl.Pipeline:
    # Read data from BigQuery
    example_gen: BigQueryExampleGen = tfx.extensions.google_cloud_big_query.BigQueryExampleGen(query=query)

    components = [example_gen]

    pipeline = tfx.dsl.Pipeline(pipeline_name=pipeline_name,
                                pipeline_root=pipeline_root,
                                components=components,
                                beam_pipeline_args=beam_pipeline_args)

    return pipeline


def main(query: str,
         pipeline_name: str,
         pipeline_root: str,
         project_id: str,
         temp_location: str):
    # Beam options: project id and a temp location
    beam_args = [f"--project={project_id}", f"--temp_location={temp_location}"]

    p = create_pipeline(query=query,
                        pipeline_root=pipeline_root,
                        pipeline_name=pipeline_name,
                        beam_pipeline_args=beam_args)


if __name__ == '__main__':
    # Do stuff here
    main()
