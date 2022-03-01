from typing import Union, Any, List

from tensorflow_metadata.proto.v0.schema_pb2 import Schema
from tensorflow_transform import TFTransformOutput
from tfx.components.trainer.fn_args_utils import FnArgs

import tensorflow as tf
import tfx.v1 as tfx
import tensorflow_transform as tft
from tfx_bsl.public import tfxio


def parse_data(location: List[str],
               data_accessor: tfx.components.DataAccessor,
               schema: Schema,
               batch_size: int) -> tf.data.Dataset:
    return data_accessor.tf_dataset_factory(
        location,
        tfxio.TensorFlowDatasetOptions(batch_size=batch_size, label_key="Class"),
        schema=schema).repeat()


def run_fn(fn_args: FnArgs):
    batch_size = fn_args.custom_config['batch-size']
    dataset_size = fn_args.custom_config['dataset-size']

    tft_output_path = fn_args.transform_graph_path
    tft_output: TFTransformOutput = tft.TFTransformOutput(tft_output_path)
    schema: Schema = tft_output.transformed_metadata.schema

    train_files: List[str] = fn_args.train_files
    eval_file = fn_args.eval_files
