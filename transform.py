from typing import Dict

import tensorflow_transform as tft
import tensorflow as tf


def preprocessing_fn(inputs: Dict):
    normlz_amount: tf.Tensor = tft.scale_to_0_1(inputs['Amount'])

    outputs = {'Amount': normlz_amount,
               'Class': inputs['Class']}

    for col in inputs.keys():
        if col.startswith("V"):
            outputs[col] = tft.scale_to_0_1(inputs[col])

    return outputs
