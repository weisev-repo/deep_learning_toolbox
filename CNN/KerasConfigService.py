import json
import tensorflow as tf
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class StringBuilder:
    def __init__(self):
        self._string = ''

    def build(self):
        return self._string

    def append(self, string, add_space=True, delimiter=None):
        if delimiter and self._string:
            self._string += str(delimiter)
        elif add_space and self._string:
            self._string.rstrip()
            self._string += " "
        self._string += str(string)

    def append_key_value(self, key, value, delimiter=None):
        self.append(key, delimiter=delimiter)
        self.append(":", add_space=False)
        self.append(value)

    def new_line(self):
        self._string += " <br> "

    def append_new_line(self, string):
        self.new_line()
        self.append(string, add_space=False)


class KerasConfigService:

    @staticmethod
    def to_json(model: tf.keras.models.Sequential):
        _config: dict = model.get_config()
        print(type(_config))
        return json.dumps(_config, cls=NumpyEncoder)

    @staticmethod
    def get_name(config):
        return config['name']

    @staticmethod
    def handle_string(parent, key):
        builder: StringBuilder = StringBuilder()
        if key in parent:
            builder.append_key_value(key, str(parent[key]), delimiter=", ")
            return builder.build().rstrip()
        else:
            return ''

    @staticmethod
    def handle_dict(parent, dictionary: dict):
        builder: StringBuilder = StringBuilder()
        if type(dictionary) is dict and len(list(dictionary.keys())) > 0:
            key = list(dictionary.keys())[0]
            if key in parent and type(dictionary[key]) is list:
                values = parent[key]
                result = KerasConfigService.handle_array(values, dictionary[key])
                if result and len(result) > 0:
                    builder.append(key)
                    builder.append(":{", add_space=False)
                    builder.append(result)
                    builder.append("}")
        return builder.build()

    @staticmethod
    def handle_array(parent, array):
        builder: StringBuilder = StringBuilder()
        for element in array:
            if type(element) is str:
                builder.append(KerasConfigService.handle_string(parent, element))
            if type(element) is dict:
                builder.append(KerasConfigService.handle_dict(parent, element))
        return builder.build().rstrip()

    @staticmethod
    def layer_to_string(layer, include_configs=[], enumerator=None):
        builder: StringBuilder = StringBuilder()
        builder.append("Layer")

        if enumerator:
            builder.append(f'{enumerator}:')
        else:
            builder.append(":")

        for element in include_configs:
            if type(element) is str:
                builder.append(KerasConfigService.handle_string(layer, element))
            elif type(element) is list:
                builder.append(KerasConfigService.handle_array(layer, element))
            elif type(element) is dict:
                builder.append(KerasConfigService.handle_dict(layer, element))
        return builder.build().rstrip()

    @staticmethod
    def to_string(model: tf.keras.models.Sequential, include_configs=[], line_breaks=False):
        builder: StringBuilder = StringBuilder()
        _config: dict = model.get_config()
        _layers = _config['layers']

        for i, layer in enumerate(_layers):
            builder.append(KerasConfigService.layer_to_string(layer, include_configs, i + 1))
            builder.append(";", add_space=False)
            if line_breaks:
                builder.append("\n")

        return builder.build().rstrip()
