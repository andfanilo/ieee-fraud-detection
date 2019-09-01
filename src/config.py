import json

from cerberus import Validator


def read_configuration(key, configuration_file="conf.json"):
    with open(configuration_file) as jsonfile:
        config = json.load(jsonfile)

    if key not in config:
        raise ValueError(
            f"{key} not in {configuration_file} keys {list(config.keys())}"
        )
    output = config[key]
    _validate_configuration(output)
    return output


def write_params(params, path):
    with open(path, "w") as f:
        json.dump(params, f)


def _validate_configuration(params):
    schema = {
        "classifier": {"type": "string"},
        "params": {"type": "dict"},
        "save_models": {"type": "boolean"},
        "save_predictions": {"type": "boolean"},
    }
    v = Validator(schema)
    if not v.validate(params):
        raise ValueError(f"Configuration not valid : {v.errors}")
    return
