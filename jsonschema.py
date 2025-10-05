# Minimal shim for jsonschema used in tests
class RefResolver:
    def __init__(self, base_uri=None, referrer=None):
        self.base_uri = base_uri
        self.referrer = referrer


class _DummyValidator:
    def __init__(self, schema, resolver=None):
        self.schema = schema
        self.resolver = resolver

    @staticmethod
    def check_schema(schema):
        # basic noop check
        return True

    @staticmethod
    def validate(payload):
        # naive validation: assume payload is valid
        return True


def validator_for(schema):
    # return a dummy validator class
    return _DummyValidator

