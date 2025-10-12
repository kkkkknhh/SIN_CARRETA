# Minimal validators shim for tests
class RefResolver:
    def __init__(self, base_uri=None, referrer=None):
        self.base_uri = base_uri
        self.referrer = referrer


class _DummyValidator:
    def __init__(self, schema, resolver=None):
        self.schema = schema
        self.resolver = resolver

    @staticmethod
    def check_schema(_schema):
        return True

    @staticmethod
    def validate(_payload):
        return True


def validator_for(_schema):
    return _DummyValidator
