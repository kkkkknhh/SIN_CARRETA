from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PipelineStage(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PIPELINE_STAGE_UNSPECIFIED: _ClassVar[PipelineStage]
    PIPELINE_STAGE_SANITIZATION: _ClassVar[PipelineStage]
    PIPELINE_STAGE_PLAN_PROCESSING: _ClassVar[PipelineStage]
    PIPELINE_STAGE_SEGMENTATION: _ClassVar[PipelineStage]
    PIPELINE_STAGE_EMBEDDING: _ClassVar[PipelineStage]
    PIPELINE_STAGE_RESPONSIBILITY: _ClassVar[PipelineStage]
    PIPELINE_STAGE_CONTRADICTION: _ClassVar[PipelineStage]
    PIPELINE_STAGE_MONETARY: _ClassVar[PipelineStage]
    PIPELINE_STAGE_FEASIBILITY: _ClassVar[PipelineStage]
    PIPELINE_STAGE_CAUSAL: _ClassVar[PipelineStage]
    PIPELINE_STAGE_TEORIA: _ClassVar[PipelineStage]
    PIPELINE_STAGE_DAG: _ClassVar[PipelineStage]
    PIPELINE_STAGE_REGISTRY_BUILD: _ClassVar[PipelineStage]
    PIPELINE_STAGE_DECALOGO_LOAD: _ClassVar[PipelineStage]
    PIPELINE_STAGE_DECALOGO_EVAL: _ClassVar[PipelineStage]
    PIPELINE_STAGE_QUESTIONNAIRE_EVAL: _ClassVar[PipelineStage]
    PIPELINE_STAGE_ANSWERS_ASSEMBLY: _ClassVar[PipelineStage]
PIPELINE_STAGE_UNSPECIFIED: PipelineStage
PIPELINE_STAGE_SANITIZATION: PipelineStage
PIPELINE_STAGE_PLAN_PROCESSING: PipelineStage
PIPELINE_STAGE_SEGMENTATION: PipelineStage
PIPELINE_STAGE_EMBEDDING: PipelineStage
PIPELINE_STAGE_RESPONSIBILITY: PipelineStage
PIPELINE_STAGE_CONTRADICTION: PipelineStage
PIPELINE_STAGE_MONETARY: PipelineStage
PIPELINE_STAGE_FEASIBILITY: PipelineStage
PIPELINE_STAGE_CAUSAL: PipelineStage
PIPELINE_STAGE_TEORIA: PipelineStage
PIPELINE_STAGE_DAG: PipelineStage
PIPELINE_STAGE_REGISTRY_BUILD: PipelineStage
PIPELINE_STAGE_DECALOGO_LOAD: PipelineStage
PIPELINE_STAGE_DECALOGO_EVAL: PipelineStage
PIPELINE_STAGE_QUESTIONNAIRE_EVAL: PipelineStage
PIPELINE_STAGE_ANSWERS_ASSEMBLY: PipelineStage

class EvidencePacket(_message.Message):
    __slots__ = ("schema_version", "stage", "source_component", "evidence_type", "content_json", "confidence", "applicable_questions", "metadata_json", "timestamp", "signature", "evidence_hash")
    SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    STAGE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_COMPONENT_FIELD_NUMBER: _ClassVar[int]
    EVIDENCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_JSON_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    APPLICABLE_QUESTIONS_FIELD_NUMBER: _ClassVar[int]
    METADATA_JSON_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    SIGNATURE_FIELD_NUMBER: _ClassVar[int]
    EVIDENCE_HASH_FIELD_NUMBER: _ClassVar[int]
    schema_version: str
    stage: PipelineStage
    source_component: str
    evidence_type: str
    content_json: str
    confidence: float
    applicable_questions: _containers.RepeatedScalarFieldContainer[str]
    metadata_json: str
    timestamp: str
    signature: str
    evidence_hash: str
    def __init__(self, schema_version: _Optional[str] = ..., stage: _Optional[_Union[PipelineStage, str]] = ..., source_component: _Optional[str] = ..., evidence_type: _Optional[str] = ..., content_json: _Optional[str] = ..., confidence: _Optional[float] = ..., applicable_questions: _Optional[_Iterable[str]] = ..., metadata_json: _Optional[str] = ..., timestamp: _Optional[str] = ..., signature: _Optional[str] = ..., evidence_hash: _Optional[str] = ...) -> None: ...

class EvidencePacketBatch(_message.Message):
    __slots__ = ("packets", "batch_id", "batch_timestamp")
    PACKETS_FIELD_NUMBER: _ClassVar[int]
    BATCH_ID_FIELD_NUMBER: _ClassVar[int]
    BATCH_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    packets: _containers.RepeatedCompositeFieldContainer[EvidencePacket]
    batch_id: str
    batch_timestamp: str
    def __init__(self, packets: _Optional[_Iterable[_Union[EvidencePacket, _Mapping]]] = ..., batch_id: _Optional[str] = ..., batch_timestamp: _Optional[str] = ...) -> None: ...

class RegistryEntry(_message.Message):
    __slots__ = ("prev_hash", "packet", "entry_hash", "sequence_number")
    PREV_HASH_FIELD_NUMBER: _ClassVar[int]
    PACKET_FIELD_NUMBER: _ClassVar[int]
    ENTRY_HASH_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    prev_hash: str
    packet: EvidencePacket
    entry_hash: str
    sequence_number: int
    def __init__(self, prev_hash: _Optional[str] = ..., packet: _Optional[_Union[EvidencePacket, _Mapping]] = ..., entry_hash: _Optional[str] = ..., sequence_number: _Optional[int] = ...) -> None: ...
