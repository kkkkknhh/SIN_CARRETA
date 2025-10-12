"""
Example usage of the embedding model.
"""

import argparse
import importlib
import os
import sys
import traceback

import numpy as np

from version_validator import validate_numpy_compatibility, validate_python_310


def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)


def _report_error(context, exc):
    """Print concise error info; set DEBUG=1 to include traceback."""
    print(f"[ERROR] {context}: {type(exc).__name__}: {exc}")
    if os.getenv("DEBUG") == "1":
        traceback.print_exc()


def parse_args():
    """CLI options to control model factory, device, and batching."""
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument(
        "--factory", choices=["auto", "industrial", "generic"], default="auto"
    )
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Embed texts in batches; 0=single call",
    )
    p.add_argument(
        "--ignite",
        action="store_true",
        help="Start full system: tracing + SLO + canary deployment",
    )
    p.add_argument(
        "--deployment-id",
        default="deployment-v2.0",
        help="Deployment identifier for canary controller",
    )
    p.add_argument(
        "--requests",
        type=int,
        default=20,
        help="Number of requests for the ignition run",
    )
    return p.parse_args()


def embed_in_batches(model, texts, batch_size: int):
    """Call model.embed in batches and return a 2D numpy array."""
    if batch_size <= 0 or len(texts) <= batch_size:
        embs = model.embed(texts)
        embs = np.asarray(embs, dtype=np.float32)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        return embs
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i: i + batch_size]
        embs = model.embed(chunk)
        embs = np.asarray(embs, dtype=np.float32)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        out.append(embs)
    return np.vstack(out)


def ignite_system(args):
    """Initialize tracing, SLO monitoring, and execute canary deployment."""
    try:
        print("Igniting deployment infrastructure...")
        from canary_deployment import create_canary_controller
        from opentelemetry_instrumentation import initialize_tracing
        from slo_monitoring import create_slo_monitor

        initialize_tracing(service_name="decalogo-evaluation-system")
        _slo_monitor = create_slo_monitor()
        controller = create_canary_controller(args.deployment_id)

        def request_generator():
            for i in range(args.requests):
                yield {"id": i, "payload": f"request-{i}"}

        result = controller.execute_deployment(request_generator())
        status = getattr(result, "status", None)
        routed = getattr(result, "routed_requests", None)
        print(
            f"Deployment complete: status={status or 'unknown'}; routed={routed if routed is not None else 'n/a'}"
        )
    except Exception as e:
        _report_error("System ignition failed", e)


def main():
    """Demonstrate the embedding model with examples."""
    args = parse_args()
    print("Validating Python 3.10 and NumPy compatibility...")
    try:
        validate_python_310()
        validate_numpy_compatibility()
        print(
            f"Environment OK -> Python: {sys.version.split()[0]} | NumPy: {np.__version__}"
        )
    except Exception as e:
        _report_error("Environment validation failed", e)
        return

    # Start full system if requested, then exit
    if args.ignite:
        ignite_system(args)
        return

    # Respect requested device before importing the model
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print(
        f"Config -> factory={args.factory}, device={args.device}, batch_size={args.batch_size}"
    )

    print("Initializing embedding model...")
    try:
        # Choose factory per CLI
        if args.factory == "industrial":
            from embedding_model import (
                create_industrial_embedding_model as _create_model,
            )
        elif args.factory == "generic":
            from embedding_model import create_embedding_model as _create_model
        else:
            try:
                from embedding_model import (
                    create_industrial_embedding_model as _create_model,
                )
            except ImportError:
                from embedding_model import create_embedding_model as _create_model
        model = _create_model()
        print(f"Model loaded successfully: {model.get_model_info()}")
    except Exception as e:
        _report_error("Model initialization failed (first attempt)", e)
        if args.device != "cpu":
            print("Retrying on CPU only...")
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                sys.modules.pop("embedding_model", None)
                em = importlib.import_module("embedding_model")
                _create_model = (
                    getattr(em, "create_industrial_embedding_model", None)
                    if args.factory in ("industrial", "auto")
                    else None
                ) or getattr(em, "create_embedding_model")
                model = _create_model()
                print(f"Model loaded successfully (CPU): {model.get_model_info()}")
            except Exception as e2:
                _report_error("Model initialization failed (CPU retry)", e2)
                return
        else:
            return

    try:
        # Example texts in Spanish and English
        texts = [
            "La inteligencia artificial está cambiando el mundo",
            "Artificial intelligence is changing the world",
            "Los modelos de lenguaje son cada vez más avanzados",
            "Me gusta la programación y la ciencia de datos",
            "I enjoy programming and data science",
        ]
        print("\nCreating embeddings for example texts...")
        embeddings = embed_in_batches(model, texts, args.batch_size)
        print(
            f"Created {len(embeddings)} embeddings of dimension {embeddings.shape[1]}"
        )

        similarity = cosine_similarity(embeddings[0], embeddings[1])
        print(f"\nSimilarity between '{texts[0]}' and '{texts[1]}': {similarity:.4f}")

        similarity = cosine_similarity(embeddings[0], embeddings[3])
        print(f"Similarity between '{texts[0]}' and '{texts[3]}': {similarity:.4f}")

        similarity = cosine_similarity(embeddings[3], embeddings[4])
        print(f"Similarity between '{texts[3]}' and '{texts[4]}': {similarity:.4f}")
    except Exception as e:
        _report_error("Embedding or similarity demo failed", e)


if __name__ == "__main__":
    main()
