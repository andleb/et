from __future__ import annotations

import copy
import math
import os
import shutil
import tempfile
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
from torch.nn import functional as F

from .eight_gauss import energy_distance_unbiased
from .mnist import MNIST_DIM, MNIST_SIDE

ONNX_MNIST12_URL = "https://github.com/onnx/models/raw/d55d2baeb0d6641643d5295a4f42b545fcf9d9e2/validated/vision/classification/mnist/model/mnist-12.onnx"
ONNX_MNIST12_FEATURE_OUTPUT_NAME = "Pooling160_Output_0_reshape0"


def ensure_onnx_mnist12_checkpoint(
    model_path: str | Path,
    *,
    url: str = ONNX_MNIST12_URL,
    timeout_seconds: float = 120.0,
) -> Path:
    """Download the public checkpoint from the configured URL when absent."""
    model_path = Path(model_path).expanduser().resolve()
    if model_path.is_file():
        return model_path
    if model_path.exists():
        raise FileExistsError(f"MNIST-12 checkpoint path is not a file: {model_path}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=model_path.parent,
            prefix=f".{model_path.name}.",
            suffix=".download",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            with urllib.request.urlopen(
                url, timeout=float(timeout_seconds)
            ) as response:
                shutil.copyfileobj(response, temporary_file)
        os.replace(temporary_path, model_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return model_path


def ensure_onnx_mnist12_feature_checkpoint(
    source_model_path: str | Path,
    feature_model_path: str | Path,
    *,
    feature_output_name: str = ONNX_MNIST12_FEATURE_OUTPUT_NAME,
) -> Path:
    """Expose one internal activation without changing the public model weights."""
    source_model_path = Path(source_model_path).expanduser().resolve()
    feature_model_path = Path(feature_model_path).expanduser().resolve()
    if not source_model_path.is_file():
        raise FileNotFoundError(
            f"ONNX Model Zoo MNIST-12 checkpoint not found: {source_model_path}"
        )
    if source_model_path == feature_model_path:
        raise ValueError("feature_model_path must not overwrite the public checkpoint")
    if feature_model_path.exists() and (not feature_model_path.is_file()):
        raise FileExistsError(
            f"derived MNIST-12 path is not a file: {feature_model_path}"
        )
    source_model = onnx.load(str(source_model_path))
    onnx.checker.check_model(source_model)
    source_output_names = [output.name for output in source_model.graph.output]
    if len(source_output_names) != 1:
        raise RuntimeError(
            "the public MNIST-12 graph must expose exactly one original output"
        )

    def validate_feature_model(path: Path) -> None:
        """Validate the ONNX graph and require the logit and requested feature outputs."""
        model = onnx.load(str(path))
        onnx.checker.check_model(model)
        output_names = [output.name for output in model.graph.output]
        expected_output_names = [source_output_names[0], feature_output_name]
        if output_names != expected_output_names:
            raise RuntimeError(
                f"derived MNIST-12 output mismatch: expected {expected_output_names}, found {output_names}"
            )

    if feature_model_path.is_file():
        validate_feature_model(feature_model_path)
        return feature_model_path
    inferred_model = onnx.shape_inference.infer_shapes(source_model)
    value_info_by_name = {
        value_info.name: value_info
        for value_info in list(inferred_model.graph.value_info)
        + list(inferred_model.graph.input)
        + list(inferred_model.graph.output)
    }
    if feature_output_name not in value_info_by_name:
        raise RuntimeError(
            f"could not infer value information for the requested MNIST-12 feature output {feature_output_name!r}"
        )
    derived_model = copy.deepcopy(inferred_model)
    derived_model.graph.output.append(
        copy.deepcopy(value_info_by_name[feature_output_name])
    )
    onnx.checker.check_model(derived_model)
    feature_model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=feature_model_path.parent,
            prefix=f".{feature_model_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        onnx.save_model(derived_model, str(temporary_path))
        validate_feature_model(temporary_path)
        os.replace(temporary_path, feature_model_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return feature_model_path


class ONNXMNIST12Classifier:

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"ONNX Model Zoo MNIST-12 checkpoint not found: {self.model_path}"
            )
        self.onnxruntime_version = onnxruntime.__version__
        self.session = onnxruntime.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError("MNIST-12 must expose exactly one input and one output")
        if tuple(inputs[0].shape) != (1, 1, MNIST_SIDE, MNIST_SIDE):
            raise RuntimeError(f"unexpected MNIST-12 input shape: {inputs[0].shape}")
        if tuple(outputs[0].shape) != (1, 10):
            raise RuntimeError(f"unexpected MNIST-12 output shape: {outputs[0].shape}")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

    @torch.inference_mode()
    def classify_vectors(
        self, vectors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return class probabilities and logits for flattened MNIST vectors."""
        vectors = torch.as_tensor(vectors).detach().cpu()
        if vectors.ndim != 2 or vectors.shape[1] != MNIST_DIM:
            raise ValueError(f"vectors must have shape (N, {MNIST_DIM})")
        if len(vectors) == 0:
            raise ValueError("vectors must contain at least one image")
        images = (
            vectors.to(torch.float32)
            .clamp(0.0, 1.0)
            .reshape(-1, 1, MNIST_SIDE, MNIST_SIDE)
            .numpy()
        )
        logits = []
        for image in images:
            output = self.session.run(
                [self.output_name], {self.input_name: image[None]}
            )[0]
            logits.append(np.asarray(output, dtype=np.float32).reshape(1, 10))
        logits_tensor = torch.from_numpy(np.concatenate(logits, axis=0))
        return (logits_tensor.softmax(dim=1), logits_tensor)

    def metadata(self) -> dict[str, object]:
        """Return the evaluator configuration and ONNX runtime details."""
        return {
            "name": "ONNX Model Zoo MNIST-12",
            "model_path": str(self.model_path),
            "input_name": self.input_name,
            "output_name": self.output_name,
            "providers": self.session.get_providers(),
            "onnxruntime_version": self.onnxruntime_version,
            "representation": "logits",
        }


class ONNXMNIST12FeatureClassifier:

    def __init__(
        self,
        source_model_path: str | Path,
        feature_model_path: str | Path,
        *,
        feature_output_name: str = ONNX_MNIST12_FEATURE_OUTPUT_NAME,
    ) -> None:
        self.source_model_path = Path(source_model_path).resolve()
        self.feature_model_path = Path(feature_model_path).resolve()
        if not self.source_model_path.is_file():
            raise FileNotFoundError(
                f"public MNIST-12 checkpoint not found: {self.source_model_path}"
            )
        if not self.feature_model_path.is_file():
            raise FileNotFoundError(
                f"derived MNIST-12 feature checkpoint not found: {self.feature_model_path}"
            )
        self.feature_output_name = str(feature_output_name)
        self.onnxruntime_version = onnxruntime.__version__
        source_session = onnxruntime.InferenceSession(
            str(self.source_model_path), providers=["CPUExecutionProvider"]
        )
        source_outputs = source_session.get_outputs()
        if len(source_outputs) != 1:
            raise RuntimeError("public MNIST-12 must expose exactly one output")
        self.logit_output_name = source_outputs[0].name
        self.session = onnxruntime.InferenceSession(
            str(self.feature_model_path), providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        outputs = {output.name: output for output in self.session.get_outputs()}
        if len(inputs) != 1:
            raise RuntimeError("derived MNIST-12 must expose exactly one input")
        if tuple(inputs[0].shape) != (1, 1, MNIST_SIDE, MNIST_SIDE):
            raise RuntimeError(
                f"unexpected derived MNIST-12 input shape: {inputs[0].shape}"
            )
        expected_output_names = {self.logit_output_name, self.feature_output_name}
        if set(outputs) != expected_output_names:
            raise RuntimeError(
                f"unexpected derived MNIST-12 outputs: expected {sorted(expected_output_names)}, found {sorted(outputs)}"
            )
        if tuple(outputs[self.logit_output_name].shape) != (1, 10):
            raise RuntimeError(
                f"unexpected derived MNIST-12 logit shape: {outputs[self.logit_output_name].shape}"
            )
        feature_shape = tuple(outputs[self.feature_output_name].shape)
        if (
            len(feature_shape) != 2
            or feature_shape[0] != 1
            or (not isinstance(feature_shape[1], int))
            or (feature_shape[1] <= 0)
        ):
            raise RuntimeError(f"unexpected MNIST-12 feature shape: {feature_shape}")
        self.feature_shape = feature_shape
        self.feature_dimension = int(feature_shape[1])
        self.input_name = inputs[0].name

    @torch.inference_mode()
    def classify_vectors(
        self, vectors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return probabilities, logits, and raw penultimate activations."""
        vectors = torch.as_tensor(vectors).detach().cpu()
        if vectors.ndim != 2 or vectors.shape[1] != MNIST_DIM:
            raise ValueError(f"vectors must have shape (N, {MNIST_DIM})")
        if len(vectors) == 0:
            raise ValueError("vectors must contain at least one image")
        images = (
            vectors.to(torch.float32)
            .clamp(0.0, 1.0)
            .reshape(-1, 1, MNIST_SIDE, MNIST_SIDE)
            .numpy()
        )
        logits = []
        features = []
        for image in images:
            output_logits, output_features = self.session.run(
                [self.logit_output_name, self.feature_output_name],
                {self.input_name: image[None]},
            )
            logits.append(np.asarray(output_logits, dtype=np.float32).reshape(1, 10))
            features.append(
                np.asarray(output_features, dtype=np.float32).reshape(
                    1, self.feature_dimension
                )
            )
        logits_tensor = torch.from_numpy(np.concatenate(logits, axis=0))
        feature_tensor = torch.from_numpy(np.concatenate(features, axis=0))
        if not torch.isfinite(logits_tensor).all():
            raise RuntimeError("derived MNIST-12 produced non-finite logits")
        if not torch.isfinite(feature_tensor).all():
            raise RuntimeError("derived MNIST-12 produced non-finite features")
        return (logits_tensor.softmax(dim=1), logits_tensor, feature_tensor)

    def metadata(self) -> dict[str, object]:
        """Return the evaluator paths and feature configuration."""
        return {
            "name": "ONNX Model Zoo MNIST-12 with exposed penultimate output",
            "source_model_path": str(self.source_model_path),
            "feature_model_path": str(self.feature_model_path),
            "input_name": self.input_name,
            "logit_output_name": self.logit_output_name,
            "feature_output_name": self.feature_output_name,
            "feature_shape": list(self.feature_shape),
            "feature_dimension": self.feature_dimension,
            "providers": self.session.get_providers(),
            "onnxruntime_version": self.onnxruntime_version,
            "representation": "raw penultimate activation",
        }


def mnist_prediction_metrics(
    probabilities: torch.Tensor,
    *,
    target_class_probabilities: torch.Tensor | None = None,
) -> tuple[dict[str, float | int], torch.Tensor]:
    """Summarize recognizability and predicted digit-class coverage."""
    probabilities = torch.as_tensor(probabilities, dtype=torch.float64).cpu()
    if probabilities.ndim != 2 or probabilities.shape[1] != 10:
        raise ValueError("probabilities must have shape (N, 10)")
    if len(probabilities) == 0:
        raise ValueError("probabilities must contain at least one image")
    confidence, predictions = probabilities.max(dim=1)
    counts = torch.bincount(predictions, minlength=10)
    class_probabilities = counts.to(torch.float64) / counts.sum()
    metrics: dict[str, float | int] = {
        "mean_classifier_confidence": float(confidence.mean()),
        "classes_present": int((counts > 0).sum()),
    }
    if target_class_probabilities is not None:
        target = torch.as_tensor(target_class_probabilities, dtype=torch.float64).cpu()
        if target.shape != (10,):
            raise ValueError("target_class_probabilities must have shape (10,)")
        target = target / target.sum()
        metrics["class_distribution_tv"] = float(
            0.5 * (class_probabilities - target).abs().sum()
        )
    return (metrics, counts)


def mnist_representation_energy_distance(
    representations: torch.Tensor,
    reference_representations: torch.Tensor,
    *,
    device: str | torch.device,
    normalize: bool = True,
) -> float:
    """Compute signed energy distance between fixed representations."""
    representations = torch.as_tensor(representations, dtype=torch.float32)
    reference_representations = torch.as_tensor(
        reference_representations, dtype=torch.float32
    )
    if representations.ndim != 2 or reference_representations.ndim != 2:
        raise ValueError("representations must both be two-dimensional")
    if representations.shape[1] != reference_representations.shape[1]:
        raise ValueError("representation dimensions must agree")
    if normalize:
        representations = F.normalize(representations, dim=1)
        reference_representations = F.normalize(reference_representations, dim=1)
    representations = representations.to(device)
    reference_representations = reference_representations.to(device)
    return float(
        energy_distance_unbiased(representations, reference_representations)
        .detach()
        .cpu()
    )


def mnist_stratified_bootstrap_primary_metrics(
    probability_batches: Sequence[torch.Tensor],
    feature_batches: Sequence[torch.Tensor],
    reference_features: torch.Tensor,
    target_class_probabilities: torch.Tensor,
    *,
    n_bootstrap: int,
    seed: int,
    device: str | torch.device,
) -> dict[str, object]:
    """Pool equal-size seed batches and bootstrap images within each seed."""
    if len(probability_batches) == 0:
        raise ValueError("at least one probability batch is required")
    if len(probability_batches) != len(feature_batches):
        raise ValueError("probability_batches and feature_batches must align")
    if int(n_bootstrap) < 2:
        raise ValueError("n_bootstrap must be at least two")
    probabilities = [
        torch.as_tensor(batch, dtype=torch.float32).detach().cpu()
        for batch in probability_batches
    ]
    features = [
        torch.as_tensor(batch, dtype=torch.float32).detach().cpu()
        for batch in feature_batches
    ]
    batch_sizes = [len(batch) for batch in probabilities]
    if min(batch_sizes) < 2:
        raise ValueError("each seed batch must contain at least two images")
    if len(set(batch_sizes)) != 1:
        raise ValueError(
            "seed-stratified bootstrap requires equal sample counts per seed"
        )
    if features[0].ndim != 2:
        raise ValueError("each feature batch must be two-dimensional")
    feature_dimension = features[0].shape[1]
    for probability_batch, feature_batch in zip(probabilities, features, strict=True):
        if probability_batch.ndim != 2 or probability_batch.shape[1] != 10:
            raise ValueError("each probability batch must have shape (N, 10)")
        if (
            feature_batch.ndim != 2
            or feature_batch.shape[1] != feature_dimension
            or len(feature_batch) != len(probability_batch)
        ):
            raise ValueError("each feature batch must align with its probability batch")
    pooled_probabilities = torch.cat(probabilities)
    pooled_features = torch.cat(features)
    reference_features = (
        torch.as_tensor(reference_features, dtype=torch.float32).detach().cpu()
    )
    if (
        reference_features.ndim != 2
        or reference_features.shape[1] != feature_dimension
        or len(reference_features) < 2
    ):
        raise ValueError("reference_features must have shape (M, D) with M >= 2")
    if not torch.isfinite(pooled_probabilities).all():
        raise ValueError("probability batches contain non-finite values")
    if not torch.isfinite(pooled_features).all():
        raise ValueError("feature batches contain non-finite values")
    if not torch.isfinite(reference_features).all():
        raise ValueError("reference_features contains non-finite values")
    target = torch.as_tensor(target_class_probabilities, dtype=torch.float32).cpu()
    if target.shape != (10,) or not torch.isfinite(target).all():
        raise ValueError("target_class_probabilities must be finite with shape (10,)")
    if float(target.sum()) <= 0.0:
        raise ValueError("target_class_probabilities must have positive mass")
    target = target / target.sum()
    metric_device = torch.device(device)
    pooled_features_device = pooled_features.to(metric_device)
    reference_features_device = reference_features.to(metric_device)
    generated_pairwise = torch.cdist(pooled_features_device, pooled_features_device)
    generated_to_reference_mean = torch.cdist(
        pooled_features_device, reference_features_device
    ).mean(dim=1)
    reference_self_mean = torch.pdist(reference_features_device).mean()
    n_samples = len(pooled_features)
    generated_self_mean = generated_pairwise.sum() / (n_samples * (n_samples - 1))
    point_feature_energy_distance = (
        2.0 * generated_to_reference_mean.mean()
        - generated_self_mean
        - reference_self_mean
    )
    pooled_confidence, pooled_predictions = pooled_probabilities.max(dim=1)
    pooled_class_probabilities = torch.bincount(pooled_predictions, minlength=10).to(
        torch.float32
    )
    pooled_class_probabilities /= pooled_class_probabilities.sum()
    point_mean_confidence = pooled_confidence.mean()
    point_class_tv = 0.5 * (pooled_class_probabilities - target).abs().sum()
    rng = np.random.default_rng(int(seed))
    bootstrap_weight_blocks = [
        rng.multinomial(
            batch_size, np.full(batch_size, 1.0 / batch_size), size=int(n_bootstrap)
        )
        for batch_size in batch_sizes
    ]
    bootstrap_weights = torch.from_numpy(
        np.concatenate(bootstrap_weight_blocks, axis=1).astype(np.float32, copy=False)
    ).to(metric_device)
    bootstrap_cross_mean = (
        bootstrap_weights * generated_to_reference_mean[None, :]
    ).sum(dim=1) / n_samples
    bootstrap_generated_self_mean = (
        bootstrap_weights @ generated_pairwise * bootstrap_weights
    ).sum(dim=1) / (n_samples * (n_samples - 1))
    bootstrap_feature_energy_distance = (
        2.0 * bootstrap_cross_mean - bootstrap_generated_self_mean - reference_self_mean
    )
    confidence_device = pooled_confidence.to(metric_device)
    bootstrap_mean_confidence = (bootstrap_weights * confidence_device[None, :]).sum(
        dim=1
    ) / n_samples
    prediction_indicators = F.one_hot(pooled_predictions, num_classes=10).to(
        device=metric_device, dtype=torch.float32
    )
    bootstrap_class_probabilities = (
        bootstrap_weights @ prediction_indicators / n_samples
    )
    bootstrap_class_tv = 0.5 * (
        bootstrap_class_probabilities - target.to(metric_device)[None, :]
    ).abs().sum(dim=1)
    return {
        "n_seeds": len(batch_sizes),
        "samples_per_seed": batch_sizes[0],
        "n_samples": n_samples,
        "point": {
            "penultimate_feature_energy_distance": float(
                point_feature_energy_distance.detach().cpu()
            ),
            "mean_classifier_confidence": float(point_mean_confidence),
            "class_distribution_tv": float(point_class_tv),
        },
        "bootstrap": {
            "penultimate_feature_energy_distance": bootstrap_feature_energy_distance.detach()
            .cpu()
            .numpy(),
            "mean_classifier_confidence": bootstrap_mean_confidence.detach()
            .cpu()
            .numpy(),
            "class_distribution_tv": bootstrap_class_tv.detach().cpu().numpy(),
        },
    }


@torch.inference_mode()
def nearest_reference_pixel_rmse(
    vectors: torch.Tensor,
    reference_vectors: torch.Tensor,
    *,
    device: str | torch.device,
    reference_chunk_size: int = 2048,
) -> torch.Tensor:
    """Return each image's nearest-reference pixel RMSE."""
    vectors = torch.as_tensor(vectors, dtype=torch.float32).cpu()
    reference_vectors = torch.as_tensor(reference_vectors, dtype=torch.float32).cpu()
    if vectors.ndim != 2 or vectors.shape[1] != MNIST_DIM:
        raise ValueError(f"vectors must have shape (N, {MNIST_DIM})")
    if reference_vectors.ndim != 2 or reference_vectors.shape[1] != MNIST_DIM:
        raise ValueError(f"reference_vectors must have shape (N, {MNIST_DIM})")
    queries = vectors.clamp(0.0, 1.0).to(device)
    nearest_l2 = torch.full(
        (len(queries),), float("inf"), device=device, dtype=queries.dtype
    )
    for start in range(0, len(reference_vectors), int(reference_chunk_size)):
        stop = min(start + int(reference_chunk_size), len(reference_vectors))
        reference = reference_vectors[start:stop].to(device=device, non_blocking=True)
        nearest_l2 = torch.minimum(
            nearest_l2, torch.cdist(queries, reference).min(dim=1).values
        )
    return (nearest_l2 / math.sqrt(MNIST_DIM)).cpu()
