# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independently verify one exact-version untimed ABI 10 H100 artifact."""

import argparse
import json
from pathlib import Path

from target1_abi10_h100_evidence import (
    load_artifact_bundle,
    load_authorization,
    load_iris_receipt,
    load_wandb_receipt,
    validate_authorized_checkout,
    validate_gpu_artifacts,
    validate_gpu_build_artifacts,
    validate_numerical_artifacts,
    validate_observer_and_cache_artifacts,
    validate_registration_artifacts,
    validate_subject_receipt,
    validate_te_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--challenge", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--subject-receipt", type=Path, required=True)
    parser.add_argument("--wandb-receipt", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()

    authorization = load_authorization()
    document = authorization.decoded()
    if arguments.authorization_sha256 != authorization.sha256 or arguments.challenge != document["challenge"]:
        raise RuntimeError("verifier argv differs from the sealed authorization")
    repository = Path(__file__).resolve().parents[4]
    validate_authorized_checkout(authorization, repository)
    artifact = load_artifact_bundle(arguments.artifact, authorization)
    subject = load_iris_receipt(arguments.subject_receipt)
    wandb = load_wandb_receipt(arguments.wandb_receipt)
    validate_subject_receipt(subject, wandb, artifact, authorization)
    numerical = validate_numerical_artifacts(artifact, authorization)
    build = validate_gpu_build_artifacts(artifact, authorization)
    transport = validate_gpu_artifacts(artifact, authorization)
    cache = validate_observer_and_cache_artifacts(artifact, authorization)
    registration = validate_registration_artifacts(artifact)
    te_results = validate_te_artifacts(artifact, authorization)
    report = {
        "schema_version": 1,
        "authorization_sha256": authorization.sha256,
        "artifact_manifest_sha256": artifact.manifest_sha256,
        "wandb": {
            "entity": wandb.entity,
            "project": wandb.project,
            "name": wandb.name,
            "version": wandb.version,
            "digest": wandb.digest,
        },
        "numerical": numerical,
        "transport_sha256": transport.transport_root,
        "jax_extension_sha256": build["jax_extension_sha256"],
        "cache": cache,
        "registration_plugin_sha256": registration["plugin_extension_sha256"],
        "te_result_count": len(te_results),
        "performance_status": "not_started",
        "verdict": "accepted_untimed_operational_evidence",
    }
    arguments.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "schema_version": 1,
                "role": "verify",
                "authorization_sha256": authorization.sha256,
                "wandb_entity": wandb.entity,
                "wandb_project": wandb.project,
                "wandb_name": wandb.name,
                "wandb_version": wandb.version,
                "wandb_digest": wandb.digest,
                "artifact_manifest_sha256": artifact.manifest_sha256,
                "verdict": "accepted",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
