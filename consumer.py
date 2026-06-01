"""
Kafka Consumer — SmartDischarge Real-Time Ingestion
====================================================
Consumes encounter events from an upstream topic and feeds them directly
into the SmartDischarge prediction pipeline, replacing the batch CSV loader
for real-time deployments (e.g. EHR discharge events via HL7/FHIR → Kafka).

Topic consumed:   smartdischarge.encounters.raw
Topic produced:   smartdischarge.predictions  (via KafkaEventProducer)

Run standalone:
    python -m src.kafka.consumer

Or import and run programmatically:
    from src.kafka.consumer import run_consumer
    run_consumer(predict_fn=my_predict_fn, max_messages=1000)

Architecture:
  EHR / upstream system
       │  (HL7/FHIR discharge event)
       ▼
  smartdischarge.encounters.raw  ← this consumer polls here
       │
       ▼
  schema validation + feature construction
       │
       ▼
  ML model (same XGBoost stack as batch pipeline)
       │
       ▼
  smartdischarge.predictions  (KafkaEventProducer)
  smartdischarge.alerts       (HIGH-risk only)
"""

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("smartdischarge.kafka.consumer")

# ── Config ────────────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP      = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_ENABLED        = os.getenv("KAFKA_ENABLED", "true").lower() == "true"
TOPIC_ENCOUNTERS_RAW = "smartdischarge.encounters.raw"
CONSUMER_GROUP       = "smartdischarge-pipeline"

# Required fields for a valid encounter event
REQUIRED_FIELDS = {
    "time_in_hospital", "number_inpatient", "number_emergency",
    "number_outpatient", "number_diagnoses", "num_medications",
    "num_lab_procedures", "age_midpoint", "diabetesMed", "change", "A1Cresult",
}


def _validate_encounter(event: dict) -> tuple[bool, list[str]]:
    """Light schema check before handing to the model."""
    missing = [f for f in REQUIRED_FIELDS if f not in event]
    errors = []
    if missing:
        errors.append(f"Missing fields: {missing}")
    if "time_in_hospital" in event:
        val = event["time_in_hospital"]
        if not (1 <= int(val) <= 14):
            errors.append(f"time_in_hospital={val} out of range [1,14]")
    return len(errors) == 0, errors


def run_consumer(
    predict_fn: Optional[Callable] = None,
    max_messages: int = 0,
    poll_timeout: float = 1.0,
) -> None:
    """
    Poll TOPIC_ENCOUNTERS_RAW, validate each message, run prediction,
    and emit results via KafkaEventProducer.

    Args:
        predict_fn:    Callable that accepts a dict and returns a prediction
                       dict (same schema as /predict response). If None,
                       loads the saved model artefacts directly.
        max_messages:  Stop after N messages (0 = run forever, until SIGINT).
        poll_timeout:  Seconds to wait for a message before looping.
    """
    if not KAFKA_ENABLED:
        log.warning("Kafka disabled — consumer exiting immediately")
        return

    try:
        from confluent_kafka import Consumer, KafkaError, KafkaException
    except ImportError:
        log.error("confluent-kafka not installed. Run: pip install confluent-kafka>=2.3.0")
        return

    # ── Load model if no predict_fn supplied ─────────────────────────────────
    if predict_fn is None:
        log.info("Loading SmartDischarge model artefacts for consumer...")
        sys.path.insert(0, str(Path(__file__).parents[3]))
        try:
            import pickle
            MODEL_DIR = Path(__file__).parents[2] / "data" / "processed"
            with open(MODEL_DIR / "model.pkl", "rb") as f:
                model = pickle.load(f)
            from src.api.server import encounter_to_features, FEATURE_ORDER
            from src.api.server import PatientEncounter

            def predict_fn(encounter_dict: dict) -> dict:
                enc = PatientEncounter(**encounter_dict)
                X   = encounter_to_features(enc)
                prob = float(model.predict_proba(X)[0, 1])
                from src.model.train import THRESHOLD_HIGH, THRESHOLD_MEDIUM
                tier = ("HIGH"   if prob >= THRESHOLD_HIGH   else
                        "MEDIUM" if prob >= THRESHOLD_MEDIUM else "LOW")
                return {
                    "encounter_id":        encounter_dict.get("encounter_id"),
                    "readmit_probability": round(prob, 4),
                    "risk_tier":           tier,
                    "needs_review":        tier == "HIGH",
                    "model_version":       "smartdischarge-v1.0",
                    "timestamp":           datetime.now(timezone.utc).isoformat(),
                }
        except Exception as exc:
            log.error("Failed to load model artefacts: %s — consumer aborting", exc)
            return

    # ── Kafka consumer config ─────────────────────────────────────────────────
    conf = {
        "bootstrap.servers":        KAFKA_BOOTSTRAP,
        "group.id":                 CONSUMER_GROUP,
        "auto.offset.reset":        "earliest",
        "enable.auto.commit":       False,        # manual commit after processing
        "max.poll.interval.ms":     300_000,
        "session.timeout.ms":       30_000,
        "heartbeat.interval.ms":    3_000,
    }

    consumer = Consumer(conf)
    consumer.subscribe([TOPIC_ENCOUNTERS_RAW])
    log.info("Kafka consumer started | broker=%s topic=%s group=%s",
             KAFKA_BOOTSTRAP, TOPIC_ENCOUNTERS_RAW, CONSUMER_GROUP)

    from src.kafka.producer import kafka_producer

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    running = True
    def _shutdown(sig, frame):
        nonlocal running
        log.info("Shutdown signal received — draining consumer...")
        running = False
    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Stats ─────────────────────────────────────────────────────────────────
    stats = {"processed": 0, "errors": 0, "high_risk": 0, "skipped": 0}
    t_start = time.time()

    try:
        while running:
            if max_messages and stats["processed"] >= max_messages:
                log.info("Reached max_messages=%d — stopping", max_messages)
                break

            msg = consumer.poll(timeout=poll_timeout)

            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    log.debug("Partition EOF | topic=%s partition=%s",
                              msg.topic(), msg.partition())
                else:
                    log.error("Consumer error: %s", msg.error())
                    stats["errors"] += 1
                continue

            # ── Deserialise ───────────────────────────────────────────────────
            try:
                raw = json.loads(msg.value().decode("utf-8"))
                # Unwrap envelope if present (matches producer schema)
                encounter = raw.get("payload", raw)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                log.warning("Bad message at offset %s: %s", msg.offset(), exc)
                stats["errors"] += 1
                consumer.commit(message=msg)
                continue

            # ── Validate ──────────────────────────────────────────────────────
            valid, validation_errors = _validate_encounter(encounter)
            if not valid:
                log.warning("Schema validation failed | offset=%s errors=%s",
                            msg.offset(), validation_errors)
                stats["skipped"] += 1
                consumer.commit(message=msg)
                continue

            # ── Predict ───────────────────────────────────────────────────────
            try:
                prediction = predict_fn(encounter)
                kafka_producer.emit_prediction(prediction)

                if prediction.get("risk_tier") == "HIGH":
                    stats["high_risk"] += 1

                stats["processed"] += 1
                log.info(
                    "Scored | encounter=%s prob=%.3f tier=%s | total=%d",
                    prediction.get("encounter_id", "?"),
                    prediction.get("readmit_probability", 0),
                    prediction.get("risk_tier"),
                    stats["processed"],
                )
            except Exception as exc:
                log.error("Prediction error at offset %s: %s", msg.offset(), exc)
                stats["errors"] += 1

            # ── Commit offset (at-least-once semantics) ───────────────────────
            consumer.commit(message=msg)

            # ── Periodic stats log ────────────────────────────────────────────
            if stats["processed"] % 100 == 0 and stats["processed"] > 0:
                elapsed = time.time() - t_start
                throughput = stats["processed"] / elapsed
                log.info(
                    "Stats | processed=%d errors=%d high_risk=%d "
                    "skipped=%d throughput=%.1f msg/s",
                    stats["processed"], stats["errors"],
                    stats["high_risk"], stats["skipped"], throughput,
                )

    finally:
        consumer.close()
        kafka_producer.flush()
        elapsed = time.time() - t_start
        log.info(
            "Consumer stopped | processed=%d errors=%d high_risk=%d "
            "skipped=%d elapsed=%.1fs",
            stats["processed"], stats["errors"],
            stats["high_risk"], stats["skipped"], elapsed,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_consumer()
