"""
Kafka Producer — SmartDischarge Event Streaming
================================================
Publishes two event streams after every prediction:

  smartdischarge.predictions  — every scored encounter (all risk tiers)
  smartdischarge.alerts       — HIGH-risk encounters only (review queue trigger)

Architecture role:
  FastAPI /predict → KafkaEventProducer.emit_prediction()
                   → smartdischarge.predictions  (downstream dashboards, audit)
                   → smartdischarge.alerts        (care-team notification consumers)

Design choices:
  - confluent-kafka (librdkafka) for production throughput; falls back
    gracefully when broker is unavailable so the API stays live.
  - Messages are JSON-serialised with envelope metadata (schema_version,
    pipeline_stage) so consumers can evolve independently.
  - Producer is a singleton (module-level) — one connection pool shared
    across FastAPI workers.
  - flush() called per-message in dev; batched in prod via linger.ms.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger("smartdischarge.kafka.producer")

# ── Topics ────────────────────────────────────────────────────────────────────
TOPIC_PREDICTIONS = "smartdischarge.predictions"
TOPIC_ALERTS      = "smartdischarge.alerts"

# ── Broker config (override via env vars in Docker / k8s) ────────────────────
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_ENABLED   = os.getenv("KAFKA_ENABLED", "true").lower() == "true"


def _delivery_report(err, msg):
    """Called by librdkafka on produce acknowledgement."""
    if err:
        log.error("Kafka delivery failed | topic=%s partition=%s err=%s",
                  msg.topic(), msg.partition(), err)
    else:
        log.debug("Kafka ACK | topic=%s partition=%s offset=%s",
                  msg.topic(), msg.partition(), msg.offset())


class KafkaEventProducer:
    """
    Thin wrapper around confluent_kafka.Producer.

    Instantiated once at FastAPI startup (lifespan); closed on shutdown.
    Falls back to a no-op stub when Kafka is disabled or broker is unreachable,
    so the prediction API never goes down due to a messaging failure.
    """

    def __init__(self):
        self._producer = None
        self._enabled  = KAFKA_ENABLED

        if not self._enabled:
            log.info("Kafka disabled via KAFKA_ENABLED=false — running stub mode")
            return

        try:
            from confluent_kafka import Producer
            conf = {
                "bootstrap.servers": KAFKA_BOOTSTRAP,
                "client.id":         "smartdischarge-api",
                # Reliability
                "acks":              "all",
                "retries":           3,
                "retry.backoff.ms":  200,
                # Throughput (tune per environment)
                "linger.ms":         5,
                "batch.size":        16384,
                # Compression
                "compression.type":  "snappy",
            }
            self._producer = Producer(conf)
            log.info("Kafka producer connected | brokers=%s", KAFKA_BOOTSTRAP)
        except ImportError:
            log.warning("confluent-kafka not installed — Kafka stub active. "
                        "Add confluent-kafka>=2.3.0 to requirements.txt.")
            self._enabled = False
        except Exception as exc:
            log.warning("Kafka broker unavailable (%s) — stub active", exc)
            self._enabled = False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _publish(self, topic: str, payload: dict[str, Any]) -> None:
        """Serialise payload and produce to topic; swallow errors gracefully."""
        if not self._enabled or self._producer is None:
            log.debug("Kafka stub: would publish to %s | keys=%s",
                      topic, list(payload.keys()))
            return

        envelope = {
            "schema_version": "1.0",
            "pipeline_stage": "api_serving",
            "published_at":   datetime.now(timezone.utc).isoformat(),
            "payload":        payload,
        }
        try:
            self._producer.produce(
                topic=topic,
                key=str(payload.get("encounter_id") or payload.get("request_id", "")),
                value=json.dumps(envelope).encode("utf-8"),
                on_delivery=_delivery_report,
            )
            self._producer.poll(0)          # trigger delivery callbacks
        except Exception as exc:
            log.error("Kafka produce error | topic=%s err=%s", topic, exc)

    # ── Public API ────────────────────────────────────────────────────────────

    def emit_prediction(self, prediction: dict[str, Any]) -> None:
        """
        Publish a scored encounter to the predictions topic.
        Called after every successful /predict response.

        Payload keys: request_id, encounter_id, readmit_probability,
                      risk_tier, needs_review, top_risk_drivers,
                      latency_ms, timestamp, model_version
        """
        self._publish(TOPIC_PREDICTIONS, prediction)

        # High-risk encounters also go to the alerts topic so care-team
        # consumers can react without filtering the full prediction stream.
        if prediction.get("needs_review") or prediction.get("risk_tier") == "HIGH":
            alert_payload = {
                "alert_type":         "HIGH_READMISSION_RISK",
                "encounter_id":       prediction.get("encounter_id"),
                "request_id":         prediction.get("request_id"),
                "readmit_probability": prediction.get("readmit_probability"),
                "top_risk_drivers":   prediction.get("top_risk_drivers", [])[:3],
                "model_version":      prediction.get("model_version"),
                "triggered_at":       datetime.now(timezone.utc).isoformat(),
            }
            self._publish(TOPIC_ALERTS, alert_payload)
            log.info("HIGH-risk alert emitted | encounter=%s prob=%.3f",
                     prediction.get("encounter_id"), prediction.get("readmit_probability", 0))

    def flush(self, timeout: float = 5.0) -> None:
        """Block until all in-flight messages are delivered (call on shutdown)."""
        if self._producer:
            pending = self._producer.flush(timeout=timeout)
            if pending:
                log.warning("Kafka flush: %d messages undelivered after %.1fs", pending, timeout)

    def close(self) -> None:
        self.flush()
        log.info("Kafka producer closed")


# ── Module-level singleton ────────────────────────────────────────────────────
# Imported by server.py:  from src.kafka.producer import kafka_producer
kafka_producer = KafkaEventProducer()
