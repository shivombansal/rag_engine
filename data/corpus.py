"""
Technical corpus for the RAG pipeline benchmark.
10 paragraphs covering system design, scalability, and distributed systems.
"""

DOCUMENTS = [
    {
        "id": "doc_001",
        "title": "Load Balancing and Peak Traffic",
        "text": (
            "The system employs a multi-tier load balancing architecture to handle peak load scenarios. "
            "At the edge layer, an Nginx reverse proxy distributes incoming requests across multiple "
            "application server instances using a weighted round-robin algorithm. During traffic spikes, "
            "the auto-scaling group detects CPU utilization exceeding 70% and provisions new EC2 instances "
            "within 90 seconds. A circuit breaker pattern prevents cascading failures when downstream "
            "services become saturated, gracefully degrading functionality rather than returning errors."
        ),
    },
    {
        "id": "doc_002",
        "title": "Horizontal Scaling Strategy",
        "text": (
            "Horizontal scaling is the primary strategy for managing increased throughput demands. "
            "Stateless application servers are deployed behind a load balancer, enabling seamless addition "
            "of compute nodes without downtime. Session state is externalized to a Redis cluster, ensuring "
            "that any server can handle any request. Database read replicas absorb query load during high-traffic "
            "periods, while write operations are funneled through the primary node with connection pooling "
            "via PgBouncer to limit concurrent connection overhead."
        ),
    },
    {
        "id": "doc_003",
        "title": "Caching Architecture",
        "text": (
            "A layered caching strategy dramatically reduces latency and backend load. The CDN layer caches "
            "static assets and API responses at edge locations globally, achieving cache-hit ratios above 85%. "
            "Application-level caching uses Redis with a TTL-based eviction policy, storing frequently accessed "
            "data such as user sessions, product catalogs, and computation results. Cache invalidation follows "
            "a write-through pattern for consistency-critical data and an eventual-consistency model for "
            "high-throughput scenarios where slight staleness is acceptable."
        ),
    },
    {
        "id": "doc_004",
        "title": "Database Sharding and Partitioning",
        "text": (
            "To manage data at scale, the database layer uses horizontal sharding based on a consistent hashing "
            "algorithm applied to the user_id field. Each shard is a self-contained PostgreSQL cluster with its "
            "own primary and two read replicas. Range-based partitioning is applied within shards on the "
            "created_at timestamp column, enabling efficient time-series queries and partition pruning. "
            "The sharding middleware transparently routes queries to the correct shard without application-layer "
            "awareness, and cross-shard joins are handled by an aggregation service."
        ),
    },
    {
        "id": "doc_005",
        "title": "Asynchronous Processing and Message Queues",
        "text": (
            "Computationally intensive operations are decoupled from the synchronous request path using "
            "Apache Kafka as a distributed message broker. Producers publish events to topic partitions, "
            "and consumer groups process them independently, enabling backpressure management during sudden "
            "demand surges. Dead-letter queues capture failed messages for retry or manual inspection. "
            "The system guarantees at-least-once delivery with idempotency keys preventing duplicate processing. "
            "Worker pools auto-scale based on Kafka consumer lag metrics exposed via Prometheus."
        ),
    },
    {
        "id": "doc_006",
        "title": "Observability and Monitoring",
        "text": (
            "The observability stack consists of three pillars: metrics, logs, and traces. Prometheus scrapes "
            "metrics from all services at 15-second intervals, and Grafana dashboards provide real-time "
            "visibility into request rates, error rates, and latency percentiles (p50, p95, p99). Structured "
            "JSON logs are shipped to Elasticsearch via Filebeat and queried through Kibana. Distributed "
            "tracing is implemented with OpenTelemetry, propagating trace context across service boundaries "
            "to diagnose inter-service latency. Alertmanager fires PagerDuty incidents when SLO burn rates "
            "exceed defined thresholds."
        ),
    },
    {
        "id": "doc_007",
        "title": "API Rate Limiting and Throttling",
        "text": (
            "Rate limiting protects backend services from abusive or runaway clients. A token-bucket algorithm "
            "implemented in the API gateway enforces per-client quotas: 1000 requests per minute for standard "
            "tier and 10,000 for enterprise clients. When a client exceeds its quota, the gateway returns "
            "HTTP 429 with a Retry-After header indicating the reset window. Burst allowances permit short "
            "spikes up to 2x the sustained rate for up to 10 seconds. Rate limit state is stored in a "
            "distributed Redis cluster with millisecond-precision sliding window counters."
        ),
    },
    {
        "id": "doc_008",
        "title": "Microservices Communication Patterns",
        "text": (
            "Inter-service communication uses a hybrid approach: synchronous REST over HTTP/2 for latency-sensitive "
            "reads and asynchronous event-driven messaging for writes and background tasks. Service discovery "
            "is handled by Consul, with health checks every 10 seconds removing unhealthy instances from the "
            "registry. gRPC is used for high-throughput internal communication between data-plane services, "
            "leveraging protocol buffers for efficient binary serialization. The service mesh (Istio) enforces "
            "mutual TLS between pods and provides fine-grained traffic policies including canary deployments "
            "and weighted routing."
        ),
    },
    {
        "id": "doc_009",
        "title": "Data Replication and Fault Tolerance",
        "text": (
            "Fault tolerance is achieved through synchronous multi-region replication for critical data stores. "
            "The primary PostgreSQL cluster replicates to a standby in a secondary availability zone with "
            "sub-second lag using WAL streaming. Automated failover triggers within 30 seconds when the "
            "primary fails a health check, promoted by Patroni with leader election via etcd. Object storage "
            "uses cross-region replication with 99.999999999% durability guarantees. Daily snapshots and "
            "continuous WAL archiving to S3 support point-in-time recovery with an RPO of under 5 minutes."
        ),
    },
    {
        "id": "doc_010",
        "title": "Security and Zero-Trust Architecture",
        "text": (
            "The system follows a zero-trust security model where no network location is inherently trusted. "
            "All internal service-to-service calls require mutual TLS certificates issued by an internal CA, "
            "rotated automatically every 24 hours. API authentication uses short-lived JWT tokens (15-minute "
            "expiry) signed with RS256, validated at the gateway before reaching any service. Role-based "
            "access control (RBAC) governs resource permissions, with policy decisions centralized in Open "
            "Policy Agent. Secrets management is handled by HashiCorp Vault with dynamic credential generation, "
            "eliminating long-lived static secrets from the codebase."
        ),
    },
]