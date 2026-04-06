"""
migrate_to_cloud.py — One-time script to copy local Qdrant data to Qdrant Cloud.

Usage:
  1. Set QDRANT_URL and QDRANT_API_KEY in your .env (cloud values)
  2. Run: python migrate_to_cloud.py

This reads from your local embedded Qdrant (qdrant_data/) and uploads
all vectors + payloads to the remote Qdrant Cloud cluster.
"""

import logging
from qdrant_client import QdrantClient, models
from src.config import QDRANT_COLLECTION, QDRANT_URL, QDRANT_API_KEY

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("migrate")

LOCAL_PATH = "qdrant_data"
BATCH_SIZE = 100


def main():
    if not QDRANT_URL or not QDRANT_API_KEY:
        log.error("Set QDRANT_URL and QDRANT_API_KEY in .env before running.")
        return

    # Connect to both
    log.info("Connecting to local Qdrant at '%s'...", LOCAL_PATH)
    local = QdrantClient(path=LOCAL_PATH)

    log.info("Connecting to Qdrant Cloud at '%s'...", QDRANT_URL)
    cloud = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        port=443,
        https=True,
        timeout=60,
        prefer_grpc=False,
    )

    # Get local collection info
    try:
        info = local.get_collection(QDRANT_COLLECTION)
    except Exception:
        log.error("Local collection '%s' not found. Run ingest.py first.", QDRANT_COLLECTION)
        return

    vector_size = info.config.params.vectors.size
    total_points = info.points_count
    log.info("Local collection: %d vectors, %d dimensions", total_points, vector_size)

    # Create collection on cloud (drop if exists)
    try:
        cloud.get_collection(QDRANT_COLLECTION)
        log.info("Cloud collection exists — dropping it for fresh upload...")
        cloud.delete_collection(QDRANT_COLLECTION)
    except Exception:
        pass

    cloud.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )
    log.info("Created cloud collection '%s'", QDRANT_COLLECTION)

    # Scroll through all local points and upload to cloud
    offset = None
    uploaded = 0

    while True:
        points, next_offset = local.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=BATCH_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )

        if not points:
            break

        # Convert to PointStruct for upload
        point_structs = [
            models.PointStruct(
                id=point.id,
                vector=point.vector,
                payload=point.payload,
            )
            for point in points
        ]

        cloud.upsert(
            collection_name=QDRANT_COLLECTION,
            points=point_structs,
        )

        uploaded += len(points)
        log.info("  Uploaded %d / %d points (%.0f%%)", uploaded, total_points, uploaded / total_points * 100)

        offset = next_offset
        if offset is None:
            break

    # Create payload index on cloud
    try:
        cloud.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="metadata.source_doc",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        log.info("Created payload index on 'metadata.source_doc'")
    except Exception:
        pass

    # Verify
    cloud_info = cloud.get_collection(QDRANT_COLLECTION)
    log.info("=" * 50)
    log.info("Migration complete!")
    log.info("  Local:  %d vectors", total_points)
    log.info("  Cloud:  %d vectors", cloud_info.points_count)
    log.info("=" * 50)

    local.close()


if __name__ == "__main__":
    main()
