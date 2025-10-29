# app/infrastructure/redis_client.py
from functools import lru_cache
from redis import Redis
from config import settings


@lru_cache
def get_redis() -> Redis[str]:
    if settings.redis_url:
        return Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_timeout=float(settings.redis_socket_timeout),
            health_check_interval=30.0,
        )

    return Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        decode_responses=True,
        socket_timeout=float(settings.redis_socket_timeout),
        health_check_interval=30.0,
    )
