"""GigSim: Standalone Deterministic Marketplace Simulator for GigBid (AI-4016).

Provides a realistic REST API stand-in for Upwork with SQLite persistence,
Bearer token authentication, per-token rate limiting, Connects ledger tracking,
idempotency-key deduplication, policy enforcement, and simulated client responses.
"""

__version__ = "0.2.0"
__all__ = ["__version__"]