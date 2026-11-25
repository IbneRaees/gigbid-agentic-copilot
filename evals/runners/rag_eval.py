"""Entrypoint required by the GigBid directory layout; delegates to run_rag_eval."""

from evals.runners.run_rag_eval import main, run_benchmark

__all__ = ["main", "run_benchmark"]

if __name__ == "__main__":
    main()
