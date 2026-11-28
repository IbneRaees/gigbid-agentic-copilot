# Extension: Connects ROI Optimizer

The **Connects ROI Optimizer** is a pluggable, type‑safe extension that evaluates the financial impact of offering a **connects** (up‑sell) feature in a GigBid proposal.

## What it does
- **Bayesian win‑probability modelling** – updates the base win probability when a connects offer is present.
- **Expected value calculation** – multiplies the (optionally inflated) prize by the posterior win probability.
- **Decision heuristic** – compares the risk‑adjusted expected value against the base prize and returns a simple `accept`/`reject` decision.

## Public API
| Symbol | Type | Description |
|--------|------|-------------|
| `ROIConfig` | `pydantic.BaseModel` | Configuration used by the optimizer. Includes `base_prize`, `connects_multiplier`, `risk_aversion`, `prior_win_prob`.
| `ConnectsROIOptimizer` | class | Core optimizer. Instantiate with a custom `ROIConfig` or use the defaults.
| `ConnectsROIOptimizer.evaluate` | method | Returns an `ROIResult` mapping with `win_probability`, `expected_value`, and `decision`.
| `ConnectsROIOptimizer.run` | method | Extension entry point – the agent runtime calls this with a payload dictionary.

## Integration
Add the extension to the agent’s extension registry (see `agent/extensions/__init__.py`). The runtime will call `ConnectsROIOptimizer.run(payload)` where `payload` contains:
```json
{
  "base_prize": 1500.0,
  "connects": true
}
```
The method returns a dictionary compatible with the LangGraph state.

## Configuration example
```python
from agent.extensions.connects_roi_optimizer import ROIConfig, ConnectsROIOptimizer

custom_cfg = ROIConfig(
    base_prize=2000.0,
    connects_multiplier=1.3,
    risk_aversion=0.8,
    prior_win_prob=0.45,
)
optimizer = ConnectsROIOptimizer(config=custom_cfg)
result = optimizer.evaluate(base_prize=2500.0, connects=True)
print(result)
```

## Testing
Unit tests live in `tests/unit/test_roi_optimizer.py` and verify the Bayesian update, expected‑value math, and decision logic under various configurations.

---
*The extension follows the same plugin contract as other Agent extensions and can be hot‑loaded without restarting the main workflow.*