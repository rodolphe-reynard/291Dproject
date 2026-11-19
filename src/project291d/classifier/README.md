# Ideas

**Maybe combining a classifier and a regression could also be interesting.**
The two models answer different questions from the same measurement.
The classifier asks: is the bridge damaged? and answers with a confidence, for example 70%.
The regressor asks: how damaged is it? and answers with a score, for example 0.52.

## Continuous damage

Damage grows gradually, but a classifier only knows the pure scenarios, so its output between two of them is arbitrary. A regressor trained on interpolated states learns a continuous mapping from the measurement to the severity, so it can score a state it has never seen as a pure scenario.

## Agreement and disagreement are information

The two outputs come from different models, so how well they agree tells the agent how far to trust its observation:

- **Consistent:** 70% damaged and a severity of 0.52 describe the same picture. The agent can act with more confidence.
- **Inconsistent:** a high damage probability with a near-zero severity, or the reverse, signals an ambiguous measurement: noise, unusual weather, or a state unlike the training data. That is a good moment to **inspect** rather than to repair or to ignore.

The agent therefore receives a signal about how reliable its own estimate is, not just two numbers. Inspection is the one action that resolves the doubt, so the mismatch has a natural use: it makes inspecting worth its cost exactly when the models disagree.
