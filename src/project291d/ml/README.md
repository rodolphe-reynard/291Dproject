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

---

# Using `get_environment()` from data/environment.py

`get_environment()` returns an `Environment` object with everything needed to simulate what the sensors would measure:

| Field        | What it is                                                                                                            |
| ------------ | --------------------------------------------------------------------------------------------------------------------- |
| `clean`      | `{scenario: [f1, f2]}`: the natural frequencies of each of the 17 scenarios, with the weather of the test day removed |
| `env_values` | `(n_hours, n_weather)`: the weather of each hour of the healthy year                                                  |
| `env_effect` | `(n_hours, 2)`: how much that weather shifts `f1` and `f2`                                                            |
| `noise_std`  | `(2,)`: sensor noise of each frequency, in Hz                                                                         |

Everything should be already cached in `outputs/ems_psd.pkl`.

# What to give to the RL simulation

Your code should have a way to provide the predictions to the simulation. It could be a function like this:

```python
def estimator(scenario, rng) -> (p_damaged, predicted_severity):
    x = observe(scenario, rng.integers(*test_rows), rng)  # same observe as training
    p = classifier.predict_proba(x[None])[0, 1]
    return p, float(regressor.predict(x[None])[0])
```

---

# Suggestions

## Applying the noise

```python
env = get_environment()

def observe(scenario, row, rng):
    freqs = env.clean[scenario] + env.env_effect[row] + rng.normal(0, env.noise_std)
    return np.concatenate([freqs, env.env_values[row]])
```

The classifier needs the weather as input too, so it can tell a shift caused by cold from a shift caused by damage.

## Making the training and testing sets

```python
rng = np.random.default_rng(0)
n = len(env.env_values)
split = int(0.7 * n)  # train on the first 70 % of the year

def make_dataset(rows, n_per_scenario):
    X, y = [], []
    for scenario in SCENARIOS:
        for _ in range(n_per_scenario):
            X.append(observe(scenario, rng.integers(*rows), rng))
            y.append(scenario)
    return np.array(X), np.array(y)

X_train, y_train = make_dataset((0, split), 300)
X_test, y_test = make_dataset((split, n), 100)  # weather the model never saw
```

Each sample is one scenario under a random hour of weather with fresh sensor noise.
