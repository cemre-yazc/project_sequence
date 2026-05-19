import numpy as np
import pandas as pd
import random

import torch

from generator import TimeSeriesGenerator


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


# ============================================================
# Single-label class definitions
# These are kept so your old single-label experiment still works.
# ============================================================

CLASS_NAMES = [
    "no_event",
    "mean_shift",
    "variance_shift",
    "trend_shift",
    "point_anomaly",
    "collective_anomaly"
]

LABEL_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_LABEL = {i: name for name, i in LABEL_TO_ID.items()}


# ============================================================
# Multi-label event definitions
# In the multi-label setting, no_event is represented by all zeros.
# ============================================================

EVENT_LABELS = [
    "mean_shift",
    "variance_shift",
    "trend_shift",
    "point_anomaly",
    "collective_anomaly"
]

EVENT_TO_ID = {name: i for i, name in enumerate(EVENT_LABELS)}
ID_TO_EVENT = {i: name for name, i in EVENT_TO_ID.items()}


SCENARIO_GROUPS = {
    "no_event": [
        []
    ],

    "single_event": [
        ["mean_shift"],
        ["variance_shift"],
        ["trend_shift"],
        ["point_anomaly"],
        ["collective_anomaly"]
    ],

    "anomaly_anomaly": [
        ["point_anomaly", "collective_anomaly"]
    ],

    "shift_shift": [
        ["mean_shift", "variance_shift"],
        ["mean_shift", "trend_shift"],
        ["variance_shift", "trend_shift"]
    ],

    "shift_anomaly": [
        ["mean_shift", "point_anomaly"],
        ["mean_shift", "collective_anomaly"],
        ["variance_shift", "point_anomaly"],
        ["variance_shift", "collective_anomaly"],
        ["trend_shift", "point_anomaly"],
        ["trend_shift", "collective_anomaly"]
    ]
}


# ============================================================
# Utility functions
# ============================================================

def z_normalize(x, eps=1e-8, clip_value=10.0):
    """
    Safe per-series z-normalization applied after all backgrounds and events
    are generated.

    Bad generated samples are rejected by raising an error. Since the series
    generation functions are wrapped in try/except retry loops, rejected
    samples are automatically regenerated.
    """
    x = np.asarray(x, dtype=np.float32)

    if not np.all(np.isfinite(x)):
        raise ValueError("Generated series contains NaN or inf before normalization.")

    mean = np.mean(x)
    std = np.std(x)

    if not np.isfinite(mean) or not np.isfinite(std) or std < eps:
        raise ValueError("Generated series has invalid mean/std before normalization.")

    x = (x - mean) / (std + eps)

    if not np.all(np.isfinite(x)):
        raise ValueError("Generated series contains NaN or inf after normalization.")

    x = np.clip(x, -clip_value, clip_value)

    return x.astype(np.float32)


def events_to_multilabel(events):
    """
    Converts a list of event names into a multi-label binary vector.

    Example:
        [] -> [0, 0, 0, 0, 0]
        ["mean_shift", "point_anomaly"] -> [1, 0, 0, 1, 0]
    """
    y = np.zeros(len(EVENT_LABELS), dtype=np.float32)

    for event in events:
        if event not in EVENT_TO_ID:
            raise ValueError(f"Unknown event label: {event}. Valid events: {EVENT_LABELS}")
        y[EVENT_TO_ID[event]] = 1.0

    return y


def _safe_std(values, fallback=1.0):
    std = np.std(values)
    if std < 1e-8 or not np.isfinite(std):
        return fallback
    return std


def add_linear_background(df, split_mode="train"):
    """
    Adds a simple deterministic linear trend.
    Returns the modified df and the slope/intercept used.
    """
    n = len(df)
    t = np.arange(n)

    sign = np.random.choice([-1, 1])

    if split_mode in ["ood_background", "ood_complex"]:
        slope_mag = np.random.uniform(0.05, 0.12)
    else:
        slope_mag = np.random.uniform(0.015, 0.06)

    slope = sign * slope_mag / (n / 100)
    intercept = np.random.uniform(-0.5, 0.5)

    df = df.copy()
    df["data"] = df["data"].values + intercept + slope * t
    df["stationary"] = 0

    return df, {"slope": slope, "intercept": intercept}


def add_single_seasonality_background(df, split_mode="train"):
    """
    Adds one sinusoidal seasonal component on top of the existing series.
    This wrapper version preserves the current base series.
    """
    n = len(df)
    t = np.arange(n)

    possible_periods = [p for p in [12, 24, 30, 52] if p <= n // 6]
    if len(possible_periods) == 0:
        possible_periods = [max(5, n // 10)]

    period = random.choice(possible_periods)

    base_std = _safe_std(df["data"].values)

    if split_mode in ["ood_background", "ood_complex"]:
        amplitude = base_std * np.random.uniform(1.2, 2.2)
    else:
        amplitude = base_std * np.random.uniform(0.5, 1.2)

    phase = np.random.uniform(0, 2 * np.pi)
    seasonal = amplitude * np.sin(2 * np.pi * t / period + phase)

    df = df.copy()
    df["data"] = df["data"].values + seasonal
    df["stationary"] = 0
    df["seasonal"] = 1

    return df, {"seasonal_period": period, "seasonal_amplitude": amplitude}


# Backward-compatible alias used by older code.
def add_seasonal_background(df, split_mode="train"):
    return add_single_seasonality_background(df, split_mode=split_mode)


def add_multiple_seasonality_background(df, split_mode="train", num_components=2):
    """
    Adds multiple sinusoidal seasonal components on top of the existing series.
    """
    n = len(df)
    t = np.arange(n)

    valid_periods = [p for p in [5, 7, 12, 24, 30, 52, 90, 180] if 5 <= p <= n // 6]
    if len(valid_periods) == 0:
        valid_periods = [max(5, n // 10)]

    num_components = min(num_components, len(valid_periods))
    periods = random.sample(valid_periods, num_components)

    base_std = _safe_std(df["data"].values)

    if split_mode in ["ood_background", "ood_complex"]:
        amp_range = (0.8, 1.8)
    else:
        amp_range = (0.3, 1.0)

    seasonal_total = np.zeros(n)
    amplitudes = []

    for period in periods:
        amplitude = base_std * np.random.uniform(*amp_range)
        phase = np.random.uniform(0, 2 * np.pi)
        seasonal_total += amplitude * np.sin(2 * np.pi * t / period + phase)
        amplitudes.append(amplitude)

    df = df.copy()
    df["data"] = df["data"].values + seasonal_total
    df["stationary"] = 0
    df["seasonal"] = 1

    return df, {"seasonal_period": periods, "seasonal_amplitudes": amplitudes}


def add_quadratic_background(df, split_mode="train"):
    """
    Adds a nonlinear quadratic trend on top of the existing series.
    """
    n = len(df)
    t = np.linspace(-1, 1, n)

    base_std = _safe_std(df["data"].values)

    if split_mode in ["ood_background", "ood_complex"]:
        strength = base_std * np.random.uniform(1.0, 2.0)
    else:
        strength = base_std * np.random.uniform(0.4, 1.0)

    sign = np.random.choice([-1, 1])
    location = random.choice(["center", "left", "right"])

    a = sign * strength
    if location == "center":
        b = 0.0
    elif location == "left":
        b = -2 * a * (-0.5)
    else:
        b = -2 * a * (0.5)

    trend = a * t**2 + b * t
    trend = trend - np.mean(trend)

    df = df.copy()
    df["data"] = df["data"].values + trend
    df["stationary"] = 0

    return df, {"quadratic_strength": strength, "quadratic_sign": sign, "quadratic_location": location}


def add_cubic_background(df, split_mode="train"):
    """
    Adds a nonlinear cubic/S-shaped trend on top of the existing series.
    Mostly intended for harder OOD background testing.
    """
    n = len(df)
    t = np.linspace(-1, 1, n)

    base_std = _safe_std(df["data"].values)

    if split_mode in ["ood_background", "ood_complex"]:
        strength = base_std * np.random.uniform(1.0, 2.2)
    else:
        strength = base_std * np.random.uniform(0.5, 1.2)

    sign = np.random.choice([-1, 1])
    profile = t**3 - 0.5 * t
    profile = profile - np.mean(profile)
    profile = profile / (_safe_std(profile))

    trend = sign * strength * profile

    df = df.copy()
    df["data"] = df["data"].values + trend
    df["stationary"] = 0

    return df, {"cubic_strength": strength, "cubic_sign": sign}


def add_volatility_background(df, split_mode="train", gen=None):
    """
    Adds ARCH/GARCH-like volatility as an additional background component.
    This is mostly used in OOD background testing.
    """
    n = len(df)
    if gen is None:
        gen = TimeSeriesGenerator(length=n)

    kind = random.choice(["arch", "garch"])

    if kind == "arch":
        vol_component, vol_info = gen.generate_arch_series(n)
    else:
        vol_component, vol_info = gen.generate_garch_series(n)

    vol_component = np.asarray(vol_component, dtype=np.float32)
    vol_component = (vol_component - np.mean(vol_component)) / (_safe_std(vol_component))

    base_std = _safe_std(df["data"].values)

    if split_mode in ["ood_background", "ood_complex"]:
        strength = base_std * np.random.uniform(0.8, 1.6)
    else:
        strength = base_std * np.random.uniform(0.4, 1.0)

    df = df.copy()
    df["data"] = df["data"].values + strength * vol_component
    df["stationary"] = 0

    return df, {"volatility_kind": kind, "volatility_strength": strength, "volatility_info": vol_info}


def add_background(df, split_mode="train", force_linear=False, gen=None, event_names=None):
    """
    Adds background structure before the target event(s).

    Train / validation backgrounds:
        none, linear, single seasonality, linear + single seasonality,
        quadratic trend

    OOD parameter test backgrounds:
        same background family as train/validation, but event parameters differ

    OOD background test backgrounds:
        harder or less familiar backgrounds: cubic trend, multiple seasonality,
        linear + multiple seasonality, quadratic + single seasonality,
        volatility background

    IMPORTANT compatibility rules:
        - If trend_shift is active, quadratic/cubic trend backgrounds are excluded.
          Trend shift is then applied only on a linear component.
        - If variance_shift is active, volatility_background is excluded.
          This avoids unrealistic or ambiguous variance-shift-on-volatility cases.

    force_linear:
        used when trend_shift is present, because trend_shift requires a known
        slope and intercept. If the selected background is not already linear,
        a linear component is added on top.
    """
    if gen is None:
        gen = TimeSeriesGenerator(length=len(df))

    if event_names is None:
        event_names = []

    event_names = set(event_names)
    has_trend_shift = "trend_shift" in event_names
    has_variance_shift = "variance_shift" in event_names

    bg_info = {
        "background": "none",
        "slope": None,
        "intercept": None,
        "seasonal_period": None,
        "components": []
    }

    if split_mode in ["train", "val"]:
        choices = [
            "none",
            "linear",
            "single_seasonality",
            "linear_single_seasonality",
            "quadratic_trend"
        ]
    elif split_mode == "ood_params":
        choices = [
            "none",
            "linear",
            "single_seasonality",
            "linear_single_seasonality",
            "quadratic_trend"
        ]
    elif split_mode in ["ood_background", "ood_complex"]:
        choices = [
            "cubic_trend",
            "multiple_seasonality",
            "linear_multiple_seasonality",
            "quadratic_single_seasonality",
            "volatility_background"
        ]
    else:
        choices = ["none"]

    # ------------------------------------------------------------
    # Remove unrealistic / conceptually ambiguous combinations.
    # ------------------------------------------------------------
    if has_trend_shift:
        choices = [
            bg for bg in choices
            if bg not in ["quadratic_trend", "cubic_trend", "quadratic_single_seasonality"]
        ]

    if has_variance_shift:
        choices = [
            bg for bg in choices
            if bg != "volatility_background"
        ]

    # Fallback in case all choices were removed by compatibility rules.
    if len(choices) == 0:
        if split_mode in ["ood_background", "ood_complex"]:
            choices = ["multiple_seasonality", "linear_multiple_seasonality"]
        else:
            choices = ["none", "linear", "single_seasonality", "linear_single_seasonality"]

    bg = random.choice(choices)

    def apply_linear_component():
        nonlocal df, bg_info
        df, lin_info = add_linear_background(df, split_mode)
        bg_info.update(lin_info)
        bg_info["components"].append("linear")

    def apply_single_seasonality_component():
        nonlocal df, bg_info
        df, seas_info = add_single_seasonality_background(df, split_mode)
        bg_info["seasonal_period"] = seas_info["seasonal_period"]
        bg_info["components"].append("single_seasonality")
        bg_info.update({k: v for k, v in seas_info.items() if k != "seasonal_period"})

    def apply_multiple_seasonality_component():
        nonlocal df, bg_info
        df, seas_info = add_multiple_seasonality_background(df, split_mode, num_components=2)
        bg_info["seasonal_period"] = seas_info["seasonal_period"]
        bg_info["components"].append("multiple_seasonality")
        bg_info.update({k: v for k, v in seas_info.items() if k != "seasonal_period"})

    if bg == "none":
        pass

    elif bg == "linear":
        apply_linear_component()

    elif bg == "single_seasonality":
        apply_single_seasonality_component()

    elif bg == "linear_single_seasonality":
        apply_linear_component()
        apply_single_seasonality_component()

    elif bg == "quadratic_trend":
        df, quad_info = add_quadratic_background(df, split_mode)
        bg_info["components"].append("quadratic_trend")
        bg_info.update(quad_info)

    elif bg == "cubic_trend":
        df, cubic_info = add_cubic_background(df, split_mode)
        bg_info["components"].append("cubic_trend")
        bg_info.update(cubic_info)

    elif bg == "multiple_seasonality":
        apply_multiple_seasonality_component()

    elif bg == "linear_multiple_seasonality":
        apply_linear_component()
        apply_multiple_seasonality_component()

    elif bg == "quadratic_single_seasonality":
        df, quad_info = add_quadratic_background(df, split_mode)
        bg_info["components"].append("quadratic_trend")
        bg_info.update(quad_info)
        apply_single_seasonality_component()

    elif bg == "volatility_background":
        df, vol_info = add_volatility_background(df, split_mode, gen=gen)
        bg_info["components"].append("volatility_background")
        bg_info.update(vol_info)

    else:
        raise ValueError(f"Unknown background type: {bg}")

    # If trend_shift is present, make sure a linear component exists so that
    # slope/intercept are available for generate_trend_shift().
    if force_linear and bg_info["slope"] is None:
        apply_linear_component()

    if len(bg_info["components"]) == 0:
        bg_info["background"] = "none"
    else:
        bg_info["background"] = "+".join(bg_info["components"])

    return df, bg_info


def get_event_settings(split_mode):
    """
    Controls how event generation differs between in-distribution and OOD
    test sets.
    """
    if split_mode in ["train", "val"]:
        return {
            "scale_factor": 1.0,
            "location": None,
            "collective_shapes": ["rectangular", "gaussian", "triangular", "ramp", "decay"],
            "allow_multiple_breaks": True,
            "allow_multiple_point_anomalies": True,
            "allow_multiple_collective_anomalies": False
        }

    elif split_mode == "ood_params":
        return {
            "scale_factor": 0.7,
            "location": random.choice(["beginning", "middle", "end"]),
            "collective_shapes": ["gaussian", "triangular", "ramp", "decay"],
            "allow_multiple_breaks": True,
            "allow_multiple_point_anomalies": True,
            "allow_multiple_collective_anomalies": True
        }

    elif split_mode == "ood_background":
        return {
            "scale_factor": 1.0,
            "location": random.choice(["beginning", "middle", "end"]),
            "collective_shapes": ["rectangular", "gaussian", "triangular", "ramp", "decay"],
            "allow_multiple_breaks": True,
            "allow_multiple_point_anomalies": True,
            "allow_multiple_collective_anomalies": False
        }

    else:
        return {
            "scale_factor": 1.0,
            "location": None,
            "collective_shapes": ["rectangular"],
            "allow_multiple_breaks": False,
            "allow_multiple_point_anomalies": False,
            "allow_multiple_collective_anomalies": False
        }


def choose_num_breaks(settings):
    """
    Randomly chooses the number of breaks for a shift type.
    This allows each shift label to include both single-break and
    multi-break examples.
    """
    if settings.get("allow_multiple_breaks", False):
        return random.choices([1, 2], weights=[0.75, 0.25], k=1)[0]
    return 1


def choose_num_collective_anomalies(settings):
    """
    Randomly chooses the number of collective anomalous intervals.
    Kept conservative because multiple interval placement can fail more often.
    """
    if settings.get("allow_multiple_collective_anomalies", False):
        return random.choices([1, 2], weights=[0.80, 0.20], k=1)[0]
    return 1


def choose_collective_shapes(num_anomalies, settings):
    """
    Chooses anomaly shapes in a way compatible with your generator:
    - one shape can be repeated
    - or one shape can be provided for each anomaly.
    """
    available_shapes = settings["collective_shapes"]

    if num_anomalies == 1:
        return random.choice(available_shapes)

    # Either repeat one shape or use different shapes for each interval.
    if random.random() < 0.5:
        return [random.choice(available_shapes)]

    return [random.choice(available_shapes) for _ in range(num_anomalies)]


def choose_shift_locations(num_breaks, settings):
    """
    Uses one location setting for all breaks of the same shift type.
    If location is None, your generator samples breakpoints from the safe range.
    """
    return settings["location"]


def choose_signs(num_breaks):
    return [random.choice([-1, 1]) for _ in range(num_breaks)]


def choose_trend_change_types(num_breaks):
    possible_change_types = [
        "direction_change",
        "magnitude_change",
        "direction_and_magnitude_change"
    ]
    return [random.choice(possible_change_types) for _ in range(num_breaks)]


# ============================================================
# Event application helpers
# ============================================================

def apply_mean_shift(gen, df, settings, seasonal_period=None):
    num_breaks = choose_num_breaks(settings)
    signs = choose_signs(num_breaks)

    df, info = gen.generate_mean_shift(
        df,
        num_breaks=num_breaks,
        scale_factor=settings["scale_factor"],
        signs=signs,
        location=choose_shift_locations(num_breaks, settings),
        seasonal_period=seasonal_period
    )

    info["wrapper_event_label"] = "mean_shift"
    return df, info


def apply_variance_shift(gen, df, settings, seasonal_period=None, slope=None, intercept=None):
    num_breaks = choose_num_breaks(settings)
    signs = choose_signs(num_breaks)

    df, info = gen.generate_variance_shift(
        df,
        num_breaks=num_breaks,
        scale_factor=settings["scale_factor"],
        signs=signs,
        location=choose_shift_locations(num_breaks, settings),
        seasonal_period=seasonal_period,
        slope=slope,
        intercept=intercept
    )

    info["wrapper_event_label"] = "variance_shift"
    return df, info


def apply_trend_shift(gen, df, settings, split_mode="train", seasonal_period=None, slope=None, intercept=None):
    if slope is None or intercept is None:
        df, lin_info = add_linear_background(df, split_mode=split_mode)
        slope = lin_info["slope"]
        intercept = lin_info["intercept"]
    else:
        lin_info = {"slope": slope, "intercept": intercept}

    num_breaks = choose_num_breaks(settings)
    change_types = choose_trend_change_types(num_breaks)

    df, info = gen.generate_trend_shift(
        df,
        num_breaks=num_breaks,
        scale_factor=settings["scale_factor"],
        location=choose_shift_locations(num_breaks, settings) if settings["location"] is not None else "middle",
        change_types=change_types,
        slope=slope,
        intercept=intercept,
        seasonal_period=seasonal_period
    )

    info["wrapper_event_label"] = "trend_shift"
    info["linear_info_used"] = lin_info
    return df, info, slope, intercept


def apply_point_anomaly(gen, df, settings):
    if settings.get("allow_multiple_point_anomalies", False) and random.random() < 0.5:
        df, info = gen.generate_point_anomalies(
            df,
            scale_factor=settings["scale_factor"]
        )
        info["point_anomaly_variant"] = "multiple_point"
    else:
        df, info = gen.generate_point_anomaly(
            df,
            location=settings["location"],
            scale_factor=settings["scale_factor"]
        )
        info["point_anomaly_variant"] = "single_point"

    info["wrapper_event_label"] = "point_anomaly"
    return df, info


def apply_collective_anomaly(gen, df, settings):
    num_anomalies = choose_num_collective_anomalies(settings)
    shapes = choose_collective_shapes(num_anomalies, settings)

    df, info = gen.generate_collective_anomalies(
        df,
        num_anomalies=num_anomalies,
        location=settings["location"] if settings["location"] is not None else "middle",
        scale_factor=settings["scale_factor"],
        anomaly_shapes=shapes
    )

    info["wrapper_event_label"] = "collective_anomaly"
    return df, info


# ============================================================
# Single-label generation
# Kept for your first experiment.
# ============================================================

def generate_one_series(label_name, length=400, split_mode="train", max_tries=100):
    """
    Generates one single-label time series.

    This function is kept for the original 6-class softmax experiment.
    It now also uses more diverse event variants, such as multiple point
    anomalies and multiple breaks, while still returning one class id.
    """
    last_error = None

    for _ in range(max_tries):
        try:
            gen = TimeSeriesGenerator(length=length)

            base_distribution = random.choice(["white_noise", "ar", "ma", "arma"])
            df, base_info = gen.generate_stationary_base_series(distribution=base_distribution)

            settings = get_event_settings(split_mode)

            force_linear = label_name == "trend_shift"
            df, bg_info = add_background(
                df,
                split_mode=split_mode,
                force_linear=force_linear,
                gen=gen,
                event_names=[] if label_name == "no_event" else [label_name]
            )

            slope = bg_info["slope"]
            intercept = bg_info["intercept"]
            seasonal_period = bg_info["seasonal_period"]

            event_info = {}

            if label_name == "no_event":
                event_info = {"event": "no_event"}

            elif label_name == "mean_shift":
                df, event_info = apply_mean_shift(
                    gen,
                    df,
                    settings,
                    seasonal_period=seasonal_period
                )

            elif label_name == "variance_shift":
                df, event_info = apply_variance_shift(
                    gen,
                    df,
                    settings,
                    seasonal_period=seasonal_period,
                    slope=slope,
                    intercept=intercept
                )

            elif label_name == "trend_shift":
                df, event_info, slope, intercept = apply_trend_shift(
                    gen,
                    df,
                    settings,
                    split_mode=split_mode,
                    seasonal_period=seasonal_period,
                    slope=slope,
                    intercept=intercept
                )

            elif label_name == "point_anomaly":
                df, event_info = apply_point_anomaly(gen, df, settings)

            elif label_name == "collective_anomaly":
                df, event_info = apply_collective_anomaly(gen, df, settings)

            else:
                raise ValueError(f"Unknown label: {label_name}")

            x = z_normalize(df["data"].values)
            y = LABEL_TO_ID[label_name]

            meta = {
                "label": label_name,
                "split_mode": split_mode,
                "base_distribution": base_distribution,
                "background": bg_info["background"],
                "base_info": str(base_info),
                "background_info": str(bg_info),
                "event_info": str(event_info)
            }

            return x, y, meta

        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"Failed to generate {label_name} after {max_tries} tries. Last error: {last_error}")


def generate_balanced_dataset(n_per_class, length=400, split_mode="train"):
    """
    Generates a balanced single-label dataset.
    Output:
        X shape: (n_samples, length)
        y shape: (n_samples,)
    """
    X, y, meta = [], [], []

    for label_name in CLASS_NAMES:
        for _ in range(n_per_class):
            x_i, y_i, meta_i = generate_one_series(
                label_name=label_name,
                length=length,
                split_mode=split_mode
            )
            X.append(x_i)
            y.append(y_i)
            meta.append(meta_i)

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)

    idx = np.random.permutation(len(y))
    X = X[idx]
    y = y[idx]
    meta = [meta[i] for i in idx]

    return X, y, pd.DataFrame(meta)


# ============================================================
# Multi-label generation
# Main extension for combinations.
# ============================================================

def generate_one_multilabel_series(events, length=400, split_mode="train", max_tries=100):
    """
    Generates one multi-label time series with zero, one, or multiple events.

    Examples:
        []
        ["mean_shift"]
        ["mean_shift", "point_anomaly"]
        ["point_anomaly", "collective_anomaly"]
        ["mean_shift", "variance_shift"]

    Event application order:
        1. mean shift
        2. variance shift
        3. trend shift
        4. point anomaly
        5. collective anomaly

    Shifts are applied before anomalies so that anomalies occur on top of
    the already-modified structural background.
    """
    last_error = None

    for _ in range(max_tries):
        try:
            events = list(events)
            for event in events:
                if event not in EVENT_LABELS:
                    raise ValueError(f"Unknown event: {event}. Valid events: {EVENT_LABELS}")

            gen = TimeSeriesGenerator(length=length)

            base_distribution = random.choice(["white_noise", "ar", "ma", "arma"])
            df, base_info = gen.generate_stationary_base_series(distribution=base_distribution)

            settings = get_event_settings(split_mode)

            force_linear = "trend_shift" in events
            df, bg_info = add_background(
                df,
                split_mode=split_mode,
                force_linear=force_linear,
                gen=gen,
                event_names=events
            )

            slope = bg_info["slope"]
            intercept = bg_info["intercept"]
            seasonal_period = bg_info["seasonal_period"]

            event_infos = []

            if "mean_shift" in events:
                df, info = apply_mean_shift(
                    gen,
                    df,
                    settings,
                    seasonal_period=seasonal_period
                )
                event_infos.append(info)

            if "variance_shift" in events:
                df, info = apply_variance_shift(
                    gen,
                    df,
                    settings,
                    seasonal_period=seasonal_period,
                    slope=slope,
                    intercept=intercept
                )
                event_infos.append(info)

            if "trend_shift" in events:
                df, info, slope, intercept = apply_trend_shift(
                    gen,
                    df,
                    settings,
                    split_mode=split_mode,
                    seasonal_period=seasonal_period,
                    slope=slope,
                    intercept=intercept
                )
                event_infos.append(info)

            if "point_anomaly" in events:
                df, info = apply_point_anomaly(gen, df, settings)
                event_infos.append(info)

            if "collective_anomaly" in events:
                df, info = apply_collective_anomaly(gen, df, settings)
                event_infos.append(info)

            x = z_normalize(df["data"].values)
            y = events_to_multilabel(events)

            scenario_name = "no_event" if len(events) == 0 else "+".join(events)

            meta = {
                "scenario": scenario_name,
                "events": ",".join(events) if len(events) > 0 else "none",
                "split_mode": split_mode,
                "base_distribution": base_distribution,
                "background": bg_info["background"],
                "base_info": str(base_info),
                "background_info": str(bg_info),
                "event_infos": str(event_infos)
            }

            return x, y, meta

        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(
        f"Failed to generate events={events} after {max_tries} tries. Last error: {last_error}"
    )


def get_all_scenarios():
    """
    Returns all scenario patterns as a list of tuples:
        (scenario_group, events)
    """
    all_scenarios = []

    for group_name, scenarios in SCENARIO_GROUPS.items():
        for events in scenarios:
            all_scenarios.append((group_name, events))

    return all_scenarios


def generate_multilabel_dataset(n_per_scenario, length=400, split_mode="train"):
    """
    Generates a balanced multi-label dataset over scenario patterns.

    n_per_scenario means how many samples to create for each scenario
    pattern, for example:
        []
        ["mean_shift"]
        ["mean_shift", "point_anomaly"]

    Output:
        X shape: (n_samples, length)
        Y shape: (n_samples, 5)
    """
    X, Y, meta = [], [], []

    all_scenarios = get_all_scenarios()

    for group_name, events in all_scenarios:
        for _ in range(n_per_scenario):
            x_i, y_i, meta_i = generate_one_multilabel_series(
                events=events,
                length=length,
                split_mode=split_mode
            )

            meta_i["scenario_group"] = group_name

            X.append(x_i)
            Y.append(y_i)
            meta.append(meta_i)

    X = np.stack(X).astype(np.float32)
    Y = np.stack(Y).astype(np.float32)

    idx = np.random.permutation(len(Y))
    X = X[idx]
    Y = Y[idx]
    meta = [meta[i] for i in idx]

    return X, Y, pd.DataFrame(meta)


def generate_multilabel_dataset_by_lengths(n_per_scenario_per_length, lengths, split_mode="train"):
    """
    Optional helper for variable-length multi-label datasets.

    This does not pad sequences. It returns dictionaries indexed by length.
    You can save each length group separately and train with one DataLoader
    per length.
    """
    X_by_length = {}
    Y_by_length = {}
    meta_tables = []

    for length in lengths:
        X, Y, meta = generate_multilabel_dataset(
            n_per_scenario=n_per_scenario_per_length,
            length=length,
            split_mode=split_mode
        )

        X_by_length[length] = X
        Y_by_length[length] = Y

        meta = meta.copy()
        meta["length"] = length
        meta["split_mode"] = split_mode
        meta_tables.append(meta)

    meta_all = pd.concat(meta_tables, ignore_index=True)

    return X_by_length, Y_by_length, meta_all


def summarize_multilabel_metadata(meta):
    """
    Convenience function for checking what was generated.
    """
    print("Scenario counts:")
    print(meta["scenario"].value_counts().sort_index())

    print("\nScenario group counts:")
    print(meta["scenario_group"].value_counts().sort_index())

    print("\nBackground counts:")
    print(meta["background"].value_counts().sort_index())

    print("\nBase distribution counts:")
    print(meta["base_distribution"].value_counts().sort_index())

    print("\nScenario x Background:")
    print(pd.crosstab(meta["scenario"], meta["background"]))

    print("\nScenario x Base Distribution:")
    print(pd.crosstab(meta["scenario"], meta["base_distribution"]))
