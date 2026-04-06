# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.preprocessing import LabelEncoder

# =========================
# 页面设置
# =========================
st.set_page_config(page_title="特肖预测系统 Pro++", page_icon="📊", layout="wide")

st.title("📊 特肖预测系统 Pro++")
st.caption("稳定版：动态窗口 + 动态权重 + 冷热修正 + 概率平滑 + 动态TopN")

# =========================
# 可选依赖检测
# =========================
XGB_OK = True
try:
    from xgboost import XGBClassifier
except Exception:
    XGB_OK = False

# =========================
# 基础配置
# =========================
ZODIAC_ORDER = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
COLOR_MAP = {"红": 0, "绿": 1, "蓝": 2}
SIZE_THRESHOLD = 24
NUM_GLOBAL_CLASSES = 12

BASE_NUM_COLS = ["平一", "平二", "平三", "平四", "平五", "平六", "特码"]
BASE_COLOR_COLS = ["平一波", "平二波", "平三波", "平四波", "平五波", "平六波", "特码波"]
BASE_ZODIAC_COLS = ["平一生肖", "平二生肖", "平三生肖", "平四生肖", "平五生肖", "平六生肖", "特码生肖"]

# =========================
# 工具函数
# =========================
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def validate_columns(df: pd.DataFrame):
    required_cols = ["expect", "openTime"] + BASE_NUM_COLS + BASE_COLOR_COLS + BASE_ZODIAC_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少字段: {missing}")


def safe_int(x):
    try:
        return int(str(x).strip())
    except Exception:
        return np.nan


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    if file_name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file, encoding="utf-8-sig")
        except Exception:
            uploaded_file.seek(0)
            try:
                return pd.read_csv(uploaded_file, encoding="gbk")
            except Exception:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file)

    raise ValueError("仅支持 .xlsx / .xls / .csv 文件")


def preprocess_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = clean_column_names(df)
    validate_columns(df)

    df["openTime"] = pd.to_datetime(df["openTime"], errors="coerce")
    df["expect"] = df["expect"].astype(str).str.strip()

    for col in BASE_NUM_COLS:
        df[col] = df[col].apply(safe_int)

    for col in BASE_COLOR_COLS + BASE_ZODIAC_COLS:
        df[col] = df[col].astype(str).str.strip()

    df = df.dropna(subset=BASE_NUM_COLS + ["openTime"])
    df = df.sort_values(["openTime", "expect"]).reset_index(drop=True)
    return df


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pm_cols = ["平一", "平二", "平三", "平四", "平五", "平六"]

    df["平码和"] = df[pm_cols].sum(axis=1)
    df["全和"] = df[BASE_NUM_COLS].sum(axis=1)
    df["平码均值"] = df[pm_cols].mean(axis=1)
    df["平码最大"] = df[pm_cols].max(axis=1)
    df["平码最小"] = df[pm_cols].min(axis=1)
    df["平码跨度"] = df["平码最大"] - df["平码最小"]

    df["平码奇数个数"] = df[pm_cols].apply(lambda row: sum(v % 2 == 1 for v in row), axis=1)
    df["平码偶数个数"] = 6 - df["平码奇数个数"]
    df["特码奇偶"] = df["特码"] % 2

    df["平码大数个数"] = df[pm_cols].apply(lambda row: sum(v >= SIZE_THRESHOLD for v in row), axis=1)
    df["平码小数个数"] = 6 - df["平码大数个数"]
    df["特码大小"] = (df["特码"] >= SIZE_THRESHOLD).astype(int)

    for col in BASE_NUM_COLS:
        df[f"{col}_尾数"] = df[col] % 10

    df["year"] = df["openTime"].dt.year
    df["month"] = df["openTime"].dt.month
    df["day"] = df["openTime"].dt.day
    df["weekday"] = df["openTime"].dt.weekday
    df["hour"] = df["openTime"].dt.hour
    df["minute"] = df["openTime"].dt.minute

    return df


def encode_categories(df: pd.DataFrame):
    df = df.copy()

    for col in BASE_COLOR_COLS:
        df[col] = df[col].map(COLOR_MAP).fillna(-1).astype(int)

    zodiac_encoder = LabelEncoder()
    zodiac_encoder.fit(ZODIAC_ORDER)

    for col in BASE_ZODIAC_COLS:
        df[col] = df[col].apply(lambda x: x if x in ZODIAC_ORDER else ZODIAC_ORDER[0])
        df[col] = zodiac_encoder.transform(df[col])

    return df, zodiac_encoder


def add_history_features(df: pd.DataFrame, windows=(5, 10, 20, 30)) -> pd.DataFrame:
    df = df.copy()

    df["特码_lag1"] = df["特码"].shift(1)
    df["特码_lag2"] = df["特码"].shift(2)
    df["特码_lag3"] = df["特码"].shift(3)
    df["特码_lag4"] = df["特码"].shift(4)
    df["特码_lag5"] = df["特码"].shift(5)

    df["特码生肖_lag1"] = df["特码生肖"].shift(1)
    df["特码生肖_lag2"] = df["特码生肖"].shift(2)
    df["特码生肖_lag3"] = df["特码生肖"].shift(3)
    df["特码生肖_lag4"] = df["特码生肖"].shift(4)
    df["特码生肖_lag5"] = df["特码生肖"].shift(5)

    df["特码波_lag1"] = df["特码波"].shift(1)
    df["特码波_lag2"] = df["特码波"].shift(2)
    df["特码波_lag3"] = df["特码波"].shift(3)

    for w in windows:
        df[f"特码均值_{w}"] = df["特码"].shift(1).rolling(w).mean()
        df[f"特码最大_{w}"] = df["特码"].shift(1).rolling(w).max()
        df[f"特码最小_{w}"] = df["特码"].shift(1).rolling(w).min()
        df[f"特码标准差_{w}"] = df["特码"].shift(1).rolling(w).std()
        df[f"特码奇数比例_{w}"] = df["特码奇偶"].shift(1).rolling(w).mean()
        df[f"特码大数比例_{w}"] = df["特码大小"].shift(1).rolling(w).mean()

    for z_idx, z_name in enumerate(ZODIAC_ORDER):
        flag = (df["特码生肖"] == z_idx).astype(int)
        for w in windows:
            df[f"特码生肖_{z_name}_近{w}期次数"] = flag.shift(1).rolling(w).sum()

    for c_idx, c_name in [(0, "红"), (1, "绿"), (2, "蓝")]:
        flag = (df["特码波"] == c_idx).astype(int)
        for w in windows:
            df[f"特码波_{c_name}_近{w}期次数"] = flag.shift(1).rolling(w).sum()

    pm_zodiac_cols = ["平一生肖", "平二生肖", "平三生肖", "平四生肖", "平五生肖", "平六生肖"]
    for z_idx, z_name in enumerate(ZODIAC_ORDER):
        count_series = (df[pm_zodiac_cols] == z_idx).sum(axis=1)
        for w in windows:
            df[f"平码生肖_{z_name}_近{w}期次数"] = count_series.shift(1).rolling(w).sum()

    pm_color_cols = ["平一波", "平二波", "平三波", "平四波", "平五波", "平六波"]
    for c_idx, c_name in [(0, "红"), (1, "绿"), (2, "蓝")]:
        count_series = (df[pm_color_cols] == c_idx).sum(axis=1)
        for w in windows:
            df[f"平码波_{c_name}_近{w}期次数"] = count_series.shift(1).rolling(w).sum()

    return df


def build_features(df: pd.DataFrame):
    df = preprocess_raw(df)
    df = add_basic_features(df)
    df, zodiac_encoder = encode_categories(df)
    df = add_history_features(df, windows=(5, 10, 20, 30))
    df = df.dropna().reset_index(drop=True)
    return df, zodiac_encoder


def get_feature_columns(df: pd.DataFrame):
    exclude_cols = ["expect", "openTime", "特码生肖"]
    return [c for c in df.columns if c not in exclude_cols]


def time_split_train_valid(df: pd.DataFrame, valid_ratio=0.2):
    n = len(df)
    split_idx = int(n * (1 - valid_ratio))
    train_df = df.iloc[:split_idx].copy()
    valid_df = df.iloc[split_idx:].copy()
    return train_df, valid_df


def safe_topk_accuracy(y_true, y_score, k=4):
    labels = np.arange(y_score.shape[1])
    return top_k_accuracy_score(y_true, y_score, k=k, labels=labels)


def remap_labels_contiguous(y_series: pd.Series):
    unique_labels = sorted(y_series.unique().tolist())
    label_to_local = {label: i for i, label in enumerate(unique_labels)}
    local_to_label = {i: label for label, i in label_to_local.items()}
    y_local = y_series.map(label_to_local).astype(int)
    return y_local, label_to_local, local_to_label


def restore_full_proba(local_proba, local_to_label, num_global_classes=NUM_GLOBAL_CLASSES):
    full_proba = np.zeros((local_proba.shape[0], num_global_classes))
    for local_idx, global_label in local_to_label.items():
        full_proba[:, global_label] = local_proba[:, local_idx]
    return full_proba


def restore_sklearn_proba_to_full(model, raw_proba, num_global_classes=NUM_GLOBAL_CLASSES):
    full_proba = np.zeros((raw_proba.shape[0], num_global_classes))
    for i, cls in enumerate(model.classes_):
        full_proba[:, int(cls)] = raw_proba[:, i]
    return full_proba


def get_recent_slice(df_features, window_size):
    if len(df_features) <= window_size:
        return df_features.copy()
    return df_features.tail(window_size).copy()


def calc_streak_zero(hit_list):
    max_streak = 0
    cur = 0
    for x in hit_list:
        if x == 0:
            cur += 1
            max_streak = max(max_streak, cur)
        else:
            cur = 0
    return max_streak


# =========================
# 模型
# =========================
@st.cache_resource(show_spinner=False)
def train_xgboost_cached(X_train_values, y_local_values, num_local_classes):
    if not XGB_OK:
        return None

    model = XGBClassifier(
        n_estimators=160,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        num_class=num_local_classes,
        eval_metric="mlogloss",
        random_state=42,
        reg_lambda=1.0,
        min_child_weight=2,
        n_jobs=1
    )
    model.fit(X_train_values, y_local_values)
    return model


@st.cache_resource(show_spinner=False)
def train_random_forest_cached(X_train_values, y_train_values):
    model = RandomForestClassifier(
        n_estimators=220,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_values, y_train_values)
    return model


def train_xgboost(X_train, y_train, enable_model=True):
    if not enable_model or not XGB_OK:
        return None, None

    y_local, _, local_to_label = remap_labels_contiguous(y_train)
    model = train_xgboost_cached(X_train.values, y_local.values, len(local_to_label))
    return model, local_to_label


def train_random_forest(X_train, y_train, enable_model=True):
    if not enable_model:
        return None
    return train_random_forest_cached(X_train.values, y_train.values)


def build_model_info_xgb(model, local_to_label):
    if model is None:
        return None
    return {
        "type": "xgb",
        "model": model,
        "local_to_label": local_to_label
    }


def build_model_info_rf(model):
    if model is None:
        return None
    return {
        "type": "rf",
        "model": model
    }


def get_model_proba(model_info, X, num_global_classes=NUM_GLOBAL_CLASSES):
    if model_info is None:
        return None

    if model_info["type"] == "xgb":
        local_proba = model_info["model"].predict_proba(X.values)
        return restore_full_proba(local_proba, model_info["local_to_label"], num_global_classes)

    elif model_info["type"] == "rf":
        raw_proba = model_info["model"].predict_proba(X.values)
        return restore_sklearn_proba_to_full(
            model_info["model"],
            raw_proba,
            num_global_classes=num_global_classes
        )

    else:
        raw_proba = model_info["model"].predict_proba(X.values)
        return restore_sklearn_proba_to_full(
            model_info["model"],
            raw_proba,
            num_global_classes=num_global_classes
        )


def evaluate_model(model_info, X_valid, y_valid, topk=4):
    if model_info is None:
        return None

    proba = get_model_proba(model_info, X_valid)
    pred = np.argmax(proba, axis=1)

    acc = accuracy_score(y_valid, pred)
    topk_acc = safe_topk_accuracy(y_valid, proba, k=topk)

    return {
        "acc": acc,
        "topk_acc": topk_acc,
        "proba": proba
    }


def ensemble_predict_proba(models_with_weights, X, num_global_classes=NUM_GLOBAL_CLASSES):
    total_weight = 0.0
    final_proba = None

    for model_info, weight in models_with_weights:
        if model_info is None or weight <= 0:
            continue

        proba = get_model_proba(model_info, X, num_global_classes)

        if final_proba is None:
            final_proba = proba * weight
        else:
            final_proba += proba * weight

        total_weight += weight

    if final_proba is None or total_weight == 0:
        raise ValueError("没有可用模型用于融合")

    final_proba /= total_weight
    return final_proba


# =========================
# 稳定版 Pro++ 组件
# =========================
def get_recent_frequency_proba(df_features, window_size=20, num_classes=NUM_GLOBAL_CLASSES):
    recent_df = df_features.tail(window_size).copy()
    counts = recent_df["特码生肖"].value_counts().to_dict()

    proba = np.zeros(num_classes, dtype=float)
    total = len(recent_df)

    if total == 0:
        proba[:] = 1.0 / num_classes
        return proba

    for cls in range(num_classes):
        proba[cls] = counts.get(cls, 0) / total

    if proba.sum() == 0:
        proba[:] = 1.0 / num_classes
    else:
        proba = proba / proba.sum()

    return proba


def get_hot_cold_score(df_features, window_size=20, num_classes=NUM_GLOBAL_CLASSES):
    recent_df = df_features.tail(window_size).copy()
    counts = recent_df["特码生肖"].value_counts().to_dict()

    scores = np.zeros(num_classes, dtype=float)
    avg_count = len(recent_df) / num_classes if len(recent_df) > 0 else 1.0

    for cls in range(num_classes):
        c = counts.get(cls, 0)

        if c > avg_count:
            scores[cls] = -0.02 * (c - avg_count)
        elif c < avg_count:
            scores[cls] = 0.02 * (avg_count - c)
        else:
            scores[cls] = 0.0

    return scores


def smooth_proba(proba, alpha=0.12):
    uniform = np.ones_like(proba) / len(proba)
    return (1 - alpha) * proba + alpha * uniform


def build_stable_final_proba(model_proba, freq_proba, hotcold_score, model_weight=0.7, freq_weight=0.3):
    final_proba = model_weight * model_proba + freq_weight * freq_proba
    final_proba = final_proba + hotcold_score
    final_proba = np.clip(final_proba, 1e-9, None)
    final_proba = final_proba / final_proba.sum()
    final_proba = smooth_proba(final_proba, alpha=0.12)
    final_proba = final_proba / final_proba.sum()
    return final_proba


def decide_strategy_state(recent_top4, max_zero_streak):
    if recent_top4 >= 0.45 and max_zero_streak <= 3:
        return "稳定"
    elif recent_top4 >= 0.33 and max_zero_streak <= 5:
        return "一般"
    else:
        return "风险高"


def get_strategy_topn(state):
    if state == "稳定":
        return 4
    elif state == "一般":
        return 5
    else:
        return 6


def get_topn_from_proba(proba_row, zodiac_encoder, topn=4):
    idxs = np.argsort(proba_row)[::-1][:topn]
    rows = []
    for i, idx in enumerate(idxs, start=1):
        rows.append({
            "排名": i,
            "生肖": zodiac_encoder.classes_[idx],
            "概率": round(float(proba_row[idx]), 6)
        })
    return pd.DataFrame(rows)


def get_all_probs_df(proba_row, zodiac_encoder):
    rows = []
    for idx in np.argsort(proba_row)[::-1]:
        rows.append({
            "生肖": zodiac_encoder.classes_[idx],
            "概率": round(float(proba_row[idx]), 6)
        })
    return pd.DataFrame(rows)


# =========================
# 轻量动态优化
# =========================
def simple_backtest_score(df_features, feature_cols, zodiac_encoder, window_size, xgb_weight, rf_weight, eval_last_n=30, freq_window=20):
    sub_df = get_recent_slice(df_features, window_size)

    if len(sub_df) < max(60, eval_last_n + 20):
        return None

    results = []
    start_idx = len(sub_df) - eval_last_n
    if start_idx < 30:
        start_idx = 30

    for i in range(start_idx, len(sub_df)):
        train_df = sub_df.iloc[:i].copy()
        test_df = sub_df.iloc[i:i+1].copy()

        if len(test_df) == 0:
            continue

        X_train = train_df[feature_cols]
        y_train = train_df["特码生肖"]
        X_test = test_df[feature_cols]
        y_test = int(test_df["特码生肖"].iloc[0])

        xgb_model, xgb_local_to_label = train_xgboost(X_train, y_train, enable_model=xgb_weight > 0)
        rf_model = train_random_forest(X_train, y_train, enable_model=rf_weight > 0)

        xgb_info = build_model_info_xgb(xgb_model, xgb_local_to_label) if xgb_model is not None else None
        rf_info = build_model_info_rf(rf_model) if rf_model is not None else None

        models_with_weights = []
        if xgb_info is not None and xgb_weight > 0:
            models_with_weights.append((xgb_info, xgb_weight))
        if rf_info is not None and rf_weight > 0:
            models_with_weights.append((rf_info, rf_weight))

        if not models_with_weights:
            continue

        model_proba = ensemble_predict_proba(models_with_weights, X_test)[0]

        freq_proba = get_recent_frequency_proba(train_df, window_size=freq_window, num_classes=NUM_GLOBAL_CLASSES)
        hotcold_score = get_hot_cold_score(train_df, window_size=freq_window, num_classes=NUM_GLOBAL_CLASSES)

        proba = build_stable_final_proba(
            model_proba=model_proba,
            freq_proba=freq_proba,
            hotcold_score=hotcold_score,
            model_weight=0.7,
            freq_weight=0.3
        )

        pred_top1 = int(np.argmax(proba))
        pred_top4 = list(np.argsort(proba)[::-1][:4])

        hit_top1 = int(y_test == pred_top1)
        hit_top4 = int(y_test in pred_top4)

        results.append({
            "Top1命中": hit_top1,
            "Top4命中": hit_top4
        })

    if not results:
        return None

    bt_df = pd.DataFrame(results)
    top1 = bt_df["Top1命中"].mean()
    top4 = bt_df["Top4命中"].mean()
    streak0 = calc_streak_zero(bt_df["Top4命中"].tolist())

    score = top4 * 0.8 + top1 * 0.2 - streak0 * 0.01

    return {
        "window_size": window_size,
        "xgb_weight": xgb_weight,
        "rf_weight": rf_weight,
        "top1": top1,
        "top4": top4,
        "max_zero_streak": streak0,
        "score": score
    }


def find_best_dynamic_config(df_features, feature_cols, zodiac_encoder, eval_last_n=30, freq_window=20):
    candidate_windows = [80, 100, 120, 150]
    candidate_weights = [
        (0.7, 0.3),
        (0.6, 0.4),
        (0.5, 0.5),
        (0.3, 0.7),
        (0.0, 1.0),
    ]

    if not XGB_OK:
        candidate_weights = [(0.0, 1.0)]

    all_scores = []
    total_tasks = len(candidate_windows) * len(candidate_weights)
    done = 0

    progress_bar = st.progress(0.0)
    status_box = st.empty()

    for w in candidate_windows:
        for xgb_w, rf_w in candidate_weights:
            done += 1
            status_box.info(f"动态优化中：{done}/{total_tasks} ｜ 窗口={w} ｜ XGB={xgb_w:.1f} ｜ RF={rf_w:.1f}")
            progress_bar.progress(done / total_tasks)

            result = simple_backtest_score(
                df_features=df_features,
                feature_cols=feature_cols,
                zodiac_encoder=zodiac_encoder,
                window_size=w,
                xgb_weight=xgb_w,
                rf_weight=rf_w,
                eval_last_n=eval_last_n,
                freq_window=freq_window
            )
            if result is not None:
                all_scores.append(result)

    progress_bar.empty()
    status_box.empty()

    if not all_scores:
        return None, None

    score_df = pd.DataFrame(all_scores).sort_values(
        ["score", "top4", "top1"],
        ascending=False
    ).reset_index(drop=True)

    best = score_df.iloc[0].to_dict()
    return best, score_df


def run_recent_monitor(df_features, feature_cols, zodiac_encoder, best_window, best_xgb_weight, best_rf_weight, eval_last_n=30, freq_window=20, strategy_topn=4):
    sub_df = get_recent_slice(df_features, best_window)

    records = []
    start_idx = len(sub_df) - eval_last_n
    if start_idx < 30:
        start_idx = 30

    for i in range(start_idx, len(sub_df)):
        train_df = sub_df.iloc[:i].copy()
        test_df = sub_df.iloc[i:i+1].copy()

        if len(test_df) == 0:
            continue

        X_train = train_df[feature_cols]
        y_train = train_df["特码生肖"]
        X_test = test_df[feature_cols]
        y_test = int(test_df["特码生肖"].iloc[0])

        xgb_model, xgb_local_to_label = train_xgboost(X_train, y_train, enable_model=best_xgb_weight > 0)
        rf_model = train_random_forest(X_train, y_train, enable_model=best_rf_weight > 0)

        xgb_info = build_model_info_xgb(xgb_model, xgb_local_to_label) if xgb_model is not None else None
        rf_info = build_model_info_rf(rf_model) if rf_model is not None else None

        models_with_weights = []
        if xgb_info is not None and best_xgb_weight > 0:
            models_with_weights.append((xgb_info, best_xgb_weight))
        if rf_info is not None and best_rf_weight > 0:
            models_with_weights.append((rf_info, best_rf_weight))

        if not models_with_weights:
            continue

        model_proba = ensemble_predict_proba(models_with_weights, X_test)[0]

        freq_proba = get_recent_frequency_proba(train_df, window_size=freq_window, num_classes=NUM_GLOBAL_CLASSES)
        hotcold_score = get_hot_cold_score(train_df, window_size=freq_window, num_classes=NUM_GLOBAL_CLASSES)

        proba = build_stable_final_proba(
            model_proba=model_proba,
            freq_proba=freq_proba,
            hotcold_score=hotcold_score,
            model_weight=0.7,
            freq_weight=0.3
        )

        pred_top1 = int(np.argmax(proba))
        pred_topn = list(np.argsort(proba)[::-1][:strategy_topn])

        actual_name = zodiac_encoder.classes_[y_test]
        pred1_name = zodiac_encoder.classes_[pred_top1]
        topn_names = [zodiac_encoder.classes_[x] for x in pred_topn]

        records.append({
            "期号": test_df["expect"].iloc[0],
            "开奖时间": test_df["openTime"].iloc[0],
            "实际生肖": actual_name,
            "预测Top1": pred1_name,
            "策略TopN": strategy_topn,
            "预测列表": ", ".join(topn_names),
            "Top1命中": int(y_test == pred_top1),
            "TopN命中": int(y_test in pred_topn)
        })

    return pd.DataFrame(records)


# =========================
# 侧边栏
# =========================
st.sidebar.header("参数设置")

valid_ratio = st.sidebar.slider("验证集比例", 0.1, 0.4, 0.2, 0.05)
eval_last_n = st.sidebar.slider("动态评估最近期数", 20, 40, 30, 5)
freq_window = st.sidebar.slider("频率/冷热窗口", 10, 30, 20, 5)

uploaded_file = st.file_uploader(
    "上传历史数据文件（Excel / CSV）",
    type=["xlsx", "xls", "csv"]
)

# =========================
# 主逻辑
# =========================
if uploaded_file is not None:
    try:
        raw_df = load_uploaded_file(uploaded_file)
        st.success("文件上传成功")

        with st.expander("查看原始数据前10行", expanded=False):
            st.dataframe(raw_df.head(10), use_container_width=True)

        df_features, zodiac_encoder = build_features(raw_df)

        if len(df_features) < 100:
            st.error("有效样本太少，建议至少100条以上，最好300条以上。")
            st.stop()

        feature_cols = get_feature_columns(df_features)
        st.info(f"有效样本数: {len(df_features)} ｜ 特征数: {len(feature_cols)}")

        if not XGB_OK:
            st.warning("当前环境未检测到 XGBoost，系统会自动使用 RandomForest 保守方案。")

        st.subheader("轻量动态优化")
        best_config, score_df = find_best_dynamic_config(
            df_features=df_features,
            feature_cols=feature_cols,
            zodiac_encoder=zodiac_encoder,
            eval_last_n=eval_last_n,
            freq_window=freq_window
        )

        if best_config is None or score_df is None or len(score_df) == 0:
            st.error("动态优化失败，没有找到可用配置。")
            st.stop()

        st.dataframe(score_df, use_container_width=True)

        best_window = int(best_config["window_size"])
        base_xgb_weight = float(best_config["xgb_weight"])
        base_rf_weight = float(best_config["rf_weight"])
        recent_top4 = float(best_config["top4"])
        max_zero = int(best_config["max_zero_streak"])

        strategy_state = decide_strategy_state(recent_top4, max_zero)
        strategy_topn = get_strategy_topn(strategy_state)

        if strategy_state == "稳定":
            best_xgb_weight = base_xgb_weight
            best_rf_weight = base_rf_weight
            st.success(f"当前状态：{strategy_state}")
        elif strategy_state == "一般":
            best_xgb_weight = min(max(base_xgb_weight, 0.3), 0.5) if XGB_OK else 0.0
            best_rf_weight = 1.0 - best_xgb_weight
            st.warning(f"当前状态：{strategy_state}")
        else:
            best_xgb_weight = 0.0
            best_rf_weight = 1.0
            st.error(f"当前状态：{strategy_state}，建议保守参考")

        st.info(
            f"最佳窗口: {best_window}期 ｜ "
            f"策略状态: {strategy_state} ｜ "
            f"XGB权重: {best_xgb_weight:.2f} ｜ "
            f"RF权重: {best_rf_weight:.2f} ｜ "
            f"最近Top4: {recent_top4:.4f} ｜ "
            f"最大连续不中: {max_zero}"
        )

        dynamic_df = get_recent_slice(df_features, best_window)
        train_df, valid_df = time_split_train_valid(dynamic_df, valid_ratio=valid_ratio)

        X_train = train_df[feature_cols]
        y_train = train_df["特码生肖"]
        X_valid = valid_df[feature_cols]
        y_valid = valid_df["特码生肖"]

        with st.spinner("正在训练最佳配置模型..."):
            xgb_model, xgb_local_to_label = train_xgboost(X_train, y_train, enable_model=best_xgb_weight > 0)
            rf_model = train_random_forest(X_train, y_train, enable_model=best_rf_weight > 0)

        xgb_info = build_model_info_xgb(xgb_model, xgb_local_to_label) if xgb_model is not None else None
        rf_info = build_model_info_rf(rf_model) if rf_model is not None else None

        c1, c2 = st.columns(2)

        xgb_eval = evaluate_model(xgb_info, X_valid, y_valid, topk=4) if xgb_info is not None else None
        rf_eval = evaluate_model(rf_info, X_valid, y_valid, topk=4) if rf_info is not None else None

        with c1:
            st.subheader("XGBoost")
            if xgb_eval:
                st.metric("Top1", f"{xgb_eval['acc']:.4f}")
                st.metric("Top4", f"{xgb_eval['topk_acc']:.4f}")
            else:
                st.warning("未启用")

        with c2:
            st.subheader("RandomForest")
            if rf_eval:
                st.metric("Top1", f"{rf_eval['acc']:.4f}")
                st.metric("Top4", f"{rf_eval['topk_acc']:.4f}")
            else:
                st.warning("未启用")

        models_with_weights = []
        if xgb_info is not None and best_xgb_weight > 0:
            models_with_weights.append((xgb_info, best_xgb_weight))
        if rf_info is not None and best_rf_weight > 0:
            models_with_weights.append((rf_info, best_rf_weight))

        if not models_with_weights:
            st.error("没有可用模型用于最终预测。")
            st.stop()

        ensemble_valid_model_proba = ensemble_predict_proba(models_with_weights, X_valid)

        freq_proba_valid = get_recent_frequency_proba(dynamic_df, window_size=freq_window, num_classes=NUM_GLOBAL_CLASSES)
        hotcold_score_valid = get_hot_cold_score(dynamic_df, window_size=freq_window, num_classes=NUM_GLOBAL_CLASSES)

        stable_valid_proba = np.array([
            build_stable_final_proba(
                model_proba=row,
                freq_proba=freq_proba_valid,
                hotcold_score=hotcold_score_valid,
                model_weight=0.7,
                freq_weight=0.3
            )
            for row in ensemble_valid_model_proba
        ])

        stable_valid_pred = np.argmax(stable_valid_proba, axis=1)
        stable_top1 = accuracy_score(y_valid, stable_valid_pred)
        stable_top4 = safe_topk_accuracy(y_valid, stable_valid_proba, k=4)

        st.subheader("稳定版 Pro++ 融合结果")
        cc1, cc2 = st.columns(2)
        cc1.metric("融合 Top1 Accuracy", f"{stable_top1:.4f}")
        cc2.metric("融合 Top4 Accuracy", f"{stable_top4:.4f}")

        X_next = dynamic_df.iloc[-1:][feature_cols].copy()
        next_model_proba = ensemble_predict_proba(models_with_weights, X_next)[0]

        next_freq_proba = get_recent_frequency_proba(dynamic_df, window_size=freq_window, num_classes=NUM_GLOBAL_CLASSES)
        next_hotcold_score = get_hot_cold_score(dynamic_df, window_size=freq_window, num_classes=NUM_GLOBAL_CLASSES)

        next_proba = build_stable_final_proba(
            model_proba=next_model_proba,
            freq_proba=next_freq_proba,
            hotcold_score=next_hotcold_score,
            model_weight=0.7,
            freq_weight=0.3
        )

        if strategy_topn == 4:
            st.subheader("下一期主推 4 肖")
        elif strategy_topn == 5:
            st.subheader("下一期主推 5 肖")
        else:
            st.subheader("下一期保守推荐 6 肖")

        st.dataframe(
            get_topn_from_proba(next_proba, zodiac_encoder, topn=strategy_topn),
            use_container_width=True
        )

        st.subheader("全部生肖概率排序")
        st.dataframe(get_all_probs_df(next_proba, zodiac_encoder), use_container_width=True)

        st.subheader("最近动态监控明细")
        monitor_df = run_recent_monitor(
            df_features=df_features,
            feature_cols=feature_cols,
            zodiac_encoder=zodiac_encoder,
            best_window=best_window,
            best_xgb_weight=best_xgb_weight,
            best_rf_weight=best_rf_weight,
            eval_last_n=eval_last_n,
            freq_window=freq_window,
            strategy_topn=strategy_topn
        )

        if len(monitor_df) > 0:
            m1, m2, m3 = st.columns(3)
            m1.metric("最近Top1命中率", f"{monitor_df['Top1命中'].mean():.4f}")
            m2.metric("最近TopN命中率", f"{monitor_df['TopN命中'].mean():.4f}")
            m3.metric("最大连续TopN不中", str(calc_streak_zero(monitor_df["TopN命中"].tolist())))

            chart_df = monitor_df.copy()
            chart_df["累计Top1命中率"] = chart_df["Top1命中"].expanding().mean()
            chart_df["累计TopN命中率"] = chart_df["TopN命中"].expanding().mean()
            st.line_chart(chart_df[["累计Top1命中率", "累计TopN命中率"]])

            with st.expander("查看最近监控详细记录", expanded=False):
                st.dataframe(monitor_df, use_container_width=True)
        else:
            st.warning("最近监控结果为空。")

    except Exception as e:
        st.error(f"运行失败: {e}")

else:
    st.warning("请先上传历史数据文件")
    st.markdown("""
### 文件字段必须包含：
- expect
- openTime
- 平一、平二、平三、平四、平五、平六、特码
- 平一波、平二波、平三波、平四波、平五波、平六波、特码波
- 平一生肖、平二生肖、平三生肖、平四生肖、平五生肖、平六生肖、特码生肖
""")
