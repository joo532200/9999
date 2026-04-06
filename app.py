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
st.set_page_config(page_title="特肖预测系统 Pro", page_icon="📊", layout="wide")

st.title("📊 特肖预测系统 Pro")
st.caption("轻量动态优化版：自动选择最近窗口与模型权重，预测下一期 Top4 特肖")

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

# =========================
# 模型
# =========================
@st.cache_resource(show_spinner=False)
def train_xgboost_cached(X_train_values, y_train_values, num_classes):
    if not XGB_OK:
        return None

    model = XGBClassifier(
        n_estimators=160,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=42,
        reg_lambda=1.0,
        min_child_weight=2,
        n_jobs=1
    )
    model.fit(X_train_values, y_train_values)
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


def train_xgboost(X_train, y_train, num_classes, enable_model=True):
    if not enable_model or not XGB_OK:
        return None
    return train_xgboost_cached(X_train.values, y_train.values, num_classes)


def train_random_forest(X_train, y_train, enable_model=True):
    if not enable_model:
        return None
    return train_random_forest_cached(X_train.values, y_train.values)


def evaluate_model(model, X_valid, y_valid, topk=4):
    if model is None:
        return None

    proba = model.predict_proba(X_valid.values)
    pred = np.argmax(proba, axis=1)

    acc = accuracy_score(y_valid, pred)
    topk_acc = safe_topk_accuracy(y_valid, proba, k=topk)

    return {
        "acc": acc,
        "topk_acc": topk_acc,
        "proba": proba
    }


def ensemble_predict_proba(models_with_weights, X):
    total_weight = 0.0
    final_proba = None

    for model, weight in models_with_weights:
        if model is None or weight <= 0:
            continue

        proba = model.predict_proba(X.values)

        if final_proba is None:
            final_proba = proba * weight
        else:
            final_proba += proba * weight

        total_weight += weight

    if final_proba is None or total_weight == 0:
        raise ValueError("没有可用模型用于融合")

    final_proba /= total_weight
    return final_proba


def build_next_issue_feature_row(df_features: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    last_row = df_features.iloc[-1:].copy()
    return last_row[feature_cols].copy()


def get_top4_from_proba(proba_row, zodiac_encoder):
    top4_idx = np.argsort(proba_row)[::-1][:4]
    rows = []
    for i, idx in enumerate(top4_idx, start=1):
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


def simple_backtest_score(df_features, feature_cols, zodiac_encoder, window_size, xgb_weight, rf_weight, eval_last_n=30):
    sub_df = get_recent_slice(df_features, window_size)

    if len(sub_df) < max(60, eval_last_n + 20):
        return None

    results = []
    num_classes = len(zodiac_encoder.classes_)

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

        xgb_model = None
        rf_model = None

        if xgb_weight > 0:
            xgb_model = train_xgboost(X_train, y_train, num_classes, enable_model=True)

        if rf_weight > 0:
            rf_model = train_random_forest(X_train, y_train, enable_model=True)

        models_with_weights = []
        if xgb_model is not None:
            models_with_weights.append((xgb_model, xgb_weight))
        if rf_model is not None:
            models_with_weights.append((rf_model, rf_weight))

        if not models_with_weights:
            continue

        proba = ensemble_predict_proba(models_with_weights, X_test)[0]
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


def find_best_dynamic_config(df_features, feature_cols, zodiac_encoder, eval_last_n=30):
    candidate_windows = [80, 100, 120, 150]
    candidate_weights = [
        (0.7, 0.3),
        (0.6, 0.4),
        (0.5, 0.5),
        (0.0, 1.0),
    ]

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
                eval_last_n=eval_last_n
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


def run_recent_monitor(df_features, feature_cols, zodiac_encoder, best_window, best_xgb_weight, best_rf_weight, eval_last_n=30):
    sub_df = get_recent_slice(df_features, best_window)
    num_classes = len(zodiac_encoder.classes_)

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

        xgb_model = train_xgboost(X_train, y_train, num_classes, enable_model=best_xgb_weight > 0)
        rf_model = train_random_forest(X_train, y_train, enable_model=best_rf_weight > 0)

        models_with_weights = []
        if xgb_model is not None and best_xgb_weight > 0:
            models_with_weights.append((xgb_model, best_xgb_weight))
        if rf_model is not None and best_rf_weight > 0:
            models_with_weights.append((rf_model, best_rf_weight))

        if not models_with_weights:
            continue

        proba = ensemble_predict_proba(models_with_weights, X_test)[0]
        pred_top1 = int(np.argmax(proba))
        pred_top4 = list(np.argsort(proba)[::-1][:4])

        actual_name = zodiac_encoder.classes_[y_test]
        pred1_name = zodiac_encoder.classes_[pred_top1]
        top4_names = [zodiac_encoder.classes_[x] for x in pred_top4]

        records.append({
            "期号": test_df["expect"].iloc[0],
            "开奖时间": test_df["openTime"].iloc[0],
            "实际生肖": actual_name,
            "预测Top1": pred1_name,
            "预测Top4": ", ".join(top4_names),
            "Top1命中": int(y_test == pred_top1),
            "Top4命中": int(y_test in pred_top4)
        })

    return pd.DataFrame(records)

# =========================
# 侧边栏
# =========================
st.sidebar.header("参数设置")

valid_ratio = st.sidebar.slider("验证集比例", 0.1, 0.4, 0.2, 0.05)
eval_last_n = st.sidebar.slider("动态评估最近期数", 20, 40, 30, 5)

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
            st.warning("当前环境未检测到 XGBoost，动态优化会自动退化为 RandomForest 方案。")

        # 动态优化
        st.subheader("轻量动态优化")
        best_config, score_df = find_best_dynamic_config(
            df_features=df_features,
            feature_cols=feature_cols,
            zodiac_encoder=zodiac_encoder,
            eval_last_n=eval_last_n
        )

        if best_config is None or score_df is None or len(score_df) == 0:
            st.error("动态优化失败，没有找到可用配置。")
            st.stop()

        st.dataframe(score_df, use_container_width=True)

        best_window = int(best_config["window_size"])
        best_xgb_weight = float(best_config["xgb_weight"])
        best_rf_weight = float(best_config["rf_weight"])

        st.info(
            f"最佳窗口: {best_window}期 ｜ "
            f"XGB权重: {best_xgb_weight:.2f} ｜ "
            f"RF权重: {best_rf_weight:.2f} ｜ "
            f"最近Top4: {best_config['top4']:.4f}"
        )

        # 风险提示
        recent_top4 = float(best_config["top4"])
        max_zero = int(best_config["max_zero_streak"])

        if recent_top4 >= 0.40 and max_zero <= 3:
            st.success("当前状态：稳定")
        elif recent_top4 >= 0.33 and max_zero <= 5:
            st.warning("当前状态：一般")
        else:
            st.error("当前状态：风险高，近期连续不中概率偏高")

        # 用最佳配置训练最终模型
        dynamic_df = get_recent_slice(df_features, best_window)
        train_df, valid_df = time_split_train_valid(dynamic_df, valid_ratio=valid_ratio)

        X_train = train_df[feature_cols]
        y_train = train_df["特码生肖"]
        X_valid = valid_df[feature_cols]
        y_valid = valid_df["特码生肖"]
        num_classes = len(zodiac_encoder.classes_)

        with st.spinner("正在训练最佳配置模型..."):
            xgb_model = train_xgboost(X_train, y_train, num_classes, enable_model=best_xgb_weight > 0)
            rf_model = train_random_forest(X_train, y_train, enable_model=best_rf_weight > 0)

        c1, c2 = st.columns(2)

        xgb_eval = evaluate_model(xgb_model, X_valid, y_valid, topk=4) if xgb_model is not None else None
        rf_eval = evaluate_model(rf_model, X_valid, y_valid, topk=4) if rf_model is not None else None

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
        if xgb_model is not None and best_xgb_weight > 0:
            models_with_weights.append((xgb_model, best_xgb_weight))
        if rf_model is not None and best_rf_weight > 0:
            models_with_weights.append((rf_model, best_rf_weight))

        if not models_with_weights:
            st.error("没有可用模型用于最终预测。")
            st.stop()

        ensemble_valid_proba = ensemble_predict_proba(models_with_weights, X_valid)
        ensemble_valid_pred = np.argmax(ensemble_valid_proba, axis=1)

        ensemble_acc = accuracy_score(y_valid, ensemble_valid_pred)
        ensemble_top4_acc = safe_topk_accuracy(y_valid, ensemble_valid_proba, k=4)

        st.subheader("最佳配置融合结果")
        cc1, cc2 = st.columns(2)
        cc1.metric("融合 Top1 Accuracy", f"{ensemble_acc:.4f}")
        cc2.metric("融合 Top4 Accuracy", f"{ensemble_top4_acc:.4f}")

        # 下一期预测
        X_next = build_next_issue_feature_row(dynamic_df, feature_cols)
        next_proba = ensemble_predict_proba(models_with_weights, X_next)[0]

        st.subheader("下一期推荐 Top4 特肖")
        st.dataframe(get_top4_from_proba(next_proba, zodiac_encoder), use_container_width=True)

        st.subheader("全部生肖概率排序")
        st.dataframe(get_all_probs_df(next_proba, zodiac_encoder), use_container_width=True)

        # 最近监控明细
        st.subheader("最近动态监控明细")
        monitor_df = run_recent_monitor(
            df_features=df_features,
            feature_cols=feature_cols,
            zodiac_encoder=zodiac_encoder,
            best_window=best_window,
            best_xgb_weight=best_xgb_weight,
            best_rf_weight=best_rf_weight,
            eval_last_n=eval_last_n
        )

        if len(monitor_df) > 0:
            m1, m2, m3 = st.columns(3)
            m1.metric("最近Top1命中率", f"{monitor_df['Top1命中'].mean():.4f}")
            m2.metric("最近Top4命中率", f"{monitor_df['Top4命中'].mean():.4f}")
            m3.metric("最大连续Top4不中", str(calc_streak_zero(monitor_df["Top4命中"].tolist())))

            chart_df = monitor_df.copy()
            chart_df["累计Top1命中率"] = chart_df["Top1命中"].expanding().mean()
            chart_df["累计Top4命中率"] = chart_df["Top4命中"].expanding().mean()
            st.line_chart(chart_df[["累计Top1命中率", "累计Top4命中率"]])

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
