"""
BUS964 Team 1 Final Report Code


-------
1. Data loading and preprocessing
2. Dataset statistics for the final report
3. Static RFM and K-Means customer segmentation
4. Rolling 6-month RFM monthly panel
5. Rolling-window validation using logistic regression and ROC-AUC
6. Lv3 transition analysis
7. ABC product-mix comparison for Lv3 Up vs Down customers
8. R/F/M variable impact comparison
9. Monetary-controlled Frequency comparison
10. Repeat vs One-time customer comparison

"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler


# =========================================================
# 0. Configuration
# =========================================================
class RetailConfig:
    FILE_PATH = r"C:\Users\garat\OneDrive\바탕 화면\BA\module1\애널리틱스 프로그래밍(김배호 교수님, 금)\20260306_주제선정\데이터셋\4_online_retail_II.xlsx"
    SHEETS = ["Year 2009-2010", "Year 2010-2011"]
    CACHE_PATH = "online_retail_II_cache.pkl"

    COUNTRY = "United Kingdom"
    START_DATE = "2010-01-01"
    END_DATE = "2012-01-01"
    STATIC_START_DATE = "2011-01-01"
    STATIC_END_DATE = "2012-01-01"

    NON_PRODUCT_CODES = ["POST", "D", "M", "BANK CHARGES", "DOT", "C2"]

    GRADE_NAMES = [
        "Lv1.이탈관리",
        "Lv2.관심필요",
        "Lv3.신규성장",
        "Lv4.우수충성",
        "Lv5.최우수VIP",
    ]

    FINAL_K = 5
    K_RANGE = range(2, 11)
    RANDOM_STATE = 42
    N_INIT = 10

    ANALYSIS_MONTH_START = "2011-01-01"
    ANALYSIS_MONTH_END = "2011-12-01"
    ROLLING_MONTHS = 6
    TARGET_GRADE = "Lv3.신규성장"

    LABEL_FONT_SIZE = 13


# =========================================================
# 1. Utilities
# =========================================================
def setup_plot_style(config: RetailConfig) -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.titlesize"] = config.LABEL_FONT_SIZE
    plt.rcParams["axes.labelsize"] = config.LABEL_FONT_SIZE
    plt.rcParams["xtick.labelsize"] = config.LABEL_FONT_SIZE
    plt.rcParams["ytick.labelsize"] = config.LABEL_FONT_SIZE
    plt.rcParams["legend.fontsize"] = config.LABEL_FONT_SIZE


def grade_to_num(series: pd.Series) -> pd.Series:
    mapping = {
        "Lv1.이탈관리": 1,
        "Lv2.관심필요": 2,
        "Lv3.신규성장": 3,
        "Lv4.우수충성": 4,
        "Lv5.최우수VIP": 5,
    }
    return series.map(mapping)


def add_movement_type(df: pd.DataFrame, current_col: str, next_col: str) -> pd.DataFrame:
    out = df.copy()
    out["CurrentGradeNum"] = grade_to_num(out[current_col])
    out["NextGradeNum"] = grade_to_num(out[next_col])

    def _movement(row):
        if row[next_col] == "NoActivity":
            return "NoActivity"
        if pd.isna(row["CurrentGradeNum"]) or pd.isna(row["NextGradeNum"]):
            return "Unknown"
        if row["NextGradeNum"] > row["CurrentGradeNum"]:
            return "Up"
        if row["NextGradeNum"] < row["CurrentGradeNum"]:
            return "Down"
        return "Stay"

    out["MovementType"] = out.apply(_movement, axis=1)
    return out


def save_table(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


# =========================================================
# 2. Data Loading and Preprocessing
# =========================================================
class RetailDataLoader:
    def __init__(self, file_path: str, sheets: List[str], cache_path: Optional[str] = None):
        self.file_path = Path(file_path)
        self.sheets = sheets
        self.cache_path = Path(cache_path) if cache_path else self.file_path.with_suffix(".pkl")

    def load_data(self, use_cache: bool = True) -> pd.DataFrame:
        if use_cache and self.cache_path.exists():
            if not self.file_path.exists() or self.cache_path.stat().st_mtime >= self.file_path.stat().st_mtime:
                print(f"[Load] 캐시 파일 사용: {self.cache_path}")
                return pd.read_pickle(self.cache_path)

        if not self.file_path.exists():
            raise FileNotFoundError(f"Excel file not found: {self.file_path}")

        print("[Load] Excel 원본 파일 로드")
        frames = [pd.read_excel(self.file_path, sheet_name=s) for s in self.sheets]
        df = pd.concat(frames, ignore_index=True)

        if use_cache:
            df.to_pickle(self.cache_path)
            print(f"[Load] 캐시 저장: {self.cache_path}")
        return df


class RetailPreprocessor:
    def __init__(self, config: RetailConfig):
        self.config = config
        self.stats_rows = []

    def _record(self, step: str, df: pd.DataFrame) -> None:
        self.stats_rows.append({
            "Step": step,
            "Rows": len(df),
            "Customers": df["CustomerID"].nunique() if "CustomerID" in df.columns else np.nan,
            "Orders": df["InvoiceNo"].nunique() if "InvoiceNo" in df.columns else np.nan,
            "Revenue": df["Revenue"].sum() if "Revenue" in df.columns else np.nan,
        })

    def preprocess(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = df_raw.copy()
        df.columns = df.columns.str.strip()
        df = df.rename(columns={
            "Invoice": "InvoiceNo",
            "Price": "UnitPrice",
            "Customer ID": "CustomerID",
        })

        required_cols = ["InvoiceNo", "StockCode", "Country", "UnitPrice", "CustomerID", "InvoiceDate", "Quantity"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}. Current columns: {list(df.columns)}")

        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
        df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
        df["InvoiceNo"] = df["InvoiceNo"].astype(str)
        df["StockCode"] = df["StockCode"].astype(str)
        df["Country"] = df["Country"].astype(str)
        df["Revenue"] = df["Quantity"] * df["UnitPrice"]

        self._record("00_raw_loaded", df)

        df = df.dropna(subset=["CustomerID", "InvoiceDate", "StockCode"])
        df["CustomerID"] = df["CustomerID"].astype(float).astype(int).astype(str)
        self._record("01_drop_missing_customer_date_stockcode", df)

        df = df[~df["InvoiceNo"].str.upper().str.startswith("C")]
        self._record("02_remove_cancel_orders", df)

        df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
        self._record("03_remove_non_positive_quantity_price", df)

        df = df[~df["StockCode"].isin(self.config.NON_PRODUCT_CODES)]
        self._record("04_remove_non_product_codes", df)

        df = df[df["Country"] == self.config.COUNTRY]
        self._record("05_filter_country_uk", df)

        df = df[(df["InvoiceDate"] >= pd.to_datetime(self.config.START_DATE)) &
                (df["InvoiceDate"] < pd.to_datetime(self.config.END_DATE))].copy()
        self._record("06_filter_analysis_period", df)

        df["Year"] = df["InvoiceDate"].dt.year
        df["Month"] = df["InvoiceDate"].dt.month
        df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
        df["MonthStart"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()

        stats = pd.DataFrame(self.stats_rows)
        stats["RemovedRowsFromPreviousStep"] = stats["Rows"].shift(1) - stats["Rows"]
        stats["RemovedRowsFromPreviousStep"] = stats["RemovedRowsFromPreviousStep"].fillna(0).astype(int)
        return df, stats


class DatasetReporter:
    @staticmethod
    def summarize(df_raw: pd.DataFrame, df_clean: pd.DataFrame, preprocessing_stats: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        summary = pd.DataFrame([{
            "RawRows": len(df_raw),
            "CleanRows": len(df_clean),
            "RemovedRowsTotal": len(df_raw) - len(df_clean),
            "CleanCustomers": df_clean["CustomerID"].nunique(),
            "CleanOrders": df_clean["InvoiceNo"].nunique(),
            "CleanSKUs": df_clean["StockCode"].nunique(),
            "TotalRevenue": df_clean["Revenue"].sum(),
            "StartDate": df_clean["InvoiceDate"].min(),
            "EndDate": df_clean["InvoiceDate"].max(),
        }])

        yearly = df_clean.groupby("Year").agg(
            Rows=("InvoiceNo", "size"),
            Customers=("CustomerID", "nunique"),
            Orders=("InvoiceNo", "nunique"),
            SKUs=("StockCode", "nunique"),
            Revenue=("Revenue", "sum"),
        ).reset_index()

        variable_info = pd.DataFrame({
            "Column": df_clean.columns,
            "Dtype": [str(df_clean[c].dtype) for c in df_clean.columns],
            "MissingValues": [df_clean[c].isna().sum() for c in df_clean.columns],
            "UniqueValues": [df_clean[c].nunique(dropna=True) for c in df_clean.columns],
        })

        print("\n[Dataset Summary]")
        print(summary)
        print("\n[Preprocessing Statistics]")
        print(preprocessing_stats)
        print("\n[Yearly Summary]")
        print(yearly)
        print("\n[Variable Information]")
        print(variable_info)

        return {
            "dataset_summary": summary,
            "preprocessing_stats": preprocessing_stats,
            "yearly_summary": yearly,
            "variable_info": variable_info,
        }


# =========================================================
# 3. RFM and Segmentation
# =========================================================
class RFMAnalyzer:
    @staticmethod
    def create_rfm(df: pd.DataFrame, snapshot_date: Optional[str] = None) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["CustomerID", "FirstPurchaseDate", "LastPurchaseDate", "Recency", "Frequency", "Monetary"])

        snapshot = pd.to_datetime(snapshot_date) if snapshot_date is not None else df["InvoiceDate"].max() + pd.Timedelta(days=1)
        dates = df.groupby("CustomerID").agg(
            FirstPurchaseDate=("InvoiceDate", "min"),
            LastPurchaseDate=("InvoiceDate", "max"),
        ).reset_index()
        rfm = df.groupby("CustomerID").agg(
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Revenue", "sum"),
        ).reset_index()
        rfm = rfm.merge(dates, on="CustomerID", how="left")
        rfm["Recency"] = ((snapshot - rfm["LastPurchaseDate"]).dt.days / 30).round(1)
        return rfm[["CustomerID", "FirstPurchaseDate", "LastPurchaseDate", "Recency", "Frequency", "Monetary"]]

    @staticmethod
    def create_rfm_full_customer_base(
        df: pd.DataFrame,
        snapshot_date: pd.Timestamp,
        all_customers: List[str],
        rolling_months: int,
    ) -> pd.DataFrame:
        snapshot = pd.to_datetime(snapshot_date)
        base = pd.DataFrame({"CustomerID": sorted(all_customers)})
        if df.empty:
            base["FirstPurchaseDate"] = pd.NaT
            base["LastPurchaseDate"] = pd.NaT
            base["Recency"] = rolling_months + 1
            base["Frequency"] = 0
            base["Monetary"] = 0.0
            return base[["CustomerID", "FirstPurchaseDate", "LastPurchaseDate", "Recency", "Frequency", "Monetary"]]

        rfm = RFMAnalyzer.create_rfm(df, snapshot_date=snapshot)
        rfm = base.merge(rfm, on="CustomerID", how="left")
        rfm["Frequency"] = rfm["Frequency"].fillna(0).astype(int)
        rfm["Monetary"] = rfm["Monetary"].fillna(0.0)
        rfm["Recency"] = rfm["Recency"].fillna(rolling_months + 1)
        return rfm[["CustomerID", "FirstPurchaseDate", "LastPurchaseDate", "Recency", "Frequency", "Monetary"]]

    @staticmethod
    def remove_outliers(rfm: pd.DataFrame, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        freq_cut = rfm["Frequency"].quantile(0.99)
        mon_cut = rfm["Monetary"].quantile(0.99)
        filtered_rfm = rfm[(rfm["Frequency"] <= freq_cut) & (rfm["Monetary"] <= mon_cut)].copy()
        valid_customers = set(filtered_rfm["CustomerID"])
        filtered_df = df[df["CustomerID"].isin(valid_customers)].copy()
        outlier_summary = pd.DataFrame([{
            "OriginalCustomers": len(rfm),
            "FilteredCustomers": len(filtered_rfm),
            "RemovedCustomers": len(rfm) - len(filtered_rfm),
            "Frequency99Cutoff": freq_cut,
            "Monetary99Cutoff": mon_cut,
        }])
        return filtered_rfm, filtered_df, outlier_summary


class CustomerSegmenter:
    def __init__(self, config: RetailConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.cluster_to_grade = {}

    def evaluate_k(self, rfm: pd.DataFrame) -> pd.DataFrame:
        X_scaled = self.scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
        rows = []
        for k in self.config.K_RANGE:
            if len(rfm) <= k:
                continue
            km = KMeans(n_clusters=k, random_state=self.config.RANDOM_STATE, n_init=self.config.N_INIT)
            labels = km.fit_predict(X_scaled)
            rows.append({
                "k": k,
                "Inertia": km.inertia_,
                "Silhouette": silhouette_score(X_scaled, labels),
            })
        result = pd.DataFrame(rows)
        print("\n[K Evaluation]")
        print(result)
        return result

    def fit_clusters(self, rfm: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X = rfm[["Recency", "Frequency", "Monetary"]]
        X_scaled = self.scaler.fit_transform(X)
        km = KMeans(n_clusters=self.config.FINAL_K, random_state=self.config.RANDOM_STATE, n_init=self.config.N_INIT)
        out = rfm.copy()
        out["Cluster"] = km.fit_predict(X_scaled)

        centers = pd.DataFrame(self.scaler.inverse_transform(km.cluster_centers_), columns=["Recency", "Frequency", "Monetary"])
        centers["Cluster"] = range(len(centers))
        centers = centers.sort_values(by=["Monetary", "Frequency", "Recency"], ascending=[True, True, False]).reset_index(drop=True)
        self.cluster_to_grade = {row["Cluster"]: self.config.GRADE_NAMES[i] for i, row in centers.iterrows()}
        out["Customer_Grade"] = out["Cluster"].map(self.cluster_to_grade)

        cluster_summary = out.groupby("Customer_Grade").agg(
            Customers=("CustomerID", "nunique"),
            Mean_Recency=("Recency", "mean"),
            Mean_Frequency=("Frequency", "mean"),
            Mean_Monetary=("Monetary", "mean"),
        ).reset_index()
        print("\n[Static Cluster Summary]")
        print(cluster_summary)
        return out, cluster_summary

    @staticmethod
    def plot_k_result(k_result: pd.DataFrame) -> None:
        if k_result.empty:
            return
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()
        ax1.plot(k_result["k"], k_result["Inertia"], marker="o", label="Elbow (Inertia)")
        ax2.plot(k_result["k"], k_result["Silhouette"], marker="s", linestyle="--", label="Silhouette")
        ax1.set_title("전체 RFM 기준 Elbow / Silhouette")
        ax1.set_xlabel("k")
        ax1.set_ylabel("Elbow (Inertia)")
        ax2.set_ylabel("Silhouette")
        ax1.set_xticks(k_result["k"])
        ax1.tick_params(axis="x", rotation=0)
        lines = ax1.lines + ax2.lines
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="best")
        plt.show()


# =========================================================
# 4. Product ABC Analysis
# =========================================================
class ProductAnalyzer:
    @staticmethod
    def abc_class(cum_ratio: float) -> str:
        if cum_ratio <= 0.80:
            return "A"
        if cum_ratio <= 0.95:
            return "B"
        return "C"

    def analyze_products(self, df: pd.DataFrame) -> pd.DataFrame:
        product_sales = df.groupby("StockCode").agg(
            Quantity=("Quantity", "sum"),
            Revenue=("Revenue", "sum"),
        ).reset_index().sort_values("Revenue", ascending=False)
        product_sales["CumulativeRevenue"] = product_sales["Revenue"].cumsum()
        product_sales["CumulativeRatio"] = product_sales["CumulativeRevenue"] / product_sales["Revenue"].sum()
        product_sales["ABC_Class"] = product_sales["CumulativeRatio"].apply(self.abc_class)
        print("\n[Product ABC Counts]")
        print(product_sales["ABC_Class"].value_counts())
        return product_sales

    @staticmethod
    def plot_abc_share(product_sales: pd.DataFrame) -> None:
        if product_sales.empty:
            return
        share = product_sales.groupby("ABC_Class")["Revenue"].sum()
        share = (share / share.sum()).reindex(["A", "B", "C"])
        share.plot(kind="bar", figsize=(6, 4))
        plt.title("ABC Class Revenue Share")
        plt.xlabel("ABC Class")
        plt.ylabel("Revenue Share")
        plt.xticks(rotation=0)
        plt.show()


class RollingABCWindowAnalyzer:
    def __init__(self, config: RetailConfig):
        self.config = config

    def _build_window_abc_map(self, df: pd.DataFrame, monthly_panel: pd.DataFrame) -> Dict[pd.Timestamp, pd.DataFrame]:
        abc_maps = {}
        windows = monthly_panel[["AnalysisMonth", "WindowStart", "WindowEnd"]].drop_duplicates()
        for _, row in windows.iterrows():
            wdf = df[(df["InvoiceDate"] >= row["WindowStart"]) & (df["InvoiceDate"] < row["WindowEnd"])].copy()
            if wdf.empty:
                abc_maps[row["AnalysisMonth"]] = pd.DataFrame(columns=["StockCode", "ABC_Class"])
                continue
            prod = ProductAnalyzer().analyze_products(wdf)
            abc_maps[row["AnalysisMonth"]] = prod[["StockCode", "ABC_Class"]].copy()
        return abc_maps

    def analyze_lv3_up_down_abc(self, monthly_panel: pd.DataFrame, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        base = prepare_next_grade_panel(monthly_panel)
        lv3 = base[(base["Customer_Grade"] == self.config.TARGET_GRADE) & (base["MovementType"].isin(["Up", "Down"]))].copy()
        if lv3.empty:
            return pd.DataFrame(), pd.DataFrame()

        abc_maps = self._build_window_abc_map(df, monthly_panel)
        rows = []
        for _, row in lv3.iterrows():
            cdf = df[(df["CustomerID"] == row["CustomerID"]) &
                     (df["InvoiceDate"] >= row["WindowStart"]) &
                     (df["InvoiceDate"] < row["WindowEnd"])].copy()
            abc_map = abc_maps.get(row["AnalysisMonth"])
            if cdf.empty or abc_map is None or abc_map.empty:
                continue
            cdf = cdf.merge(abc_map, on="StockCode", how="left").dropna(subset=["ABC_Class"])
            total = cdf["Revenue"].sum()
            if total <= 0:
                continue
            share = cdf.groupby("ABC_Class")["Revenue"].sum().div(total)
            rows.append({
                "CustomerID": row["CustomerID"],
                "AnalysisMonth": row["AnalysisMonth"],
                "MovementType": row["MovementType"],
                "A_Share": share.get("A", 0.0),
                "B_Share": share.get("B", 0.0),
                "C_Share": share.get("C", 0.0),
            })
        detail = pd.DataFrame(rows)
        if detail.empty:
            return detail, pd.DataFrame()
        summary = detail.groupby("MovementType").agg(
            Obs=("CustomerID", "size"),
            Customers=("CustomerID", "nunique"),
            Mean_A_Share=("A_Share", "mean"),
            Mean_B_Share=("B_Share", "mean"),
            Mean_C_Share=("C_Share", "mean"),
        ).reset_index()
        print("\n[Lv3 Up/Down Rolling ABC Summary]")
        print(summary)
        return detail, summary

    @staticmethod
    def plot_lv3_abc_summary(summary: pd.DataFrame) -> None:
        if summary.empty:
            return
        plot_df = summary.set_index("MovementType")[["Mean_A_Share", "Mean_B_Share", "Mean_C_Share"]]
        plot_df.columns = ["A", "B", "C"]
        plot_df.plot(kind="bar", stacked=True, figsize=(8, 5))
        plt.title("Lv3 고객 이동별 6개월 Rolling 상품 구성")
        plt.xlabel("Movement Type")
        plt.ylabel("Mean Revenue Share")
        plt.xticks(rotation=0)
        plt.legend(title="ABC Class")
        plt.show()


# =========================================================
# 5. Rolling Monthly Panel and Transition Analysis
# =========================================================
class RollingMonthlyAnalyzer:
    def __init__(self, config: RetailConfig):
        self.config = config

    def month_range(self) -> pd.DatetimeIndex:
        return pd.date_range(self.config.ANALYSIS_MONTH_START, self.config.ANALYSIS_MONTH_END, freq="MS")

    def build_monthly_panel(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        all_customers = sorted(df["CustomerID"].unique())
        for month_start in self.month_range():
            window_start = month_start - pd.DateOffset(months=self.config.ROLLING_MONTHS)
            window_end = month_start
            window_df = df[(df["InvoiceDate"] >= window_start) & (df["InvoiceDate"] < window_end)].copy()
            rfm = RFMAnalyzer.create_rfm_full_customer_base(window_df, month_start, all_customers, self.config.ROLLING_MONTHS)
            if len(rfm) < self.config.FINAL_K:
                continue
            segmenter = CustomerSegmenter(self.config)
            rfm, _ = segmenter.fit_clusters(rfm)
            rfm["AnalysisMonth"] = month_start
            rfm["AnalysisMonthStr"] = month_start.strftime("%Y-%m")
            rfm["WindowStart"] = window_start
            rfm["WindowEnd"] = window_end
            rows.append(rfm)
        panel = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        print("\n[Rolling Monthly Panel]")
        print(panel.shape)
        return panel

    @staticmethod
    def plot_monthly_grade_counts(monthly_panel: pd.DataFrame, grade_names: List[str]) -> None:
        if monthly_panel.empty:
            return
        counts = monthly_panel.groupby(["AnalysisMonthStr", "Customer_Grade"])["CustomerID"].nunique().reset_index()
        pivot = counts.pivot(index="AnalysisMonthStr", columns="Customer_Grade", values="CustomerID").fillna(0)
        pivot = pivot[[g for g in grade_names if g in pivot.columns]]
        pivot.plot(kind="bar", stacked=True, figsize=(11, 6))
        plt.title("월별 고객등급 구성")
        plt.xlabel("Month")
        plt.ylabel("Number of Customers")
        plt.xticks(rotation=0)
        plt.show()


def prepare_next_grade_panel(monthly_panel: pd.DataFrame) -> pd.DataFrame:
    panel = monthly_panel.copy().sort_values(["CustomerID", "AnalysisMonth"])
    shifted = panel[["CustomerID", "AnalysisMonth", "Customer_Grade"]].copy()
    shifted["AnalysisMonth"] = shifted["AnalysisMonth"] - pd.DateOffset(months=1)
    shifted = shifted.rename(columns={"Customer_Grade": "NextGrade"})
    merged = panel.merge(shifted, on=["CustomerID", "AnalysisMonth"], how="left")
    merged["NextGrade"] = merged["NextGrade"].fillna("NoActivity")
    return add_movement_type(merged, "Customer_Grade", "NextGrade")


class TransitionAnalyzer:
    def __init__(self, config: RetailConfig):
        self.config = config

    def compare_grade_mobility(self, monthly_panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        merged = prepare_next_grade_panel(monthly_panel)
        merged["Moved"] = (merged["Customer_Grade"] != merged["NextGrade"]).astype(int)
        mobility_summary = merged.groupby("Customer_Grade").agg(
            Customers=("CustomerID", "nunique"),
            Obs=("CustomerID", "size"),
            Move_Rate=("Moved", "mean"),
        ).reset_index()
        movement_dist = merged.groupby(["Customer_Grade", "MovementType"]).size().reset_index(name="Count")
        movement_dist["Ratio"] = movement_dist["Count"] / movement_dist.groupby("Customer_Grade")["Count"].transform("sum")
        print("\n[Grade Mobility Summary]")
        print(mobility_summary)
        return mobility_summary, movement_dist, merged

    def lv3_next_month_transition(self, monthly_panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        merged = prepare_next_grade_panel(monthly_panel)
        lv3 = merged[merged["Customer_Grade"] == self.config.TARGET_GRADE]
        count = pd.crosstab(lv3["Customer_Grade"], lv3["NextGrade"], dropna=False)
        ratio = count.div(count.sum(axis=1), axis=0)
        print("\n[Lv3 Next-Month Transition Ratio]")
        print(ratio)
        return count, ratio

    @staticmethod
    def plot_mobility_compare(mobility_summary: pd.DataFrame) -> None:
        if mobility_summary.empty:
            return
        plt.figure(figsize=(8, 5))
        plt.bar(mobility_summary["Customer_Grade"], mobility_summary["Move_Rate"])
        plt.title("등급별 다음달 이동률 비교")
        plt.xlabel("Customer Grade")
        plt.ylabel("Move Rate")
        plt.xticks(rotation=0)
        plt.show()


# =========================================================
# 6. RFM Impact, Controlled F Analysis, Repeat Analysis
# =========================================================
class RFMImpactAnalyzer:
    def __init__(self, config: RetailConfig):
        self.config = config

    def _lv3_base(self, monthly_panel: pd.DataFrame) -> pd.DataFrame:
        merged = prepare_next_grade_panel(monthly_panel)
        return merged[merged["Customer_Grade"] == self.config.TARGET_GRADE].copy()

    @staticmethod
    def _make_binary_group(df: pd.DataFrame, var: str, low_label: str, high_label: str) -> Tuple[pd.DataFrame, float]:
        cut = df[var].median()
        out = df.copy()
        out[f"{var}_Group"] = np.where(out[var] <= cut, low_label, high_label)
        return out, cut

    def analyze_variable(self, monthly_panel: pd.DataFrame, var: str) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        df = self._lv3_base(monthly_panel)
        labels = {
            "Recency": ("LowRecency(최근구매)", "HighRecency(오래전구매)"),
            "Frequency": ("LowFrequency(저빈도)", "HighFrequency(고빈도)"),
            "Monetary": ("LowMonetary(저매출)", "HighMonetary(고매출)"),
        }
        if var not in labels:
            raise ValueError("var must be Recency, Frequency, or Monetary")
        df, cut = self._make_binary_group(df, var, *labels[var])
        group_col = f"{var}_Group"
        summary = df.groupby(group_col).agg(
            Obs=("CustomerID", "size"),
            Customers=("CustomerID", "nunique"),
            Mean_Value=(var, "mean"),
            Up_Rate=("MovementType", lambda x: (x == "Up").mean()),
            Down_Rate=("MovementType", lambda x: (x == "Down").mean()),
            NoActivity_Rate=("MovementType", lambda x: (x == "NoActivity").mean()),
        ).reset_index()
        dist = df.groupby([group_col, "MovementType"]).size().reset_index(name="Count")
        dist["Ratio"] = dist["Count"] / dist.groupby(group_col)["Count"].transform("sum")
        print(f"\n[Lv3 {var} Impact | median cut={cut}]")
        print(summary)
        return summary, dist, cut

    def compare_all(self, monthly_panel: pd.DataFrame) -> Dict[str, Dict[str, pd.DataFrame]]:
        return {var: dict(zip(["summary", "dist", "cut"], self.analyze_variable(monthly_panel, var)))
                for var in ["Recency", "Frequency", "Monetary"]}

    @staticmethod
    def plot_variable_impact(summary: pd.DataFrame, var: str) -> None:
        if summary.empty:
            return
        group_col = summary.columns[0]
        plot_df = summary.set_index(group_col)[["Up_Rate", "Down_Rate"]]
        if var == "Recency":
            order = ["LowRecency(최근구매)", "HighRecency(오래전구매)"]
            plot_df = plot_df.reindex([x for x in order if x in plot_df.index])
        plot_df.plot(kind="bar", figsize=(9, 5))
        plt.title(f"Lv3 내부 {var} 그룹별 다음달 그룹 이동 비교")
        plt.xlabel(var)
        plt.ylabel("Rate")
        plt.xticks(rotation=0)
        plt.show()


class HighMFrequencyAnalyzer:
    def __init__(self, config: RetailConfig):
        self.config = config

    def analyze(self, monthly_panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        lv3 = prepare_next_grade_panel(monthly_panel)
        lv3 = lv3[lv3["Customer_Grade"] == self.config.TARGET_GRADE].copy()
        monetary_cut = lv3["Monetary"].median()
        high_m = lv3[lv3["Monetary"] >= monetary_cut].copy()
        freq_cut = high_m["Frequency"].median()
        high_m["FreqGroup_in_HighM"] = np.where(high_m["Frequency"] >= freq_cut, "HighF_in_HighM", "LowF_in_HighM")
        summary = high_m.groupby("FreqGroup_in_HighM").agg(
            Obs=("CustomerID", "size"),
            Customers=("CustomerID", "nunique"),
            Mean_Frequency=("Frequency", "mean"),
            Mean_Monetary=("Monetary", "mean"),
            Up_Rate=("MovementType", lambda x: (x == "Up").mean()),
            Down_Rate=("MovementType", lambda x: (x == "Down").mean()),
            NoActivity_Rate=("MovementType", lambda x: (x == "NoActivity").mean()),
        ).reset_index()
        dist = high_m.groupby(["FreqGroup_in_HighM", "MovementType"]).size().reset_index(name="Count")
        dist["Ratio"] = dist["Count"] / dist.groupby("FreqGroup_in_HighM")["Count"].transform("sum")
        print("\n[High Monetary group: Frequency Effect]")
        print(f"Lv3 Monetary median cut={monetary_cut}, High-M Frequency median cut={freq_cut}")
        print(summary)
        return summary, dist

    @staticmethod
    def plot(summary: pd.DataFrame) -> None:
        if summary.empty:
            return
        plot_df = summary.set_index("FreqGroup_in_HighM")[["Up_Rate", "Down_Rate"]]
        plot_df.plot(kind="bar", figsize=(8, 5))
        plt.title("같은 소비 수준에서 반복 여부에 따른 그룹 이동 비교")
        plt.xlabel("Frequency Group")
        plt.ylabel("Rate")
        plt.xticks(rotation=0)
        plt.show()


class RepeatCustomerAnalyzer:
    def __init__(self, config: RetailConfig):
        self.config = config

    def compare(self, monthly_panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        lv3 = prepare_next_grade_panel(monthly_panel)
        lv3 = lv3[lv3["Customer_Grade"] == self.config.TARGET_GRADE].copy()
        lv3["RepeatStatus"] = np.where(lv3["Frequency"] >= 2, "Repeat", "OneTime")
        summary = lv3.groupby("RepeatStatus").agg(
            Obs=("CustomerID", "size"),
            Customers=("CustomerID", "nunique"),
            Mean_Recency=("Recency", "mean"),
            Mean_Frequency=("Frequency", "mean"),
            Mean_Monetary=("Monetary", "mean"),
            Up_Rate=("MovementType", lambda x: (x == "Up").mean()),
            Stay_Rate=("MovementType", lambda x: (x == "Stay").mean()),
            Down_Rate=("MovementType", lambda x: (x == "Down").mean()),
            NoActivity_Rate=("MovementType", lambda x: (x == "NoActivity").mean()),
        ).reset_index()
        dist = lv3.groupby(["RepeatStatus", "MovementType"]).size().reset_index(name="Count")
        dist["Ratio"] = dist["Count"] / dist.groupby("RepeatStatus")["Count"].transform("sum")
        print("\n[Repeat vs OneTime Summary]")
        print(summary)
        return lv3, summary, dist

    @staticmethod
    def plot(summary: pd.DataFrame) -> None:
        if summary.empty:
            return
        summary.set_index("RepeatStatus")[["Up_Rate", "Stay_Rate", "Down_Rate"]].plot(kind="bar", figsize=(9, 5))
        plt.title("Lv3 재구매 고객 vs 일회성 고객의 다음달 그룹 이동 비교")
        plt.ylabel("Rate")
        plt.xticks(rotation=0)
        plt.show()


# =========================================================
# 7. Rolling Window AUC Validation
# =========================================================
class RollingRFMPredictor:
    def __init__(self, analysis_start: str = "2011-01-01", analysis_end: str = "2011-10-01"):
        self.analysis_start = pd.to_datetime(analysis_start)
        self.analysis_end = pd.to_datetime(analysis_end)

    def month_range(self) -> pd.DatetimeIndex:
        return pd.date_range(self.analysis_start, self.analysis_end, freq="MS")

    def build_snapshot_panel(self, df: pd.DataFrame, window_months: int) -> pd.DataFrame:
        rows = []
        for month_start in self.month_range():
            window_start = month_start - pd.DateOffset(months=window_months)
            next_month_start = month_start + pd.DateOffset(months=1)
            next_month_end = month_start + pd.DateOffset(months=2)
            window_df = df[(df["InvoiceDate"] >= window_start) & (df["InvoiceDate"] < month_start)].copy()
            if window_df.empty:
                continue
            active_customers = sorted(window_df["CustomerID"].unique())
            rfm = RFMAnalyzer.create_rfm_full_customer_base(window_df, month_start, active_customers, window_months)
            rfm = rfm[rfm["Frequency"] > 0].copy()
            next_df = df[(df["InvoiceDate"] >= next_month_start) & (df["InvoiceDate"] < next_month_end)].copy()
            next_buy = next_df.groupby("CustomerID")["InvoiceNo"].nunique().reset_index(name="NextMonthOrders")
            next_buy["TargetBuyNextMonth"] = (next_buy["NextMonthOrders"] > 0).astype(int)
            rfm = rfm.merge(next_buy[["CustomerID", "TargetBuyNextMonth"]], on="CustomerID", how="left")
            rfm["TargetBuyNextMonth"] = rfm["TargetBuyNextMonth"].fillna(0).astype(int)
            rfm["BaseMonth"] = month_start
            rfm["BaseMonthStr"] = month_start.strftime("%Y-%m")
            rfm["WindowMonths"] = window_months
            rows.append(rfm[["CustomerID", "BaseMonth", "BaseMonthStr", "WindowMonths", "Recency", "Frequency", "Monetary", "TargetBuyNextMonth"]])
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    @staticmethod
    def evaluate_auc(panel: pd.DataFrame, train_end_month: str = "2011-07-01") -> Dict[str, pd.DataFrame]:
        train_end = pd.to_datetime(train_end_month)
        train = panel[panel["BaseMonth"] < train_end].copy()
        test = panel[panel["BaseMonth"] >= train_end].copy()
        features = ["Recency", "Frequency", "Monetary"]
        target = "TargetBuyNextMonth"
        if train.empty or test.empty or train[target].nunique() < 2 or test[target].nunique() < 2:
            return {"overall_auc": np.nan, "monthly_auc_df": pd.DataFrame(), "test_scored_df": test, "train_n": len(train), "test_n": len(test)}
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        model.fit(train[features], train[target])
        test["PredProb"] = model.predict_proba(test[features])[:, 1]
        overall_auc = roc_auc_score(test[target], test["PredProb"])
        monthly = []
        for m, g in test.groupby("BaseMonthStr"):
            auc = roc_auc_score(g[target], g["PredProb"]) if g[target].nunique() >= 2 else np.nan
            monthly.append({"BaseMonthStr": m, "AUC": auc, "N": len(g), "PositiveRate": g[target].mean()})
        return {"overall_auc": overall_auc, "monthly_auc_df": pd.DataFrame(monthly), "test_scored_df": test, "train_n": len(train), "test_n": len(test)}

    def run_all_windows(self, df: pd.DataFrame, windows: Tuple[int, ...] = (3, 6, 12)) -> Dict[str, pd.DataFrame]:
        panels = {}
        summary_rows = []
        monthly_rows = []
        scored_rows = []
        for w in windows:
            panel = self.build_snapshot_panel(df, w)
            panels[w] = panel
            result = self.evaluate_auc(panel)
            summary_rows.append({"WindowMonths": w, "OverallAUC": result["overall_auc"], "TrainN": result["train_n"], "TestN": result["test_n"]})
            if not result["monthly_auc_df"].empty:
                temp = result["monthly_auc_df"].copy(); temp["WindowMonths"] = w; monthly_rows.append(temp)
            if not result["test_scored_df"].empty:
                temp = result["test_scored_df"].copy(); temp["WindowMonths"] = w; scored_rows.append(temp)
        summary = pd.DataFrame(summary_rows).sort_values("WindowMonths")
        monthly = pd.concat(monthly_rows, ignore_index=True) if monthly_rows else pd.DataFrame()
        scored = pd.concat(scored_rows, ignore_index=True) if scored_rows else pd.DataFrame()
        print("\n[Rolling Window AUC Summary]")
        print(summary)
        return {"panels": panels, "summary_df": summary, "monthly_auc_all": monthly, "scored_test_all": scored}

    @staticmethod
    def plot_overall_auc(summary_df: pd.DataFrame) -> None:
        if summary_df.empty:
            return
        labels = summary_df["WindowMonths"].astype(str) + "M"
        values = summary_df["OverallAUC"].astype(float)
        plt.figure(figsize=(7, 5))
        bars = plt.bar(labels, values)
        plt.title("RFM Window별 전체 ROC-AUC 비교")
        plt.xlabel("Window")
        plt.ylabel("ROC-AUC")
        plt.ylim(0, 1)
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{val:.3f}", ha="center", va="bottom")
        plt.show()

    @staticmethod
    def plot_monthly_auc(monthly_auc_all: pd.DataFrame) -> None:
        if monthly_auc_all.empty:
            return
        pivot = monthly_auc_all.pivot(index="BaseMonthStr", columns="WindowMonths", values="AUC").sort_index()
        pivot = pivot.rename(columns=lambda x: f"{x}M")
        plt.figure(figsize=(9, 5))
        for col in pivot.columns:
            plt.plot(pivot.index, pivot[col], marker="o", label=col)
            for x, y in zip(pivot.index, pivot[col]):
                if pd.notna(y):
                    plt.text(x, y + 0.01, f"{y:.3f}", ha="center", va="bottom", fontsize=9)
        plt.title("기준월별 ROC-AUC 비교")
        plt.xlabel("Base Month")
        plt.ylabel("ROC-AUC")
        plt.ylim(0, 1)
        plt.xticks(rotation=0)
        plt.legend()
        plt.show()


# =========================================================
# 8. Final Pipeline
# =========================================================
class RetailAnalysisPipeline:
    def __init__(self, config: RetailConfig):
        self.config = config
        self.loader = RetailDataLoader(config.FILE_PATH, config.SHEETS, config.CACHE_PATH)
        self.preprocessor = RetailPreprocessor(config)

    def run(self) -> Dict[str, object]:
        setup_plot_style(self.config)

        df_raw = self.loader.load_data()
        df_clean, preprocessing_stats = self.preprocessor.preprocess(df_raw)
        dataset_reports = DatasetReporter.summarize(df_raw, df_clean, preprocessing_stats)

        df_static = df_clean[(df_clean["InvoiceDate"] >= pd.to_datetime(self.config.STATIC_START_DATE)) &
                             (df_clean["InvoiceDate"] < pd.to_datetime(self.config.STATIC_END_DATE))].copy()
        rfm_static = RFMAnalyzer.create_rfm(df_static, snapshot_date=self.config.STATIC_END_DATE)
        rfm_static, df_static_filtered, outlier_summary = RFMAnalyzer.remove_outliers(rfm_static, df_static)
        print("\n[Outlier Summary]")
        print(outlier_summary)

        dynamic_customers = set(df_static_filtered["CustomerID"])
        df_dynamic = df_clean[df_clean["CustomerID"].isin(dynamic_customers)].copy()

        segmenter = CustomerSegmenter(self.config)
        k_result = segmenter.evaluate_k(rfm_static)
        segmenter.plot_k_result(k_result)
        rfm_segmented, cluster_summary = segmenter.fit_clusters(rfm_static)

        product_analyzer = ProductAnalyzer()
        product_sales = product_analyzer.analyze_products(df_static_filtered)
        product_analyzer.plot_abc_share(product_sales)

        rolling_analyzer = RollingMonthlyAnalyzer(self.config)
        monthly_panel = rolling_analyzer.build_monthly_panel(df_dynamic)
        rolling_analyzer.plot_monthly_grade_counts(monthly_panel, self.config.GRADE_NAMES)

        predictor = RollingRFMPredictor()
        auc_results = predictor.run_all_windows(df_dynamic)
        predictor.plot_overall_auc(auc_results["summary_df"])
        predictor.plot_monthly_auc(auc_results["monthly_auc_all"])

        transition_analyzer = TransitionAnalyzer(self.config)
        mobility_summary, movement_dist, transition_base = transition_analyzer.compare_grade_mobility(monthly_panel)
        transition_analyzer.plot_mobility_compare(mobility_summary)
        lv3_count, lv3_ratio = transition_analyzer.lv3_next_month_transition(monthly_panel)

        abc_analyzer = RollingABCWindowAnalyzer(self.config)
        lv3_abc_detail, lv3_abc_summary = abc_analyzer.analyze_lv3_up_down_abc(monthly_panel, df_dynamic)
        abc_analyzer.plot_lv3_abc_summary(lv3_abc_summary)

        rfm_impact = RFMImpactAnalyzer(self.config)
        rfm_impact_results = rfm_impact.compare_all(monthly_panel)
        for var in ["Recency", "Frequency", "Monetary"]:
            rfm_impact.plot_variable_impact(rfm_impact_results[var]["summary"], var)

        high_m_freq = HighMFrequencyAnalyzer(self.config)
        high_m_freq_summary, high_m_freq_dist = high_m_freq.analyze(monthly_panel)
        high_m_freq.plot(high_m_freq_summary)

        repeat_analyzer = RepeatCustomerAnalyzer(self.config)
        lv3_repeat_base, repeat_summary, repeat_dist = repeat_analyzer.compare(monthly_panel)
        repeat_analyzer.plot(repeat_summary)

        return {
            **dataset_reports,
            "outlier_summary": outlier_summary,
            "df_clean": df_clean,
            "df_static_filtered": df_static_filtered,
            "df_dynamic": df_dynamic,
            "rfm_static": rfm_static,
            "rfm_segmented": rfm_segmented,
            "cluster_summary": cluster_summary,
            "product_sales": product_sales,
            "monthly_panel": monthly_panel,
            "auc_summary": auc_results["summary_df"],
            "monthly_auc": auc_results["monthly_auc_all"],
            "mobility_summary": mobility_summary,
            "movement_dist": movement_dist,
            "lv3_next_month_count": lv3_count,
            "lv3_next_month_ratio": lv3_ratio,
            "lv3_abc_detail": lv3_abc_detail,
            "lv3_abc_summary": lv3_abc_summary,
            "rfm_impact_results": rfm_impact_results,
            "high_m_freq_summary": high_m_freq_summary,
            "high_m_freq_dist": high_m_freq_dist,
            "repeat_summary": repeat_summary,
            "repeat_dist": repeat_dist,
        }


if __name__ == "__main__":
    config = RetailConfig()
    pipeline = RetailAnalysisPipeline(config)
    results = pipeline.run()

    # Optional: save report tables as CSV files.
    output_dir = Path("report_outputs")
    output_dir.mkdir(exist_ok=True)
    tables_to_save = {
        "dataset_summary": results["dataset_summary"],
        "preprocessing_stats": results["preprocessing_stats"],
        "yearly_summary": results["yearly_summary"],
        "variable_info": results["variable_info"],
        "outlier_summary": results["outlier_summary"],
        "cluster_summary": results["cluster_summary"],
        "auc_summary": results["auc_summary"],
        "monthly_auc": results["monthly_auc"],
        "mobility_summary": results["mobility_summary"],
        "movement_dist": results["movement_dist"],
        "lv3_next_month_ratio": results["lv3_next_month_ratio"].reset_index(),
        "lv3_abc_summary": results["lv3_abc_summary"],
        "high_m_freq_summary": results["high_m_freq_summary"],
        "repeat_summary": results["repeat_summary"],
        "repeat_dist": results["repeat_dist"],
    }
    for name, table in tables_to_save.items():
        if isinstance(table, pd.DataFrame) and not table.empty:
            save_table(table, str(output_dir / f"{name}.csv"))
    print(f"\n[Done] Report tables saved to: {output_dir.resolve()}")

    ##

def add_revenue_share_to_cluster_profile(monthly_panel):
    df = monthly_panel.copy()

    monthly_revenue = (
        df.groupby(["AnalysisMonthStr", "Customer_Grade"])
        .agg(
            Grade_Revenue=("Monetary", "sum")
        )
        .reset_index()
    )

    total_revenue = (
        df.groupby("AnalysisMonthStr")
        .agg(
            Total_Revenue=("Monetary", "sum")
        )
        .reset_index()
    )

    monthly_revenue = monthly_revenue.merge(
        total_revenue,
        on="AnalysisMonthStr",
        how="left"
    )

    monthly_revenue["Revenue_Share"] = (
        monthly_revenue["Grade_Revenue"] / monthly_revenue["Total_Revenue"]
    )

    revenue_share_summary = (
        monthly_revenue.groupby("Customer_Grade")
        .agg(
            Avg_Revenue_Share=("Revenue_Share", "mean")
        )
        .reset_index()
    )

    revenue_share_summary["Avg_Revenue_Share"] = (
        revenue_share_summary["Avg_Revenue_Share"] * 100
    ).round(2)

    return revenue_share_summary


revenue_share_summary = add_revenue_share_to_cluster_profile(
    results["monthly_panel"]
)

cluster_profile = cluster_profile.merge(
    revenue_share_summary,
    on="Customer_Grade",
    how="left"
)

cluster_profile