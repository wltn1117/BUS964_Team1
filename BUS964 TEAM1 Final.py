import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.ticker import MultipleLocator
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# =========================================================
# 0. 설정 클래스
# =========================================================
class RetailConfig:
    FILE_PATH = r"C:\Users\garat\OneDrive\바탕 화면\BA\module1\애널리틱스 프로그래밍(김배호 교수님, 금)\20260306_주제선정\데이터셋\4_online_retail_II.xlsx"
    SHEETS = ["Year 2009-2010", "Year 2010-2011"]

    COUNTRY = "United Kingdom"
    START_DATE = "2010-01-01"
    END_DATE = "2012-01-01"

    # 정적 분석은 2011년만
    STATIC_START_DATE = "2011-01-01"
    STATIC_END_DATE = "2012-01-01"

    NON_PRODUCT_CODES = ["POST", "D", "M", "BANK CHARGES", "DOT", "C2"]

    GRADE_NAMES = [
        "Lv1.이탈관리",
        "Lv2.관심필요",
        "Lv3.신규성장",
        "Lv4.우수충성",
        "Lv5.최우수VIP"
    ]

    K_RANGE = range(2, 11)
    FINAL_K = 5
    RANDOM_STATE = 42
    N_INIT = 10

    CACHE_PATH = "online_retail_II_cache.pkl"
    LABEL_FONT_SIZE = 13

    PRE_START = "2010-07-01"
    PRE_END = "2011-01-01"
    POST_START = "2011-07-01"
    POST_END = "2012-01-01"

    ANALYSIS_MONTH_START = "2011-01-01"
    ANALYSIS_MONTH_END = "2011-12-01"

    ROLLING_MONTHS = 6
    TARGET_GRADE = "Lv3.신규성장"


# =========================================================
# 1. 데이터 로더 클래스
# =========================================================
class RetailDataLoader:
    def __init__(self, file_path, sheets, cache_path=None):
        self.file_path = Path(file_path)
        self.sheets = sheets
        self.cache_path = Path(cache_path) if cache_path else self.file_path.with_suffix(".pkl")
        self._memory_cache = None

    def _is_cache_valid(self):
        return self.cache_path.exists() and self.cache_path.stat().st_mtime >= self.file_path.stat().st_mtime

    def load_data(self, use_cache=True):
        if use_cache and self._memory_cache is not None:
            print("메모리 캐시에서 데이터를 불러왔습니다.")
            return self._memory_cache.copy()

        if use_cache and self.cache_path.exists():
            try:
                if self.file_path.exists() and self._is_cache_valid():
                    print(f"파일 캐시에서 데이터를 불러왔습니다: {self.cache_path}")
                    df = pd.read_pickle(self.cache_path)
                    self._memory_cache = df.copy()
                    return df
                elif not self.file_path.exists():
                    print(f"원본 파일이 없어 캐시를 사용합니다: {self.cache_path}")
                    df = pd.read_pickle(self.cache_path)
                    self._memory_cache = df.copy()
                    return df
            except Exception:
                print("캐시 로드 실패, 원본 Excel 파일에서 다시 불러옵니다.")

        return self._load_from_excel(use_cache=use_cache)

    def _load_from_excel(self, use_cache=True):
        if not self.file_path.exists():
            raise FileNotFoundError(f"원본 Excel 파일이 없습니다: {self.file_path}")

        print("원본 Excel 파일에서 데이터를 불러옵니다.")
        df_1 = pd.read_excel(self.file_path, sheet_name=self.sheets[0])
        df_2 = pd.read_excel(self.file_path, sheet_name=self.sheets[1])
        df = pd.concat([df_1, df_2], ignore_index=True)

        if use_cache:
            df.to_pickle(self.cache_path)
            print(f"캐시 파일을 저장했습니다: {self.cache_path}")

        self._memory_cache = df.copy()
        return df


# =========================================================
# 2. 전처리 클래스
# =========================================================
class RetailPreprocessor:
    def __init__(self, country, start_date, end_date, non_product_codes):
        self.country = country
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.non_product_codes = non_product_codes

    def preprocess(self, df):
        df = df.copy()
        df.columns = df.columns.str.strip()

        rename_dict = {}
        if "Invoice" in df.columns:
            rename_dict["Invoice"] = "InvoiceNo"
        if "Price" in df.columns:
            rename_dict["Price"] = "UnitPrice"
        if "Customer ID" in df.columns:
            rename_dict["Customer ID"] = "CustomerID"
        df = df.rename(columns=rename_dict)

        required_cols = [
            "InvoiceNo", "StockCode", "Country",
            "UnitPrice", "CustomerID", "InvoiceDate", "Quantity"
        ]
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"{col} 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")

        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
        df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

        df["InvoiceNo"] = df["InvoiceNo"].astype(str)
        df["StockCode"] = df["StockCode"].astype(str)
        df["Country"] = df["Country"].astype(str)

        if "Description" not in df.columns:
            df["Description"] = ""
        df["Description"] = df["Description"].astype(str)

        df["Revenue"] = df["Quantity"] * df["UnitPrice"]

        print("CustomerID 결측 전:", df["CustomerID"].isna().sum())
        df = df.dropna(subset=["CustomerID", "InvoiceDate", "StockCode"])
        print("CustomerID 결측 제거 후:", df["CustomerID"].isna().sum())

        df["CustomerID"] = df["CustomerID"].astype(float).astype(int).astype(str)

        cancel_count = df["InvoiceNo"].str.upper().str.startswith("C").sum()
        print("cancel_count:", cancel_count)
        df = df[~df["InvoiceNo"].str.upper().str.startswith("C")]

        df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
        df = df[~df["StockCode"].isin(self.non_product_codes)]
        df = df[df["Country"] == self.country]

        df = df[
            (df["InvoiceDate"] >= self.start_date) &
            (df["InvoiceDate"] < self.end_date)
        ].copy()

        df["Year"] = df["InvoiceDate"].dt.year
        df["Month"] = df["InvoiceDate"].dt.month
        df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
        df["MonthStart"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()

        dup_count = df.duplicated().sum()
        print("중복 개수(참고용):", dup_count)

        print("전처리 후 shape:", df.shape)
        print(df.head())
        print(df.info())

        return df


# =========================================================
# 3. 기간 분리 클래스
# =========================================================
class PeriodSplitter:
    def __init__(self, pre_start, pre_end, post_start, post_end):
        self.pre_start = pd.to_datetime(pre_start)
        self.pre_end = pd.to_datetime(pre_end)
        self.post_start = pd.to_datetime(post_start)
        self.post_end = pd.to_datetime(post_end)

    def split(self, df):
        pre_df = df[
            (df["InvoiceDate"] >= self.pre_start) &
            (df["InvoiceDate"] < self.pre_end)
        ].copy()

        post_df = df[
            (df["InvoiceDate"] >= self.post_start) &
            (df["InvoiceDate"] < self.post_end)
        ].copy()

        print("\n[Period Split]")
        print("Pre period shape:", pre_df.shape)
        print("Post period shape:", post_df.shape)

        return pre_df, post_df


# =========================================================
# 4. RFM 분석 클래스
# =========================================================
class RFMAnalyzer:
    def create_rfm(self, df, snapshot_date=None):
        if df.empty:
            return pd.DataFrame(columns=[
                "CustomerID", "FirstPurchaseDate", "LastPurchaseDate",
                "Recency", "Frequency", "Monetary"
            ])

        if snapshot_date is None:
            snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
        else:
            snapshot_date = pd.to_datetime(snapshot_date)

        customer_dates = df.groupby("CustomerID").agg(
            FirstPurchaseDate=("InvoiceDate", "min"),
            LastPurchaseDate=("InvoiceDate", "max")
        ).reset_index()

        rfm = df.groupby("CustomerID").agg(
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Revenue", "sum")
        ).reset_index()

        rfm = rfm.merge(customer_dates, on="CustomerID", how="left")
        rfm["Recency"] = ((snapshot_date - rfm["LastPurchaseDate"]).dt.days / 30).round(1)

        rfm = rfm[
            ["CustomerID", "FirstPurchaseDate", "LastPurchaseDate",
             "Recency", "Frequency", "Monetary"]
        ]
        return rfm

    def create_rfm_full_customer_base(self, df, snapshot_date, all_customers, rolling_months=6):
        snapshot_date = pd.to_datetime(snapshot_date)

        if df.empty:
            rfm = pd.DataFrame({"CustomerID": sorted(all_customers)})
            rfm["FirstPurchaseDate"] = pd.NaT
            rfm["LastPurchaseDate"] = pd.NaT
            rfm["Recency"] = rolling_months + 1
            rfm["Frequency"] = 0
            rfm["Monetary"] = 0.0
            return rfm

        customer_dates = df.groupby("CustomerID").agg(
            FirstPurchaseDate=("InvoiceDate", "min"),
            LastPurchaseDate=("InvoiceDate", "max")
        ).reset_index()

        rfm = df.groupby("CustomerID").agg(
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Revenue", "sum")
        ).reset_index()

        rfm = rfm.merge(customer_dates, on="CustomerID", how="left")
        rfm["Recency"] = ((snapshot_date - rfm["LastPurchaseDate"]).dt.days / 30).round(1)

        base = pd.DataFrame({"CustomerID": sorted(all_customers)})
        rfm = base.merge(rfm, on="CustomerID", how="left")

        rfm["Frequency"] = rfm["Frequency"].fillna(0).astype(int)
        rfm["Monetary"] = rfm["Monetary"].fillna(0.0)
        rfm["Recency"] = rfm["Recency"].fillna(rolling_months + 1)

        return rfm[
            ["CustomerID", "FirstPurchaseDate", "LastPurchaseDate",
             "Recency", "Frequency", "Monetary"]
        ]

    def remove_outliers(self, rfm, df):
        if rfm.empty:
            return rfm.copy(), df.iloc[0:0].copy()

        freq_cut = rfm["Frequency"].quantile(0.99)
        mon_cut = rfm["Monetary"].quantile(0.99)

        rfm = rfm[
            (rfm["Frequency"] <= freq_cut) &
            (rfm["Monetary"] <= mon_cut)
        ].copy()

        print("이상치 제거 후 고객 수:", len(rfm))
        print("Frequency 상한:", freq_cut)
        print("Monetary 상한:", mon_cut)

        valid_customers = set(rfm["CustomerID"])
        df_filtered = df[df["CustomerID"].isin(valid_customers)].copy()

        return rfm, df_filtered


# =========================================================
# 5. 고객 세분화 클래스
# =========================================================
class CustomerSegmenter:
    def __init__(self, k_range, final_k, random_state=42, n_init=10):
        self.k_range = k_range
        self.final_k = final_k
        self.random_state = random_state
        self.n_init = n_init

        self.scaler = StandardScaler()
        self.kmeans = None
        self.cluster_to_grade = {}
        self.inertia = []
        self.silhouette_scores = []
        self.valid_k = []

    def evaluate_k(self, rfm):
        if rfm.empty:
            self.inertia = []
            self.silhouette_scores = []
            self.valid_k = []
            return

        X = rfm[["Recency", "Frequency", "Monetary"]]
        X_scaled = self.scaler.fit_transform(X)

        self.inertia = []
        self.silhouette_scores = []
        self.valid_k = []

        print("\n[전체 RFM 기준 K 평가 결과]")
        for k in self.k_range:
            if len(rfm) <= k:
                continue

            km = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=self.n_init
            )
            labels = km.fit_predict(X_scaled)

            self.valid_k.append(k)
            self.inertia.append(km.inertia_)
            self.silhouette_scores.append(silhouette_score(X_scaled, labels))

            print(
                f"k={k}, inertia={km.inertia_:.2f}, "
                f"silhouette={self.silhouette_scores[-1]:.3f}"
            )

    def plot_k_result(self):
        if not self.valid_k:
            return

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()

        line1 = ax1.plot(
            self.valid_k,
            self.inertia,
            marker="o",
            linestyle="-",
            linewidth=2,
            color="tab:blue",
            label="Elbow (Inertia)"
        )

        line2 = ax2.plot(
            self.valid_k,
            self.silhouette_scores,
            marker="s",
            linestyle="--",
            linewidth=2,
            color="tab:red",
            label="Silhouette Score"
        )

        ax1.set_title("전체 RFM 기준 Elbow / Silhouette", fontsize=16)
        ax1.set_xlabel("k", fontsize=14)

        ax1.set_ylabel("Elbow (Inertia)", fontsize=14, color="tab:blue")
        ax2.set_ylabel("Silhouette Score", fontsize=14, color="tab:red")

        ax1.set_xticks(self.valid_k)
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        ax1.tick_params(axis="x", labelsize=12)
        ax1.tick_params(axis="y", labelsize=12)
        ax2.tick_params(axis="y", labelsize=12)

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, fontsize=12, loc="best")

        plt.show()

    def fit_clusters(self, rfm, grade_names):
        X = rfm[["Recency", "Frequency", "Monetary"]]
        X_scaled = self.scaler.fit_transform(X)

        self.kmeans = KMeans(
            n_clusters=self.final_k,
            random_state=self.random_state,
            n_init=self.n_init
        )
        rfm = rfm.copy()
        rfm["Cluster"] = self.kmeans.fit_predict(X_scaled)

        centers = pd.DataFrame(
            self.scaler.inverse_transform(self.kmeans.cluster_centers_),
            columns=["Recency", "Frequency", "Monetary"]
        )
        centers["Cluster"] = range(len(centers))

        centers = centers.sort_values(
            by=["Monetary", "Frequency", "Recency"],
            ascending=[True, True, False]
        ).reset_index(drop=True)

        self.cluster_to_grade = {}
        for i, row in centers.iterrows():
            self.cluster_to_grade[row["Cluster"]] = grade_names[i]

        rfm["Customer_Grade"] = rfm["Cluster"].map(self.cluster_to_grade)

        cluster_summary = rfm.groupby("Customer_Grade")[["Recency", "Frequency", "Monetary"]].mean()
        print("\n[Cluster Summary]")
        print(cluster_summary)

        return rfm

    def _scatter_by_grade(self, rfm_plot, x_col, y_col, x_label, y_label, title):
        fig, ax = plt.subplots(figsize=(8, 8))

        for grade in sorted(rfm_plot["Customer_Grade"].dropna().unique()):
            subset = rfm_plot[rfm_plot["Customer_Grade"] == grade]
            ax.scatter(
                subset[x_col],
                subset[y_col],
                label=grade,
                alpha=0.5,
                s=20
            )

        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.set_title(title, fontsize=16)

        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_major_locator(MultipleLocator(1.0))
        ax.set_aspect("equal", adjustable="box")

        ax.tick_params(axis="both", labelsize=12)
        ax.legend(fontsize=11)
        plt.show()

    def plot_cluster_result(self, rfm):
        if rfm.empty:
            return

        rfm_plot = rfm.copy()
        rfm_plot["log_Recency"] = np.log1p(rfm_plot["Recency"])
        rfm_plot["log_Frequency"] = np.log1p(rfm_plot["Frequency"])
        rfm_plot["log_Monetary"] = np.log1p(rfm_plot["Monetary"])

        self._scatter_by_grade(
            rfm_plot,
            x_col="log_Monetary",
            y_col="log_Frequency",
            x_label="Monetary (log)",
            y_label="Frequency (log)",
            title="고객 등급별 세분화 결과: FM"
        )

        self._scatter_by_grade(
            rfm_plot,
            x_col="log_Frequency",
            y_col="log_Recency",
            x_label="Frequency (log)",
            y_label="Recency (log)",
            title="고객 등급별 세분화 결과: FR"
        )

        self._scatter_by_grade(
            rfm_plot,
            x_col="log_Monetary",
            y_col="log_Recency",
            x_label="Monetary (log)",
            y_label="Recency (log)",
            title="고객 등급별 세분화 결과: MR"
        )

        self.plot_cluster_result_3d(rfm_plot)

    def plot_cluster_result_3d(self, rfm_plot):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        for grade in sorted(rfm_plot["Customer_Grade"].dropna().unique()):
            subset = rfm_plot[rfm_plot["Customer_Grade"] == grade]
            ax.scatter(
                subset["log_Recency"],
                subset["log_Frequency"],
                subset["log_Monetary"],
                label=grade,
                alpha=0.5,
                s=20
            )

        ax.set_xlabel("Recency (log)", fontsize=12, labelpad=10)
        ax.set_ylabel("Frequency (log)", fontsize=12, labelpad=10)
        ax.set_zlabel("Monetary (log)", fontsize=12, labelpad=10)
        ax.set_title("고객 등급별 세분화 결과: RFM 3D", fontsize=16, pad=20)

        x = rfm_plot["log_Recency"].to_numpy()
        y = rfm_plot["log_Frequency"].to_numpy()
        z = rfm_plot["log_Monetary"].to_numpy()

        max_axis = max(x.max(), y.max(), z.max())

        ax.set_xlim(0, 3)
        ax.set_ylim(0, 6)
        ax.set_zlim(0, max_axis)

        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_major_locator(MultipleLocator(1.0))
        ax.zaxis.set_major_locator(MultipleLocator(1.0))

        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.tick_params(axis="z", labelsize=10)
        ax.legend(fontsize=10)

        plt.show()


# =========================================================
# 6. 상품 분석 클래스
# =========================================================
class ProductAnalyzer:
    @staticmethod
    def abc_class(x):
        if x <= 0.80:
            return "A"
        elif x <= 0.95:
            return "B"
        else:
            return "C"

    def analyze_products(self, df_filtered):
        if df_filtered.empty:
            return pd.DataFrame(columns=[
                "StockCode", "Quantity", "Revenue",
                "CumulativeRevenue", "CumulativeRatio", "ABC_Class"
            ])

        product_sales = df_filtered.groupby("StockCode").agg(
            Quantity=("Quantity", "sum"),
            Revenue=("Revenue", "sum")
        ).reset_index()

        product_sales = product_sales.sort_values(by="Revenue", ascending=False)
        product_sales["CumulativeRevenue"] = product_sales["Revenue"].cumsum()
        product_sales["CumulativeRatio"] = (
            product_sales["CumulativeRevenue"] / product_sales["Revenue"].sum()
        )
        product_sales["ABC_Class"] = product_sales["CumulativeRatio"].apply(self.abc_class)

        print("상위 매출 상품:\n", product_sales.head(10))
        print("A/B/C 상품 개수:\n", product_sales["ABC_Class"].value_counts())

        return product_sales

    def plot_abc_share(self, product_sales):
        if product_sales.empty:
            return

        abc_revenue_share = product_sales.groupby("ABC_Class")["Revenue"].sum()
        abc_revenue_share = abc_revenue_share / abc_revenue_share.sum()
        abc_revenue_share = abc_revenue_share.reindex(["A", "B", "C"])

        plt.figure(figsize=(6, 4))
        abc_revenue_share.plot(kind="bar")
        plt.title("ABC Class Revenue Share")
        plt.xlabel("ABC Class")
        plt.ylabel("Revenue Share")
        plt.show()

    def plot_segment_product_mix(self, df_filtered, rfm, product_sales, grade_names):
        if df_filtered.empty or rfm.empty or product_sales.empty:
            return pd.DataFrame()

        abc_map = product_sales[["StockCode", "ABC_Class"]].copy()
        grade_map = rfm[["CustomerID", "Customer_Grade"]].copy()

        temp = df_filtered.merge(abc_map, on="StockCode", how="left")
        temp = temp.merge(grade_map, on="CustomerID", how="left")
        temp = temp.dropna(subset=["ABC_Class", "Customer_Grade"])

        mix = temp.groupby(["Customer_Grade", "ABC_Class"])["Revenue"].sum().reset_index()
        pivot = mix.pivot(index="Customer_Grade", columns="ABC_Class", values="Revenue").fillna(0)

        ordered_rows = [g for g in grade_names if g in pivot.index]
        pivot = pivot.reindex(ordered_rows)

        share = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)
        share = share.reindex(columns=["A", "B", "C"])

        share.plot(kind="bar", stacked=True, figsize=(10, 6))
        plt.title("세그먼트별 상품 비중(ABC 매출 비중)")
        plt.xlabel("Customer Grade")
        plt.ylabel("Revenue Share within Segment")
        plt.xticks(rotation=0)
        plt.legend(title="ABC Class")
        plt.show()

        print("\n[세그먼트별 상품 비중]")
        print(share)

        return share


# =========================================================
# 7. rolling 월별 RFM + 등급 생성 클래스
# =========================================================
class RollingMonthlyAnalyzer:
    def __init__(self, config):
        self.config = config
        self.rfm_analyzer = RFMAnalyzer()

    def _month_range(self):
        return pd.date_range(
            start=self.config.ANALYSIS_MONTH_START,
            end=self.config.ANALYSIS_MONTH_END,
            freq="MS"
        )

    def build_monthly_panel(self, df):
        months = self._month_range()
        all_rows = []

        all_customers = sorted(df["CustomerID"].unique())

        for month_start in months:
            window_start = month_start - pd.DateOffset(months=self.config.ROLLING_MONTHS)
            window_end = month_start

            window_df = df[
                (df["InvoiceDate"] >= window_start) &
                (df["InvoiceDate"] < window_end)
            ].copy()

            rfm = self.rfm_analyzer.create_rfm_full_customer_base(
                df=window_df,
                snapshot_date=month_start,
                all_customers=all_customers,
                rolling_months=self.config.ROLLING_MONTHS
            )

            if len(rfm) < self.config.FINAL_K:
                continue

            segmenter = CustomerSegmenter(
                self.config.K_RANGE,
                self.config.FINAL_K,
                self.config.RANDOM_STATE,
                self.config.N_INIT
            )
            rfm = segmenter.fit_clusters(rfm, self.config.GRADE_NAMES)

            rfm["AnalysisMonth"] = month_start
            rfm["WindowStart"] = window_start
            rfm["WindowEnd"] = window_end

            current_month_df = df[
                (df["InvoiceDate"] >= month_start) &
                (df["InvoiceDate"] < month_start + pd.DateOffset(months=1))
            ].copy()

            month_perf = current_month_df.groupby("CustomerID").agg(
                CurrentMonthRevenue=("Revenue", "sum"),
                CurrentMonthOrders=("InvoiceNo", "nunique")
            ).reset_index()

            rfm = rfm.merge(month_perf, on="CustomerID", how="left")
            rfm["CurrentMonthRevenue"] = rfm["CurrentMonthRevenue"].fillna(0)
            rfm["CurrentMonthOrders"] = rfm["CurrentMonthOrders"].fillna(0)

            rfm["IsActiveThisMonth"] = np.where(rfm["CurrentMonthOrders"] > 0, 1, 0)

            all_rows.append(rfm)

        if not all_rows:
            return pd.DataFrame()

        panel = pd.concat(all_rows, ignore_index=True)
        panel["AnalysisMonthStr"] = panel["AnalysisMonth"].dt.strftime("%Y-%m")

        return panel

    def plot_monthly_grade_counts(self, monthly_panel):
        if monthly_panel.empty:
            return

        counts = monthly_panel.groupby(["AnalysisMonthStr", "Customer_Grade"])["CustomerID"].nunique().reset_index()
        pivot = counts.pivot(index="AnalysisMonthStr", columns="Customer_Grade", values="CustomerID").fillna(0)

        ordered_cols = [g for g in self.config.GRADE_NAMES if g in pivot.columns]
        pivot = pivot[ordered_cols]

        pivot.plot(kind="bar", stacked=True, figsize=(11, 6))
        plt.title("월별 고객등급 구성")
        plt.xlabel("Month")
        plt.ylabel("Number of Customers")
        plt.xticks(rotation=0)
        plt.show()


# =========================================================
# 8. Lv3 Down / Stay / Up 별 ABC 비중 비교 함수
# =========================================================
def plot_abc_by_movement(monthly_panel, df_filtered, product_sales):
    # 1. Movement 계산
    panel = monthly_panel.copy()
    panel = panel.sort_values(["CustomerID", "AnalysisMonth"])

    shifted = panel[["CustomerID", "AnalysisMonth", "Customer_Grade"]].copy()
    shifted["AnalysisMonth"] = shifted["AnalysisMonth"] - pd.DateOffset(months=1)
    shifted = shifted.rename(columns={"Customer_Grade": "NextGrade"})

    merged = panel.merge(
        shifted,
        on=["CustomerID", "AnalysisMonth"],
        how="left"
    )

    merged["NextGrade"] = merged["NextGrade"].fillna("NoActivity")

    grade_num_map = {
        "Lv1.이탈관리": 1,
        "Lv2.관심필요": 2,
        "Lv3.신규성장": 3,
        "Lv4.우수충성": 4,
        "Lv5.최우수VIP": 5
    }

    merged["GradeNum"] = merged["Customer_Grade"].map(grade_num_map)
    merged["NextGradeNum"] = merged["NextGrade"].map(grade_num_map)

    def movement_type(row):
        if row["NextGrade"] == "NoActivity":
            return "NoActivity"
        if pd.isna(row["NextGradeNum"]):
            return "Unknown"
        if row["NextGradeNum"] > row["GradeNum"]:
            return "Up"
        elif row["NextGradeNum"] < row["GradeNum"]:
            return "Down"
        else:
            return "Stay"

    merged["MovementType"] = merged.apply(movement_type, axis=1)

    # 2. Lv3만 필터
    lv3 = merged[merged["Customer_Grade"] == "Lv3.신규성장"].copy()

    # 3. 상품 ABC 붙이기
    abc_map = product_sales[["StockCode", "ABC_Class"]].copy()

    temp = df_filtered.merge(
        lv3[["CustomerID", "AnalysisMonth", "MovementType"]],
        on="CustomerID",
        how="inner"
    )

    temp = temp.merge(abc_map, on="StockCode", how="left")
    temp = temp.dropna(subset=["ABC_Class", "MovementType"])

    # 4. 비중 계산
    mix = temp.groupby(["MovementType", "ABC_Class"])["Revenue"].sum().reset_index()
    pivot = mix.pivot(index="MovementType", columns="ABC_Class", values="Revenue").fillna(0)
    share = pivot.div(pivot.sum(axis=1), axis=0)

    order = ["Up", "Stay", "Down"]
    share = share.reindex(order)
    share = share[["A", "B", "C"]]

    print("\n[Lv3 Down / Stay / Up 별 ABC 비중]")
    print(share)

    # 5. 그래프
    share.plot(kind="bar", stacked=True, figsize=(8, 5))
    plt.title("Lv3 고객 이동별 상품 구성 (ABC 비중)")
    plt.xlabel("Movement Type")
    plt.ylabel("Revenue Share")
    plt.xticks(rotation=0)
    plt.legend(title="ABC Class")
    plt.show()

    return share


# =========================================================
# 9. Lv3 이동성 분석 클래스
# =========================================================
class TransitionAnalyzer:
    def __init__(self, config):
        self.config = config

    def build_transition_table(self, monthly_panel, horizon=1):
        if monthly_panel.empty:
            return pd.DataFrame(), pd.DataFrame()

        panel = monthly_panel[
            ["CustomerID", "AnalysisMonth", "Customer_Grade", "IsActiveThisMonth"]
        ].copy()
        panel = panel.sort_values(["CustomerID", "AnalysisMonth"])

        shifted = panel.copy()
        shifted["AnalysisMonth"] = shifted["AnalysisMonth"] - pd.DateOffset(months=horizon)
        shifted = shifted.rename(columns={
            "Customer_Grade": f"FutureGrade_{horizon}M",
            "IsActiveThisMonth": f"FutureActive_{horizon}M"
        })

        merged = panel.merge(
            shifted,
            on=["CustomerID", "AnalysisMonth"],
            how="left"
        )

        lv3_base = merged[merged["Customer_Grade"] == self.config.TARGET_GRADE].copy()

        future_grade_col = f"FutureGrade_{horizon}M"
        future_active_col = f"FutureActive_{horizon}M"

        lv3_base[future_active_col] = lv3_base[future_active_col].fillna(0)
        lv3_base[future_grade_col] = lv3_base[future_grade_col].fillna("NoActivity")

        transition_count = pd.crosstab(
            lv3_base["Customer_Grade"],
            lv3_base[future_grade_col],
            dropna=False
        )

        transition_ratio = transition_count.div(transition_count.sum(axis=1), axis=0)

        return transition_count, transition_ratio

    def build_transition_by_month(self, monthly_panel, horizon=1):
        if monthly_panel.empty:
            return pd.DataFrame()

        panel = monthly_panel[
            ["CustomerID", "AnalysisMonth", "AnalysisMonthStr", "Customer_Grade", "IsActiveThisMonth"]
        ].copy()
        panel = panel.sort_values(["CustomerID", "AnalysisMonth"])

        shifted = panel[["CustomerID", "AnalysisMonth", "Customer_Grade", "IsActiveThisMonth"]].copy()
        shifted["AnalysisMonth"] = shifted["AnalysisMonth"] - pd.DateOffset(months=horizon)
        shifted = shifted.rename(columns={
            "Customer_Grade": f"FutureGrade_{horizon}M",
            "IsActiveThisMonth": f"FutureActive_{horizon}M"
        })

        merged = panel.merge(
            shifted,
            on=["CustomerID", "AnalysisMonth"],
            how="left"
        )

        lv3_base = merged[merged["Customer_Grade"] == self.config.TARGET_GRADE].copy()

        future_grade_col = f"FutureGrade_{horizon}M"
        future_active_col = f"FutureActive_{horizon}M"

        lv3_base[future_active_col] = lv3_base[future_active_col].fillna(0)
        lv3_base[future_grade_col] = lv3_base[future_grade_col].fillna("NoActivity")

        by_month = lv3_base.groupby(
            ["AnalysisMonthStr", future_grade_col]
        )["CustomerID"].nunique().reset_index(name="Customers")

        total_month = by_month.groupby("AnalysisMonthStr")["Customers"].sum().reset_index(name="TotalCustomers")
        by_month = by_month.merge(total_month, on="AnalysisMonthStr", how="left")
        by_month["Ratio"] = by_month["Customers"] / by_month["TotalCustomers"]

        return by_month

    def plot_transition_ratio(self, transition_ratio, title):
        if transition_ratio.empty:
            return

        transition_ratio.T.plot(kind="bar", figsize=(9, 5))
        plt.title(title)
        plt.xlabel("Future Grade")
        plt.ylabel("Ratio")
        plt.xticks(rotation=0)
        plt.show()


# =========================================================
# 9-2. 추가 해석 보강 클래스
# =========================================================
class InsightBoosterAnalyzer:
    def __init__(self, config):
        self.config = config

    def compare_grade_mobility(self, monthly_panel):
        panel = monthly_panel[["CustomerID", "AnalysisMonth", "Customer_Grade"]].copy()
        panel = panel.sort_values(["CustomerID", "AnalysisMonth"])

        shifted = panel.copy()
        shifted["AnalysisMonth"] = shifted["AnalysisMonth"] - pd.DateOffset(months=1)
        shifted = shifted.rename(columns={"Customer_Grade": "NextGrade"})

        merged = panel.merge(
            shifted,
            on=["CustomerID", "AnalysisMonth"],
            how="left"
        )
        merged["NextGrade"] = merged["NextGrade"].fillna("NoActivity")

        merged["Stayed"] = (merged["Customer_Grade"] == merged["NextGrade"]).astype(int)
        merged["Moved"] = 1 - merged["Stayed"]

        grade_num_map = {
            "Lv1.이탈관리": 1,
            "Lv2.관심필요": 2,
            "Lv3.신규성장": 3,
            "Lv4.우수충성": 4,
            "Lv5.최우수VIP": 5
        }
        merged["GradeNum"] = merged["Customer_Grade"].map(grade_num_map)
        merged["NextGradeNum"] = merged["NextGrade"].map(grade_num_map)

        def movement_type(row):
            if row["NextGrade"] == "NoActivity":
                return "NoActivity"
            if pd.isna(row["GradeNum"]) or pd.isna(row["NextGradeNum"]):
                return "Unknown"
            if row["NextGradeNum"] > row["GradeNum"]:
                return "Up"
            elif row["NextGradeNum"] < row["GradeNum"]:
                return "Down"
            else:
                return "Stay"

        merged["MovementType"] = merged.apply(movement_type, axis=1)

        mobility_summary = merged.groupby("Customer_Grade").agg(
            Customers=("CustomerID", "nunique"),
            Obs=("CustomerID", "size"),
            Move_Rate=("Moved", "mean")
        ).reset_index()

        movement_dist = merged.groupby(["Customer_Grade", "MovementType"]).size().reset_index(name="Count")
        movement_total = movement_dist.groupby("Customer_Grade")["Count"].sum().reset_index(name="Total")
        movement_dist = movement_dist.merge(movement_total, on="Customer_Grade", how="left")
        movement_dist["Ratio"] = movement_dist["Count"] / movement_dist["Total"]

        print("\n[등급별 이동성 비교]")
        print(mobility_summary)

        print("\n[등급별 이동 방향 분포]")
        print(movement_dist)

        return mobility_summary, movement_dist, merged

    def plot_mobility_compare(self, mobility_summary):
        if mobility_summary.empty:
            return

        plt.figure(figsize=(8, 5))
        plt.bar(mobility_summary["Customer_Grade"], mobility_summary["Move_Rate"])
        plt.title("등급별 다음달 이동률 비교")
        plt.xlabel("Customer Grade")
        plt.ylabel("Move Rate")
        plt.xticks(rotation=0, fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()


# =========================================================
# 10. EDA 클래스
# =========================================================
class EDAAnalyzer:
    def run_basic_eda(self, df, rfm):
        print("\n[매출 분포]")
        print(df["Revenue"].describe())

        print("\n[수량/가격 분포]")
        print(df[["Quantity", "UnitPrice"]].describe())

        monthly_sales = df.groupby("YearMonth")["Revenue"].sum()

        plt.figure(figsize=(10, 5))
        monthly_sales.plot(marker="o")
        plt.title("월별 총매출 추이")
        plt.xlabel("Year-Month")
        plt.ylabel("Revenue")
        plt.xticks(rotation=0)
        plt.show()

        print("\n[RFM 분포]")
        print(rfm.describe())

        print("\n[변수 관계]")
        print(df[["Quantity", "UnitPrice", "Revenue"]].corr())


# =========================================================
# 10-1. Lv3 재구매 고객 분석 클래스
# =========================================================
class RepeatCustomerAnalyzer:
    def __init__(self, config):
        self.config = config

    def classify_repeat_status(self, monthly_panel):
        df = monthly_panel.copy()
        df = df[df["Customer_Grade"] == self.config.TARGET_GRADE].copy()

        df["RepeatStatus"] = np.where(df["Frequency"] >= 2, "Repeat", "OneTime")

        summary = df.groupby("RepeatStatus").agg(
            Customers=("CustomerID", "nunique"),
            Obs=("CustomerID", "size"),
            Mean_Recency=("Recency", "mean"),
            Mean_Frequency=("Frequency", "mean"),
            Mean_Monetary=("Monetary", "mean"),
            Mean_CurrentMonthRevenue=("CurrentMonthRevenue", "mean"),
            Mean_CurrentMonthOrders=("CurrentMonthOrders", "mean")
        ).reset_index()

        print("\n[Lv3 재구매 고객 구분 요약]")
        print(summary)

        return df, summary

    def compare_next_month_transition(self, monthly_panel):
        panel = monthly_panel.copy()

        target_panel = panel[panel["Customer_Grade"] == self.config.TARGET_GRADE].copy()
        target_panel["RepeatStatus"] = np.where(
            target_panel["Frequency"] >= 2, "Repeat", "OneTime"
        )

        shifted = panel[[
            "CustomerID", "AnalysisMonth", "Customer_Grade", "IsActiveThisMonth"
        ]].copy()
        shifted["AnalysisMonth"] = shifted["AnalysisMonth"] - pd.DateOffset(months=1)
        shifted = shifted.rename(columns={
            "Customer_Grade": "NextGrade",
            "IsActiveThisMonth": "NextActive"
        })

        merged = target_panel.merge(
            shifted,
            on=["CustomerID", "AnalysisMonth"],
            how="left"
        )

        merged["NextActive"] = merged["NextActive"].fillna(0)
        merged["NextGrade"] = merged["NextGrade"].fillna("NoActivity")

        grade_num_map = {
            "Lv1.이탈관리": 1,
            "Lv2.관심필요": 2,
            "Lv3.신규성장": 3,
            "Lv4.우수충성": 4,
            "Lv5.최우수VIP": 5
        }

        merged["GradeNum"] = 3
        merged["NextGradeNum"] = merged["NextGrade"].map(grade_num_map)

        def movement_type(row):
            if pd.isna(row["NextGradeNum"]):
                return "Unknown"
            if row["NextGradeNum"] > row["GradeNum"]:
                return "Up"
            elif row["NextGradeNum"] < row["GradeNum"]:
                return "Down"
            else:
                return "Stay"

        merged["MovementType"] = merged.apply(movement_type, axis=1)

        transition_summary = merged.groupby("RepeatStatus").agg(
            Obs=("CustomerID", "size"),
            Up_Rate=("MovementType", lambda x: (x == "Up").mean()),
            Stay_Rate=("MovementType", lambda x: (x == "Stay").mean()),
            Down_Rate=("MovementType", lambda x: (x == "Down").mean()),
            Active_Rate=("NextActive", lambda x: (x == 1).mean()),
            Inactive_Rate=("NextActive", lambda x: (x == 0).mean())
        ).reset_index()

        transition_dist = merged.groupby(
            ["RepeatStatus", "MovementType"]
        ).size().reset_index(name="Count")
        total = transition_dist.groupby("RepeatStatus")["Count"].sum().reset_index(name="Total")
        transition_dist = transition_dist.merge(total, on="RepeatStatus", how="left")
        transition_dist["Ratio"] = transition_dist["Count"] / transition_dist["Total"]

        print("\n[Lv3 재구매 고객 vs 일회성 고객의 다음달 이동 비교]")
        print(transition_summary)

        print("\n[Lv3 재구매 고객 vs 일회성 고객의 이동 방향 분포]")
        print(transition_dist)

        return merged, transition_summary, transition_dist

    def plot_repeat_transition(self, transition_summary):
        if transition_summary.empty:
            return

        plot_df = transition_summary.set_index("RepeatStatus")[[
            "Up_Rate", "Stay_Rate", "Down_Rate"
        ]]

        plot_df.plot(kind="bar", figsize=(9, 5))
        plt.title("Lv3 재구매 고객 vs 일회성 고객의 다음달 그룹 이동 비교")
        plt.ylabel("Rate")
        plt.xticks(rotation=0)
        plt.show()


# =========================================================
# 10-2. Lv3 내부 R/F/M 영향 비교 클래스
# =========================================================
class RFMImpactAnalyzer:
    def __init__(self, config):
        self.config = config

    def _prepare_lv3_next_month(self, monthly_panel):
        if monthly_panel.empty:
            return pd.DataFrame()

        panel = monthly_panel.copy()
        panel = panel.sort_values(["CustomerID", "AnalysisMonth"])

        shifted = panel[[
            "CustomerID", "AnalysisMonth", "Customer_Grade"
        ]].copy()
        shifted["AnalysisMonth"] = shifted["AnalysisMonth"] - pd.DateOffset(months=1)
        shifted = shifted.rename(columns={"Customer_Grade": "NextGrade"})

        merged = panel.merge(
            shifted,
            on=["CustomerID", "AnalysisMonth"],
            how="left"
        )

        lv3 = merged[merged["Customer_Grade"] == self.config.TARGET_GRADE].copy()
        lv3["NextGrade"] = lv3["NextGrade"].fillna("NoActivity")

        grade_num_map = {
            "Lv1.이탈관리": 1,
            "Lv2.관심필요": 2,
            "Lv3.신규성장": 3,
            "Lv4.우수충성": 4,
            "Lv5.최우수VIP": 5
        }

        lv3["CurrentGradeNum"] = 3
        lv3["NextGradeNum"] = lv3["NextGrade"].map(grade_num_map)

        def movement_type(row):
            if row["NextGrade"] == "NoActivity":
                return "NoActivity"
            if pd.isna(row["NextGradeNum"]):
                return "Unknown"
            if row["NextGradeNum"] > row["CurrentGradeNum"]:
                return "Up"
            elif row["NextGradeNum"] < row["CurrentGradeNum"]:
                return "Down"
            else:
                return "Stay"

        lv3["MovementType"] = lv3.apply(movement_type, axis=1)
        return lv3

    def _make_binary_group(self, df, var_name, low_label, high_label):
        cut = df[var_name].median()
        out = df.copy()
        out[f"{var_name}_Group"] = np.where(
            out[var_name] <= cut,
            low_label,
            high_label
        )
        return out, cut

    def analyze_variable(self, monthly_panel, var_name):
        df = self._prepare_lv3_next_month(monthly_panel)
        if df.empty:
            return pd.DataFrame(), pd.DataFrame(), None

        if var_name == "Recency":
            df, cut = self._make_binary_group(
                df, "Recency",
                "LowRecency(최근구매)",
                "HighRecency(오래전구매)"
            )
        elif var_name == "Frequency":
            df, cut = self._make_binary_group(
                df, "Frequency",
                "LowFrequency(저빈도)",
                "HighFrequency(고빈도)"
            )
        elif var_name == "Monetary":
            df, cut = self._make_binary_group(
                df, "Monetary",
                "LowMonetary(저매출)",
                "HighMonetary(고매출)"
            )
        else:
            raise ValueError("var_name must be one of ['Recency', 'Frequency', 'Monetary']")

        group_col = f"{var_name}_Group"

        summary = df.groupby(group_col).agg(
            Obs=("CustomerID", "size"),
            Customers=("CustomerID", "nunique"),
            Mean_Value=(var_name, "mean"),
            Up_Rate=("MovementType", lambda x: (x == "Up").mean()),
            Stay_Rate=("MovementType", lambda x: (x == "Stay").mean()),
            Down_Rate=("MovementType", lambda x: (x == "Down").mean()),
            NoActivity_Rate=("MovementType", lambda x: (x == "NoActivity").mean())
        ).reset_index()

        dist = df.groupby([group_col, "MovementType"]).size().reset_index(name="Count")
        total = dist.groupby(group_col)["Count"].sum().reset_index(name="Total")
        dist = dist.merge(total, on=group_col, how="left")
        dist["Ratio"] = dist["Count"] / dist["Total"]

        print(f"\n[Lv3 내부 {var_name} 그룹별 다음달 이동 비교]")
        print(f"{var_name} median cut = {cut}")
        print(summary)

        print(f"\n[Lv3 내부 {var_name} 그룹별 이동 방향 분포]")
        print(dist)

        return summary, dist, cut

    def compare_rfm_impact(self, monthly_panel):
        results = {}

        for var in ["Recency", "Frequency", "Monetary"]:
            summary, dist, cut = self.analyze_variable(monthly_panel, var)
            results[var] = {
                "summary": summary,
                "dist": dist,
                "cut": cut
            }

        impact_rows = []

        for var, obj in results.items():
            summary = obj["summary"]
            if summary.empty or len(summary) < 2:
                continue

            summary_sorted = summary.sort_values("Mean_Value").reset_index(drop=True)
            low = summary_sorted.iloc[0]
            high = summary_sorted.iloc[1]

            impact_rows.append({
                "Variable": var,
                "Cut": obj["cut"],
                "LowGroup": low.iloc[0],
                "HighGroup": high.iloc[0],
                "Up_Rate_Gap": abs(high["Up_Rate"] - low["Up_Rate"]),
                "Stay_Rate_Gap": abs(high["Stay_Rate"] - low["Stay_Rate"]),
                "Down_Rate_Gap": abs(high["Down_Rate"] - low["Down_Rate"]),
                "NoActivity_Rate_Gap": abs(high["NoActivity_Rate"] - low["NoActivity_Rate"]),
                "Total_Gap": (
                    abs(high["Up_Rate"] - low["Up_Rate"]) +
                    abs(high["Stay_Rate"] - low["Stay_Rate"]) +
                    abs(high["Down_Rate"] - low["Down_Rate"]) +
                    abs(high["NoActivity_Rate"] - low["NoActivity_Rate"])
                )
            })

        impact_summary = pd.DataFrame(impact_rows).sort_values("Total_Gap", ascending=False)

        print("\n[R/F/M 중 어떤 변수가 다음달 이동 차이를 가장 크게 만드는가?]")
        print(impact_summary)

        return results, impact_summary

    def plot_variable_impact(self, summary, var_name):
        if summary.empty:
            return

        group_col = summary.columns[0]
        plot_df = summary.set_index(group_col)[[
            "Up_Rate", "Down_Rate"
        ]]

        if var_name == "Recency":
            recency_order = ["LowRecency(최근구매)", "HighRecency(오래전구매)"]
            existing_order = [x for x in recency_order if x in plot_df.index]
            plot_df = plot_df.reindex(existing_order)

        plot_df.plot(kind="bar", figsize=(9, 5))
        plt.title(f"Lv3 내부 {var_name} 그룹별 다음달 그룹 이동 비교", fontsize=16)
        plt.xlabel(var_name, fontsize=14)
        plt.ylabel("Rate", fontsize=14)
        plt.xticks(rotation=0, fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.show()

    def plot_rfm_gap_summary(self, impact_summary):
        if impact_summary.empty:
            return

        plt.figure(figsize=(8, 5))
        plt.bar(impact_summary["Variable"], impact_summary["Total_Gap"])
        plt.title("R/F/M 변수별 다음달 이동 차이 크기", fontsize=16)
        plt.xlabel("Variable", fontsize=14)
        plt.ylabel("Total Gap", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()

    def plot_rfm_up_down_gap(self, impact_summary):
        if impact_summary.empty:
            return

        plot_df = impact_summary.copy().sort_values("Total_Gap", ascending=False)

        x = np.arange(len(plot_df))
        width = 0.35

        plt.figure(figsize=(8, 5))
        plt.bar(x - width/2, plot_df["Up_Rate_Gap"], width, label="Up Gap")
        plt.bar(x + width/2, plot_df["Down_Rate_Gap"], width, label="Down Gap")

        plt.title("R/F/M 변수별 다음달 상승/하락 차이", fontsize=16)
        plt.xlabel("Variable", fontsize=14)
        plt.ylabel("Gap", fontsize=14)
        plt.xticks(x, plot_df["Variable"], fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.show()


# =========================================================
# 10-3. High M 내부에서 F 차이 분석 클래스
# =========================================================
class HighMFrequencyAnalyzer:
    def __init__(self, config):
        self.config = config

    def analyze_high_m_frequency_effect(self, monthly_panel):
        if monthly_panel.empty:
            return pd.DataFrame(), pd.DataFrame(), None, None, pd.DataFrame()

        panel = monthly_panel.copy()
        panel = panel.sort_values(["CustomerID", "AnalysisMonth"])

        shifted = panel[["CustomerID", "AnalysisMonth", "Customer_Grade"]].copy()
        shifted["AnalysisMonth"] = shifted["AnalysisMonth"] - pd.DateOffset(months=1)
        shifted = shifted.rename(columns={"Customer_Grade": "NextGrade"})

        merged = panel.merge(
            shifted,
            on=["CustomerID", "AnalysisMonth"],
            how="left"
        )

        lv3 = merged[merged["Customer_Grade"] == self.config.TARGET_GRADE].copy()
        lv3["NextGrade"] = lv3["NextGrade"].fillna("NoActivity")

        grade_num_map = {
            "Lv1.이탈관리": 1,
            "Lv2.관심필요": 2,
            "Lv3.신규성장": 3,
            "Lv4.우수충성": 4,
            "Lv5.최우수VIP": 5
        }

        lv3["CurrentGradeNum"] = 3
        lv3["NextGradeNum"] = lv3["NextGrade"].map(grade_num_map)

        def movement_type(row):
            if row["NextGrade"] == "NoActivity":
                return "NoActivity"
            if pd.isna(row["NextGradeNum"]):
                return "Unknown"
            if row["NextGradeNum"] > row["CurrentGradeNum"]:
                return "Up"
            elif row["NextGradeNum"] < row["CurrentGradeNum"]:
                return "Down"
            else:
                return "Stay"

        lv3["MovementType"] = lv3.apply(movement_type, axis=1)

        monetary_cut = lv3["Monetary"].median()
        high_m = lv3[lv3["Monetary"] >= monetary_cut].copy()

        if high_m.empty:
            return pd.DataFrame(), pd.DataFrame(), monetary_cut, None, high_m

        freq_cut = high_m["Frequency"].median()

        high_m["FreqGroup_in_HighM"] = np.where(
            high_m["Frequency"] >= freq_cut,
            "HighF_in_HighM",
            "LowF_in_HighM"
        )

        summary = high_m.groupby("FreqGroup_in_HighM").agg(
            Obs=("CustomerID", "size"),
            Customers=("CustomerID", "nunique"),
            Mean_Frequency=("Frequency", "mean"),
            Mean_Monetary=("Monetary", "mean"),
            Up_Rate=("MovementType", lambda x: (x == "Up").mean()),
            Stay_Rate=("MovementType", lambda x: (x == "Stay").mean()),
            Down_Rate=("MovementType", lambda x: (x == "Down").mean()),
            NoActivity_Rate=("MovementType", lambda x: (x == "NoActivity").mean())
        ).reset_index()

        dist = high_m.groupby(["FreqGroup_in_HighM", "MovementType"]).size().reset_index(name="Count")
        total = dist.groupby("FreqGroup_in_HighM")["Count"].sum().reset_index(name="Total")
        dist = dist.merge(total, on="FreqGroup_in_HighM", how="left")
        dist["Ratio"] = dist["Count"] / dist["Total"]

        print("\n[Lv3 내부 High M 고객에서 Frequency 차이에 따른 다음달 이동 비교]")
        print(f"Lv3 Monetary median cut = {monetary_cut}")
        print(f"High M 내부 Frequency median cut = {freq_cut}")
        print(summary)

        print("\n[Lv3 내부 High M 고객에서 Frequency 차이에 따른 이동 방향 분포]")
        print(dist)

        return summary, dist, monetary_cut, freq_cut, high_m

    def plot_high_m_frequency_up_down(self, summary):
        if summary.empty:
            return

        plot_df = summary.set_index("FreqGroup_in_HighM")[[
            "Up_Rate", "Down_Rate"
        ]]

        x = np.arange(len(plot_df))
        width = 0.35

        plt.figure(figsize=(8, 5))
        plt.bar(x - width/2, plot_df["Up_Rate"], width, label="Up (상승)")
        plt.bar(x + width/2, plot_df["Down_Rate"], width, label="Down (하락)")

        plt.title("같은 소비 수준에서 반복 여부에 따른 그룹 이동 비교", fontsize=16)
        plt.xlabel("Frequency Group", fontsize=14)
        plt.ylabel("Rate", fontsize=14)
        plt.xticks(x, plot_df.index, fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.show()


# =========================================================
# 10-3-2. High F 내부에서 M 차이 분석 클래스
# =========================================================
class HighFrequencyMonetaryAnalyzer:
    def __init__(self, config):
        self.config = config

    def analyze_high_f_monetary_effect(self, monthly_panel):
        if monthly_panel.empty:
            return pd.DataFrame(), pd.DataFrame(), None, None, pd.DataFrame()

        panel = monthly_panel.copy()
        panel = panel.sort_values(["CustomerID", "AnalysisMonth"])

        shifted = panel[["CustomerID", "AnalysisMonth", "Customer_Grade"]].copy()
        shifted["AnalysisMonth"] = shifted["AnalysisMonth"] - pd.DateOffset(months=1)
        shifted = shifted.rename(columns={"Customer_Grade": "NextGrade"})

        merged = panel.merge(
            shifted,
            on=["CustomerID", "AnalysisMonth"],
            how="left"
        )

        lv3 = merged[merged["Customer_Grade"] == self.config.TARGET_GRADE].copy()
        lv3["NextGrade"] = lv3["NextGrade"].fillna("NoActivity")

        grade_num_map = {
            "Lv1.이탈관리": 1,
            "Lv2.관심필요": 2,
            "Lv3.신규성장": 3,
            "Lv4.우수충성": 4,
            "Lv5.최우수VIP": 5
        }

        lv3["CurrentGradeNum"] = 3
        lv3["NextGradeNum"] = lv3["NextGrade"].map(grade_num_map)

        def movement_type(row):
            if row["NextGrade"] == "NoActivity":
                return "NoActivity"
            if pd.isna(row["NextGradeNum"]):
                return "Unknown"
            if row["NextGradeNum"] > row["CurrentGradeNum"]:
                return "Up"
            elif row["NextGradeNum"] < row["CurrentGradeNum"]:
                return "Down"
            else:
                return "Stay"

        lv3["MovementType"] = lv3.apply(movement_type, axis=1)

        freq_cut = lv3["Frequency"].median()
        high_f = lv3[lv3["Frequency"] >= freq_cut].copy()

        if high_f.empty:
            return pd.DataFrame(), pd.DataFrame(), freq_cut, None, high_f

        monetary_cut = high_f["Monetary"].median()

        high_f["MonetaryGroup_in_HighF"] = np.where(
            high_f["Monetary"] >= monetary_cut,
            "HighM_in_HighF",
            "LowM_in_HighF"
        )

        summary = high_f.groupby("MonetaryGroup_in_HighF").agg(
            Obs=("CustomerID", "size"),
            Customers=("CustomerID", "nunique"),
            Mean_Frequency=("Frequency", "mean"),
            Mean_Monetary=("Monetary", "mean"),
            Up_Rate=("MovementType", lambda x: (x == "Up").mean()),
            Stay_Rate=("MovementType", lambda x: (x == "Stay").mean()),
            Down_Rate=("MovementType", lambda x: (x == "Down").mean()),
            NoActivity_Rate=("MovementType", lambda x: (x == "NoActivity").mean())
        ).reset_index()

        dist = high_f.groupby(["MonetaryGroup_in_HighF", "MovementType"]).size().reset_index(name="Count")
        total = dist.groupby("MonetaryGroup_in_HighF")["Count"].sum().reset_index(name="Total")
        dist = dist.merge(total, on="MonetaryGroup_in_HighF", how="left")
        dist["Ratio"] = dist["Count"] / dist["Total"]

        print("\n[Lv3 내부 High F 고객에서 Monetary 차이에 따른 다음달 그룹 이동 비교]")
        print(f"Lv3 Frequency median cut = {freq_cut}")
        print(f"High F 내부 Monetary median cut = {monetary_cut}")
        print(summary)

        print("\n[Lv3 내부 High F 고객에서 Monetary 차이에 따른 이동 방향 분포]")
        print(dist)

        return summary, dist, freq_cut, monetary_cut, high_f

    def plot_high_f_monetary_up_down(self, summary):
        if summary.empty:
            return

        plot_df = summary.set_index("MonetaryGroup_in_HighF")[[
            "Up_Rate", "Down_Rate"
        ]]

        x = np.arange(len(plot_df))
        width = 0.35

        plt.figure(figsize=(8, 5))
        plt.bar(x - width/2, plot_df["Up_Rate"], width, label="Up (상승)")
        plt.bar(x + width/2, plot_df["Down_Rate"], width, label="Down (하락)")

        plt.title("같은 반복 수준에서 소비 수준에 따른 그룹 이동 비교", fontsize=16)
        plt.xlabel("Monetary Group", fontsize=14)
        plt.ylabel("Rate", fontsize=14)
        plt.xticks(x, plot_df.index, fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.show()

# =========================================================
# 10-4. 6개월 롤링 ABC 구성비 분석 클래스
# =========================================================
class RollingABCWindowAnalyzer:
    def __init__(self, config):
        self.config = config

    def _build_window_abc_map(self, df_filtered, monthly_panel):
        """
        각 AnalysisMonth별로 같은 6개월 window를 사용하여
        상품 ABC 맵(StockCode -> ABC_Class)을 만든다.
        """
        abc_maps = {}

        unique_windows = (
            monthly_panel[["AnalysisMonth", "WindowStart", "WindowEnd"]]
            .drop_duplicates()
            .sort_values("AnalysisMonth")
        )

        for _, row in unique_windows.iterrows():
            analysis_month = row["AnalysisMonth"]
            window_start = row["WindowStart"]
            window_end = row["WindowEnd"]

            window_df = df_filtered[
                (df_filtered["InvoiceDate"] >= window_start) &
                (df_filtered["InvoiceDate"] < window_end)
            ].copy()

            if window_df.empty:
                abc_maps[analysis_month] = pd.DataFrame(
                    columns=["StockCode", "ABC_Class"]
                )
                continue

            product_sales = (
                window_df.groupby("StockCode", as_index=False)["Revenue"]
                .sum()
                .sort_values("Revenue", ascending=False)
            )
            product_sales["CumulativeRevenue"] = product_sales["Revenue"].cumsum()
            product_sales["CumulativeRatio"] = (
                product_sales["CumulativeRevenue"] / product_sales["Revenue"].sum()
            )
            product_sales["ABC_Class"] = product_sales["CumulativeRatio"].apply(
                ProductAnalyzer.abc_class
            )

            abc_maps[analysis_month] = product_sales[["StockCode", "ABC_Class"]].copy()

        return abc_maps

    def analyze_lv3_window_abc(self, monthly_panel, df_filtered, include_stay=False):
        """
        Lv3 고객-월 기준으로, 같은 6개월 rolling window의 ABC 구성비를 계산.
        기본값은 Up/Down만 비교하고, Stay는 제외.
        """
        if monthly_panel.empty or df_filtered.empty:
            return pd.DataFrame(), pd.DataFrame()

        panel = monthly_panel.copy()
        panel = panel.sort_values(["CustomerID", "AnalysisMonth"])

        shifted = panel[["CustomerID", "AnalysisMonth", "Customer_Grade"]].copy()
        shifted["AnalysisMonth"] = shifted["AnalysisMonth"] - pd.DateOffset(months=1)
        shifted = shifted.rename(columns={"Customer_Grade": "NextGrade"})

        merged = panel.merge(
            shifted,
            on=["CustomerID", "AnalysisMonth"],
            how="left"
        )

        lv3 = merged[merged["Customer_Grade"] == self.config.TARGET_GRADE].copy()
        lv3["NextGrade"] = lv3["NextGrade"].fillna("NoActivity")

        grade_num_map = {
            "Lv1.이탈관리": 1,
            "Lv2.관심필요": 2,
            "Lv3.신규성장": 3,
            "Lv4.우수충성": 4,
            "Lv5.최우수VIP": 5
        }

        lv3["CurrentGradeNum"] = 3
        lv3["NextGradeNum"] = lv3["NextGrade"].map(grade_num_map)

        def movement_type(row):
            if row["NextGrade"] == "NoActivity":
                return "NoActivity"
            if pd.isna(row["NextGradeNum"]):
                return "Unknown"
            if row["NextGradeNum"] > row["CurrentGradeNum"]:
                return "Up"
            elif row["NextGradeNum"] < row["CurrentGradeNum"]:
                return "Down"
            else:
                return "Stay"

        lv3["MovementType"] = lv3.apply(movement_type, axis=1)

        if include_stay:
            lv3 = lv3[lv3["MovementType"].isin(["Up", "Stay", "Down"])].copy()
        else:
            lv3 = lv3[lv3["MovementType"].isin(["Up", "Down"])].copy()

        if lv3.empty:
            return pd.DataFrame(), pd.DataFrame()

        # 같은 6개월 window 기준 상품 ABC 맵
        abc_maps = self._build_window_abc_map(df_filtered, monthly_panel)

        rows = []

        for _, row in lv3.iterrows():
            customer_id = row["CustomerID"]
            analysis_month = row["AnalysisMonth"]
            movement = row["MovementType"]
            window_start = row["WindowStart"]
            window_end = row["WindowEnd"]

            customer_window_df = df_filtered[
                (df_filtered["CustomerID"] == customer_id) &
                (df_filtered["InvoiceDate"] >= window_start) &
                (df_filtered["InvoiceDate"] < window_end)
            ].copy()

            if customer_window_df.empty:
                continue

            abc_map = abc_maps.get(analysis_month)
            if abc_map is None or abc_map.empty:
                continue

            customer_window_df = customer_window_df.merge(
                abc_map,
                on="StockCode",
                how="left"
            )
            customer_window_df = customer_window_df.dropna(subset=["ABC_Class"])

            total_revenue = customer_window_df["Revenue"].sum()
            if total_revenue <= 0:
                continue

            abc_share = (
                customer_window_df.groupby("ABC_Class")["Revenue"]
                .sum()
                .div(total_revenue)
            )

            rows.append({
                "CustomerID": customer_id,
                "AnalysisMonth": analysis_month,
                "AnalysisMonthStr": row["AnalysisMonthStr"],
                "MovementType": movement,
                "A_Share": abc_share.get("A", 0.0),
                "B_Share": abc_share.get("B", 0.0),
                "C_Share": abc_share.get("C", 0.0),
                "WindowStart": window_start,
                "WindowEnd": window_end
            })

        share_df = pd.DataFrame(rows)

        if share_df.empty:
            return share_df, pd.DataFrame()

        summary = (
            share_df.groupby("MovementType")
            .agg(
                Obs=("CustomerID", "size"),
                Customers=("CustomerID", "nunique"),
                Mean_A_Share=("A_Share", "mean"),
                Mean_B_Share=("B_Share", "mean"),
                Mean_C_Share=("C_Share", "mean")
            )
            .reset_index()
        )

        order = ["Up", "Stay", "Down"] if include_stay else ["Up", "Down"]
        summary["MovementType"] = pd.Categorical(
            summary["MovementType"], categories=order, ordered=True
        )
        summary = summary.sort_values("MovementType").reset_index(drop=True)

        print("\n[Lv3 고객-월 기준 6개월 rolling ABC 구성비 비교]")
        print(summary)

        return share_df, summary

    def plot_lv3_window_abc(self, summary):
        if summary.empty:
            return

        plot_df = summary.set_index("MovementType")[[
            "Mean_A_Share", "Mean_B_Share", "Mean_C_Share"
        ]]
        plot_df.columns = ["A", "B", "C"]

        plot_df.plot(kind="bar", stacked=True, figsize=(8, 5))
        plt.title("Lv3 고객 이동별 6개월 Rolling 상품 구성")
        plt.xlabel("Movement Type")
        plt.ylabel("Mean Revenue Share")
        plt.xticks(rotation=0)
        plt.legend(title="ABC Class")
        plt.show()

# =========================================================
# 11. 전체 실행 클래스
# =========================================================
class RetailAnalysisPipeline:
    def __init__(self, config):
        self.config = config
        self.repeat_analyzer = RepeatCustomerAnalyzer(config)
        self.rfm_impact_analyzer = RFMImpactAnalyzer(config)
        self.high_m_freq_analyzer = HighMFrequencyAnalyzer(config)
        self.high_f_m_analyzer = HighFrequencyMonetaryAnalyzer(config)
        self.rolling_abc_analyzer = RollingABCWindowAnalyzer(config)

        self.loader = RetailDataLoader(config.FILE_PATH, config.SHEETS, config.CACHE_PATH)
        self.preprocessor = RetailPreprocessor(
            config.COUNTRY,
            config.START_DATE,
            config.END_DATE,
            config.NON_PRODUCT_CODES
        )
        self.period_splitter = PeriodSplitter(
            config.PRE_START,
            config.PRE_END,
            config.POST_START,
            config.POST_END
        )
        self.rfm_analyzer = RFMAnalyzer()
        self.segmenter = CustomerSegmenter(
            config.K_RANGE,
            config.FINAL_K,
            config.RANDOM_STATE,
            config.N_INIT
        )
        self.product_analyzer = ProductAnalyzer()
        self.rolling_analyzer = RollingMonthlyAnalyzer(config)
        self.transition_analyzer = TransitionAnalyzer(config)
        self.insight_booster = InsightBoosterAnalyzer(config)
        self.eda_analyzer = EDAAnalyzer()

    def run(self):
        plt.rcParams["font.family"] = "Malgun Gothic"
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["axes.titlesize"] = self.config.LABEL_FONT_SIZE
        plt.rcParams["axes.labelsize"] = self.config.LABEL_FONT_SIZE
        plt.rcParams["xtick.labelsize"] = self.config.LABEL_FONT_SIZE
        plt.rcParams["ytick.labelsize"] = self.config.LABEL_FONT_SIZE
        plt.rcParams["legend.fontsize"] = self.config.LABEL_FONT_SIZE

        df = self.loader.load_data()
        df = self.preprocessor.preprocess(df)

        # =========================================================
        # A. 정적 분석용 데이터: 2011년만
        # =========================================================
        df_static = df[
            (df["InvoiceDate"] >= pd.to_datetime(self.config.STATIC_START_DATE)) &
            (df["InvoiceDate"] < pd.to_datetime(self.config.STATIC_END_DATE))
        ].copy()

        print("\n[정적 분석용 데이터: 2011년]")
        print("shape:", df_static.shape)
        print(
            "date range:",
            df_static["InvoiceDate"].min(),
            "~",
            df_static["InvoiceDate"].max()
        )

        rfm = self.rfm_analyzer.create_rfm(
            df_static,
            snapshot_date=self.config.STATIC_END_DATE
        )
        rfm, df_static_filtered = self.rfm_analyzer.remove_outliers(rfm, df_static)

        # =========================================================
        # B. 동적 분석용 데이터: 2010~2011 전체 이력 유지
        #    단, 정적 분석에서 살아남은 고객만 추적
        # =========================================================
        dynamic_customers = set(df_static_filtered["CustomerID"])
        df_filtered = df[df["CustomerID"].isin(dynamic_customers)].copy()

        print("\n[동적 분석용 데이터: 2010~2011 중 정적분석 대상 고객만 유지]")
        print("shape:", df_filtered.shape)
        print(
            "date range:",
            df_filtered["InvoiceDate"].min(),
            "~",
            df_filtered["InvoiceDate"].max()
        )

        # 전체 클러스터링 평가 + 정적 세분화
        if len(rfm) >= self.config.FINAL_K:
            self.segmenter.evaluate_k(rfm)
            self.segmenter.plot_k_result()

            rfm = self.segmenter.fit_clusters(rfm, self.config.GRADE_NAMES)
            self.segmenter.plot_cluster_result(rfm)

        # 정적 분석용 상품 분석도 2011년 기준
        product_sales = self.product_analyzer.analyze_products(df_static_filtered)
        self.product_analyzer.plot_abc_share(product_sales)

        segment_product_mix = self.product_analyzer.plot_segment_product_mix(
            df_filtered=df_static_filtered,
            rfm=rfm,
            product_sales=product_sales,
            grade_names=self.config.GRADE_NAMES
        )

        # EDA도 2011년 기준
        self.eda_analyzer.run_basic_eda(df_static_filtered, rfm)

        # rolling은 2010~2011 이력 사용
        monthly_panel = self.rolling_analyzer.build_monthly_panel(df_filtered)

        # R/F/M 영향 비교
        rfm_impact_results, rfm_impact_summary = self.rfm_impact_analyzer.compare_rfm_impact(
            monthly_panel
        )

        self.rfm_impact_analyzer.plot_variable_impact(
            rfm_impact_results["Recency"]["summary"], "Recency"
        )
        self.rfm_impact_analyzer.plot_variable_impact(
            rfm_impact_results["Frequency"]["summary"], "Frequency"
        )
        self.rfm_impact_analyzer.plot_variable_impact(
            rfm_impact_results["Monetary"]["summary"], "Monetary"
        )
        self.rfm_impact_analyzer.plot_rfm_gap_summary(rfm_impact_summary)
        self.rfm_impact_analyzer.plot_rfm_up_down_gap(rfm_impact_summary)

        print("\n[Rolling Monthly Panel]")
        print(monthly_panel.head())
        print("shape:", monthly_panel.shape)

        self.rolling_analyzer.plot_monthly_grade_counts(monthly_panel)

        t1_count, t1_ratio = self.transition_analyzer.build_transition_table(monthly_panel, horizon=1)
        print("\n[Lv3 -> 다음달 등급 이동 수]")
        print(t1_count)
        print("\n[Lv3 -> 다음달 등급 이동 비율]")
        print(t1_ratio)

        self.transition_analyzer.plot_transition_ratio(
            t1_ratio,
            "Lv3 고객의 다음달 등급 이동 비율"
        )

        t2_count, t2_ratio = self.transition_analyzer.build_transition_table(monthly_panel, horizon=2)
        print("\n[Lv3 -> 2개월 후 등급 이동 수]")
        print(t2_count)
        print("\n[Lv3 -> 2개월 후 등급 이동 비율]")
        print(t2_ratio)

        self.transition_analyzer.plot_transition_ratio(
            t2_ratio,
            "Lv3 고객의 2개월 후 등급 이동 비율"
        )

        by_month_t1 = self.transition_analyzer.build_transition_by_month(monthly_panel, horizon=1)
        by_month_t2 = self.transition_analyzer.build_transition_by_month(monthly_panel, horizon=2)

        mobility_summary, movement_dist, mobility_base = self.insight_booster.compare_grade_mobility(
            monthly_panel
        )
        self.insight_booster.plot_mobility_compare(mobility_summary)

        print("\n[등급별 이동성 요약 테이블]")
        print(mobility_summary)

        print("\n[등급별 이동 방향 분포 테이블]")
        print(movement_dist)

        print("\n[월별 Lv3 -> 다음달 상세 이동성]")
        print(by_month_t1.head(20))

        print("\n[월별 Lv3 -> 2개월 후 상세 이동성]")
        print(by_month_t2.head(20))

        lv3_repeat_df, lv3_repeat_summary = self.repeat_analyzer.classify_repeat_status(
            monthly_panel
        )

        repeat_transition_base, repeat_transition_summary, repeat_transition_dist = (
            self.repeat_analyzer.compare_next_month_transition(monthly_panel)
        )
        self.repeat_analyzer.plot_repeat_transition(repeat_transition_summary)

        print("\n[Lv3 재구매 고객 구분 요약 테이블]")
        print(lv3_repeat_summary)

        print("\n[Lv3 재구매 고객 다음달 이동 요약 테이블]")
        print(repeat_transition_summary)

        print("\n[Lv3 재구매 고객 다음달 이동 분포 테이블]")
        print(repeat_transition_dist)

        # High M 내부 Frequency 차이 분석
        high_m_freq_summary, high_m_freq_dist, high_m_cut, high_m_freq_cut, high_m_base = (
            self.high_m_freq_analyzer.analyze_high_m_frequency_effect(monthly_panel)
        )
        self.high_m_freq_analyzer.plot_high_m_frequency_up_down(high_m_freq_summary)

        print("\n[High M 내부 Frequency 차이 요약 테이블]")
        print(high_m_freq_summary)

        print("\n[High M 내부 Frequency 차이 분포 테이블]")
        print(high_m_freq_dist)

        # High F 내부 Monetary 차이 분석
        high_f_m_summary, high_f_m_dist, high_f_cut, high_f_m_cut, high_f_base = (
            self.high_f_m_analyzer.analyze_high_f_monetary_effect(monthly_panel)
        )
        self.high_f_m_analyzer.plot_high_f_monetary_up_down(high_f_m_summary)

        print("\n[High F 내부 Monetary 차이 요약 테이블]")
        print(high_f_m_summary)

        print("\n[High F 내부 Monetary 차이 분포 테이블]")
        print(high_f_m_dist)

        # =========================================================
        # 16. Lv3 고객-월 기준 6개월 rolling ABC 구성비 비교
        # =========================================================
        lv3_window_abc_df, lv3_window_abc_summary = (
            self.rolling_abc_analyzer.analyze_lv3_window_abc(
                monthly_panel=monthly_panel,
                df_filtered=df_filtered,
                include_stay=False   # Up / Down만 비교
            )
        )
        self.rolling_abc_analyzer.plot_lv3_window_abc(lv3_window_abc_summary)

        print("\n[Lv3 rolling ABC 구성비 요약 테이블]")
        print(lv3_window_abc_summary)

        # =========================================================
        # Lv3 Down / Stay / Up 별 ABC 비중 비교
        # =========================================================
        abc_movement_share = plot_abc_by_movement(
            monthly_panel,
            df_filtered,
            product_sales
        )

        return {
            "df_static_filtered": df_static_filtered,
            "df_filtered": df_filtered,
            "rfm": rfm,
            "product_sales": product_sales,
            "segment_product_mix": segment_product_mix,
            "monthly_panel": monthly_panel,
            "lv3_next_month_count": t1_count,
            "lv3_next_month_ratio": t1_ratio,
            "lv3_2m_count": t2_count,
            "lv3_2m_ratio": t2_ratio,
            "lv3_next_month_by_month": by_month_t1,
            "lv3_2m_by_month": by_month_t2,
            "mobility_summary": mobility_summary,
            "movement_dist": movement_dist,
            "repeat_transition_summary": repeat_transition_summary,
            "repeat_transition_dist": repeat_transition_dist,
            "lv3_repeat_summary": lv3_repeat_summary,
            "rfm_impact_summary": rfm_impact_summary,
            "rfm_impact_results": rfm_impact_results,
            "high_m_freq_summary": high_m_freq_summary,
            "high_m_freq_dist": high_m_freq_dist,
            "high_m_cut": high_m_cut,
            "high_m_freq_cut": high_m_freq_cut,
            "high_f_m_summary": high_f_m_summary,
            "high_f_m_dist": high_f_m_dist,
            "high_f_cut": high_f_cut,
            "high_f_m_cut": high_f_m_cut,
            "abc_movement_share": abc_movement_share,
            "lv3_window_abc_df": lv3_window_abc_df,
            "lv3_window_abc_summary": lv3_window_abc_summary,
        }



# =========================================================
# 11-1. 3M / 6M / 12M RFM 기반 다음달 구매예측 검증 클래스
# =========================================================
class RollingRFMPredictor:
    def __init__(self, analysis_start="2011-01-01", analysis_end="2011-10-01"):
        self.analysis_start = pd.to_datetime(analysis_start)
        self.analysis_end = pd.to_datetime(analysis_end)
        self.rfm_analyzer = RFMAnalyzer()

    def _month_range(self):
        return pd.date_range(
            start=self.analysis_start,
            end=self.analysis_end,
            freq="MS"
        )

    def build_snapshot_panel(self, df, window_months):
        months = self._month_range()
        all_rows = []

        for month_start in months:
            window_start = month_start - pd.DateOffset(months=window_months)
            window_end = month_start
            next_month_start = month_start + pd.DateOffset(months=1)
            next_month_end = month_start + pd.DateOffset(months=2)

            window_df = df[
                (df["InvoiceDate"] >= window_start) &
                (df["InvoiceDate"] < window_end)
            ].copy()

            if window_df.empty:
                continue

            active_customers = sorted(window_df["CustomerID"].unique())
            if len(active_customers) == 0:
                continue

            rfm = self.rfm_analyzer.create_rfm_full_customer_base(
                df=window_df,
                snapshot_date=month_start,
                all_customers=active_customers,
                rolling_months=window_months
            )

            rfm = rfm[rfm["Frequency"] > 0].copy()
            if rfm.empty:
                continue

            next_month_df = df[
                (df["InvoiceDate"] >= next_month_start) &
                (df["InvoiceDate"] < next_month_end)
            ].copy()

            next_buy = (
                next_month_df.groupby("CustomerID")["InvoiceNo"]
                .nunique()
                .reset_index(name="NextMonthOrders")
            )
            next_buy["TargetBuyNextMonth"] = np.where(next_buy["NextMonthOrders"] > 0, 1, 0)

            rfm = rfm.merge(
                next_buy[["CustomerID", "TargetBuyNextMonth"]],
                on="CustomerID",
                how="left"
            )
            rfm["TargetBuyNextMonth"] = rfm["TargetBuyNextMonth"].fillna(0).astype(int)

            rfm["BaseMonth"] = month_start
            rfm["BaseMonthStr"] = month_start.strftime("%Y-%m")
            rfm["WindowMonths"] = window_months

            all_rows.append(
                rfm[[
                    "CustomerID",
                    "BaseMonth",
                    "BaseMonthStr",
                    "WindowMonths",
                    "Recency",
                    "Frequency",
                    "Monetary",
                    "TargetBuyNextMonth"
                ]]
            )

        if not all_rows:
            return pd.DataFrame(columns=[
                "CustomerID", "BaseMonth", "BaseMonthStr", "WindowMonths",
                "Recency", "Frequency", "Monetary", "TargetBuyNextMonth"
            ])

        panel = pd.concat(all_rows, ignore_index=True)
        return panel

    def evaluate_window_auc(self, panel, train_end_month="2011-07-01"):
        if panel.empty:
            return {
                "overall_auc": np.nan,
                "monthly_auc_df": pd.DataFrame(),
                "test_scored_df": pd.DataFrame(),
                "train_n": 0,
                "test_n": 0
            }

        train_end_month = pd.to_datetime(train_end_month)

        train_df = panel[panel["BaseMonth"] < train_end_month].copy()
        test_df = panel[panel["BaseMonth"] >= train_end_month].copy()

        feature_cols = ["Recency", "Frequency", "Monetary"]
        target_col = "TargetBuyNextMonth"

        if train_df.empty or test_df.empty:
            return {
                "overall_auc": np.nan,
                "monthly_auc_df": pd.DataFrame(),
                "test_scored_df": pd.DataFrame(),
                "train_n": len(train_df),
                "test_n": len(test_df)
            }

        if train_df[target_col].nunique() < 2 or test_df[target_col].nunique() < 2:
            return {
                "overall_auc": np.nan,
                "monthly_auc_df": pd.DataFrame(),
                "test_scored_df": pd.DataFrame(),
                "train_n": len(train_df),
                "test_n": len(test_df)
            }

        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )

        model.fit(train_df[feature_cols], train_df[target_col])

        test_df = test_df.copy()
        test_df["PredProb"] = model.predict_proba(test_df[feature_cols])[:, 1]

        try:
            overall_auc = roc_auc_score(test_df[target_col], test_df["PredProb"])
        except Exception:
            overall_auc = np.nan

        monthly_rows = []
        for month_str, g in test_df.groupby("BaseMonthStr"):
            if g[target_col].nunique() < 2:
                monthly_auc = np.nan
            else:
                try:
                    monthly_auc = roc_auc_score(g[target_col], g["PredProb"])
                except Exception:
                    monthly_auc = np.nan

            monthly_rows.append({
                "BaseMonthStr": month_str,
                "AUC": monthly_auc,
                "N": len(g),
                "PositiveRate": g[target_col].mean()
            })

        monthly_auc_df = pd.DataFrame(monthly_rows).sort_values("BaseMonthStr")

        return {
            "overall_auc": overall_auc,
            "monthly_auc_df": monthly_auc_df,
            "test_scored_df": test_df,
            "train_n": len(train_df),
            "test_n": len(test_df)
        }

    def run_all_windows(self, df, windows=(3, 6, 12), train_end_month="2011-07-01"):
        all_panels = {}
        all_results = []
        monthly_auc_list = []
        scored_test_list = []

        for w in windows:
            print(f"\n[예측 검증] {w}개월 window panel 생성 중...")

            panel = self.build_snapshot_panel(df, window_months=w)
            all_panels[w] = panel

            print(f"{w}M panel shape: {panel.shape}")
            if not panel.empty:
                print(panel.head())

            result = self.evaluate_window_auc(
                panel=panel,
                train_end_month=train_end_month
            )

            all_results.append({
                "WindowMonths": w,
                "OverallAUC": result["overall_auc"],
                "TrainN": result["train_n"],
                "TestN": result["test_n"],
                "TestPositiveRate": (
                    result["test_scored_df"]["TargetBuyNextMonth"].mean()
                    if not result["test_scored_df"].empty else np.nan
                )
            })

            monthly_auc_df = result["monthly_auc_df"].copy()
            if not monthly_auc_df.empty:
                monthly_auc_df["WindowMonths"] = w
                monthly_auc_list.append(monthly_auc_df)

            scored_df = result["test_scored_df"].copy()
            if not scored_df.empty:
                scored_df["WindowMonths"] = w
                scored_test_list.append(scored_df)

            print(f"\n[{w}M Overall ROC-AUC]")
            print(result["overall_auc"])

            print(f"\n[{w}M Monthly ROC-AUC]")
            print(result["monthly_auc_df"])

        summary_df = pd.DataFrame(all_results).sort_values("WindowMonths")

        if monthly_auc_list:
            monthly_auc_all = pd.concat(monthly_auc_list, ignore_index=True)
        else:
            monthly_auc_all = pd.DataFrame()

        if scored_test_list:
            scored_test_all = pd.concat(scored_test_list, ignore_index=True)
        else:
            scored_test_all = pd.DataFrame()

        return {
            "panels": all_panels,
            "summary_df": summary_df,
            "monthly_auc_all": monthly_auc_all,
            "scored_test_all": scored_test_all
        }

    def plot_overall_auc(self, summary_df):
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
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=11
            )

        plt.show()

    def plot_monthly_auc(self, monthly_auc_all):
        if monthly_auc_all.empty:
            return

        pivot = monthly_auc_all.pivot(
            index="BaseMonthStr",
            columns="WindowMonths",
            values="AUC"
        ).sort_index()

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

    def make_top_bottom_decile_summary(self, scored_test_all):
        if scored_test_all.empty:
            return pd.DataFrame()

        rows = []

        for w, g in scored_test_all.groupby("WindowMonths"):
            g = g.copy().sort_values("PredProb", ascending=False)

            n = len(g)
            cut_n = max(int(np.floor(n * 0.1)), 1)

            top = g.head(cut_n)
            bottom = g.tail(cut_n)

            rows.append({
                "WindowMonths": w,
                "Top10_N": len(top),
                "Bottom10_N": len(bottom),
                "Top10_ActualBuyRate": top["TargetBuyNextMonth"].mean(),
                "Bottom10_ActualBuyRate": bottom["TargetBuyNextMonth"].mean(),
                "Gap": top["TargetBuyNextMonth"].mean() - bottom["TargetBuyNextMonth"].mean(),
                "OverallBuyRate": g["TargetBuyNextMonth"].mean(),
                "Top10_Lift": (
                    top["TargetBuyNextMonth"].mean() / g["TargetBuyNextMonth"].mean()
                    if g["TargetBuyNextMonth"].mean() > 0 else np.nan
                )
            })

        out = pd.DataFrame(rows).sort_values("WindowMonths")
        return out

    def plot_top_bottom_gap(self, decile_summary):
        if decile_summary.empty:
            return

        x = np.arange(len(decile_summary))
        width = 0.35

        plt.figure(figsize=(8, 5))
        plt.bar(x - width/2, decile_summary["Top10_ActualBuyRate"], width, label="Top 10%")
        plt.bar(x + width/2, decile_summary["Bottom10_ActualBuyRate"], width, label="Bottom 10%")

        plt.title("Window별 Top10% vs Bottom10% 실제 구매율")
        plt.xlabel("Window")
        plt.ylabel("Actual Buy Rate")
        plt.xticks(x, decile_summary["WindowMonths"].astype(str) + "M")
        plt.legend()
        plt.show()


# =========================================================
# 12. 실행
# =========================================================
print("!!실행 시작")
if __name__ == "__main__":
    print("main 진입")
    config = RetailConfig()
    pipeline = RetailAnalysisPipeline(config)
    results = pipeline.run()
    print("run 완료")

    print(results["monthly_panel"]["AnalysisMonthStr"].value_counts().sort_index())

    print(results["monthly_panel"][[
        "CustomerID", "AnalysisMonthStr", "Customer_Grade", "IsActiveThisMonth"
    ]].head(20))

    print(results["lv3_next_month_ratio"])
    print(results["repeat_transition_summary"])
    print(results["repeat_transition_dist"])
    print(results["abc_movement_share"])



predictor = RollingRFMPredictor()
auc_results = predictor.run_all_windows(results["df_filtered"])

predictor.plot_overall_auc(auc_results["summary_df"])
predictor.plot_monthly_auc(auc_results["monthly_auc_all"])