import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# =========================================================
# 0. 설정 클래스
# =========================================================
# [발표용 설명]
# 분석에 필요한 고정값(파일명, 시트명, 분석 기간, 군집 개수 등)을
# 한 곳에서 관리하기 위한 클래스이다.
# 이렇게 하면 나중에 파일 경로, 분석 대상 국가, 군집 개수 등이 바뀌어도
# 코드 여러 곳을 수정할 필요 없이 설정 클래스만 수정하면 된다.
class RetailConfig:
    FILE_PATH = r"G:\다른 컴퓨터\내 컴퓨터\BA\module1\애널리틱스 프로그래밍(김배호 교수님, 금)\20260306_주제선정\데이터셋\4_online_retail_II.xlsx"
    SHEETS = ["Year 2009-2010", "Year 2010-2011"]
    COUNTRY = "United Kingdom"
    START_DATE = "2011-01-01"
    END_DATE = "2012-01-01"
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
    LABEL_FONT_SIZE = 16


# =========================================================
# 1. 데이터 로더 클래스
# =========================================================
# [발표용 설명]
# 이 클래스는 원본 데이터를 불러오는 역할만 담당한다.
# 기존 절차형 코드에서는 데이터 로드 코드가 전체 흐름 안에 직접 들어 있었지만,
# 객체지향 구조에서는 "데이터를 불러오는 책임"만 별도로 분리하였다.
# 이렇게 하면 데이터 적재 로직을 재사용하거나 수정하기 쉬워진다.
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
            df = self._memory_cache.copy()
        elif use_cache and self._is_cache_valid():
            print(f"파일 캐시에서 데이터를 불러왔습니다: {self.cache_path}")
            df = pd.read_pickle(self.cache_path)
            self._memory_cache = df.copy()
        else:
            print("원본 Excel 파일에서 데이터를 불러옵니다.")
            df_2009 = pd.read_excel(self.file_path, sheet_name=self.sheets[0])
            df_2010 = pd.read_excel(self.file_path, sheet_name=self.sheets[1])
            df = pd.concat([df_2009, df_2010], ignore_index=True)

            if use_cache:
                df.to_pickle(self.cache_path)
                print(f"캐시 파일을 저장했습니다: {self.cache_path}")

            self._memory_cache = df.copy()

        print("원본 shape:", df.shape)
        print(df.head())
        print(df.info())

        return df


# =========================================================
# 2. 전처리 클래스
# =========================================================
# [발표용 설명]
# 이 단계는 분석에 사용할 수 있도록 데이터를 정리하는 과정이다.
# 주요 작업은 다음과 같다.
# 1) 컬럼명 통일
# 2) 자료형 변환
# 3) 결측치 제거
# 4) 취소 주문 제거
# 5) 음수/0 데이터 제거
# 6) 비상품 코드 제거
# 7) 영국 거래만 필터링
# 8) 분석 기간 필터링
# 9) 파생변수(Year, Month, YearMonth, Revenue) 생성
#
# 객체지향 구조에서는 이 전처리 과정을 하나의 클래스 안에 넣어
# 데이터 정제 로직을 독립적으로 관리할 수 있도록 했다.
class RetailPreprocessor:
    def __init__(self, country, start_date, end_date, non_product_codes):
        self.country = country
        self.start_date = start_date
        self.end_date = end_date
        self.non_product_codes = non_product_codes

    def preprocess(self, df):
        df = df.copy()

        # 2-1. 컬럼명 공백 제거
        df.columns = df.columns.str.strip()

        # 2-2. 컬럼명 통일
        rename_dict = {}
        if "Invoice" in df.columns:
            rename_dict["Invoice"] = "InvoiceNo"
        if "Price" in df.columns:
            rename_dict["Price"] = "UnitPrice"
        if "Customer ID" in df.columns:
            rename_dict["Customer ID"] = "CustomerID"

        df = df.rename(columns=rename_dict)

        # 2-3. 필수 컬럼 존재 여부 확인
        required_cols = [
            "InvoiceNo", "StockCode", "Country",
            "UnitPrice", "CustomerID", "InvoiceDate", "Quantity"
        ]
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"{col} 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")

        # 2-4. 자료형 변환
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
        df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

        df["InvoiceNo"] = df["InvoiceNo"].astype(str)
        df["StockCode"] = df["StockCode"].astype(str)
        df["Country"] = df["Country"].astype(str)

        if "Description" not in df.columns:
            df["Description"] = ""
        df["Description"] = df["Description"].astype(str)

        # 2-5. 매출액 변수 생성
        df["Revenue"] = df["Quantity"] * df["UnitPrice"]

        # 2-6. 핵심 결측치 제거
        print("CustomerID 결측 전:", df["CustomerID"].isna().sum())
        df = df.dropna(subset=["CustomerID", "InvoiceDate", "StockCode"])
        print("CustomerID 결측 제거 후:", df["CustomerID"].isna().sum())

        # 2-7. CustomerID 형식 정리
        df["CustomerID"] = df["CustomerID"].astype(float).astype(int).astype(str)

        # 2-8. 취소 주문 제거
        cancel_count = df["InvoiceNo"].str.upper().str.startswith("C").sum()
        print("cancel_count:", cancel_count)
        df = df[~df["InvoiceNo"].str.upper().str.startswith("C")]

        # 2-9. 수량/단가가 양수인 거래만 사용
        df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

        # 2-10. 비상품 코드 제거
        df = df[~df["StockCode"].isin(self.non_product_codes)]

        # 2-11. 영국 거래만 사용
        df = df[df["Country"] == self.country]

        # 2-12. 분석 기간 필터링
        df = df[
            (df["InvoiceDate"] >= self.start_date) &
            (df["InvoiceDate"] < self.end_date)
        ].copy()

        # 2-13. 날짜 파생변수 생성
        df["Year"] = df["InvoiceDate"].dt.year
        df["Month"] = df["InvoiceDate"].dt.month
        df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)

        # 2-14. 중복 확인
        dup_count = df.duplicated().sum()
        print("중복 개수(참고용):", dup_count)

        print("전처리 후 shape:", df.shape)
        print(df.head())
        print(df.info())

        return df


# =========================================================
# 3. RFM 분석 클래스
# =========================================================
# [발표용 설명]
# RFM은 고객을 구매 행동 기준으로 분석하는 대표적인 방법이다.
# - Recency: 얼마나 최근에 구매했는가
# - Frequency: 얼마나 자주 구매했는가
# - Monetary: 얼마나 많이 구매했는가
#
# 이 클래스는 고객 단위로 RFM 테이블을 생성하고,
# 이상치를 제거해 분석에 적합한 고객 데이터만 남기는 역할을 한다.
class RFMAnalyzer:
    def create_rfm(self, df):
        # 3-1. 기준 시점 설정
        snapshot_date = df["InvoiceDate"].max()

        # 3-2. 고객별 첫 구매일 / 마지막 구매일 계산
        customer_dates = df.groupby("CustomerID").agg(
            FirstPurchaseDate=("InvoiceDate", "min"),
            LastPurchaseDate=("InvoiceDate", "max")
        ).reset_index()

        # 3-3. 고객별 구매 빈도와 총매출 계산
        rfm = df.groupby("CustomerID").agg(
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Revenue", "sum")
        ).reset_index()

        rfm = rfm.merge(customer_dates, on="CustomerID", how="left")

        # 3-4. Recency 계산
        rfm["Recency"] = ((snapshot_date - rfm["LastPurchaseDate"]).dt.days / 30).round(1)

        rfm = rfm[
            ["CustomerID", "FirstPurchaseDate", "LastPurchaseDate",
             "Recency", "Frequency", "Monetary"]
        ]

        return rfm

    def remove_outliers(self, rfm, df):
        # [발표용 설명]
        # RFM 값이 지나치게 큰 일부 고객은 전체 군집화 결과를 왜곡할 수 있으므로
        # Frequency와 Monetary 기준 상위 1%를 제거하여 분석 안정성을 높였다.

        # 4-1. 상위 1% 컷오프
        freq_cut = rfm["Frequency"].quantile(0.99)
        mon_cut = rfm["Monetary"].quantile(0.99)

        rfm = rfm[
            (rfm["Frequency"] <= freq_cut) &
            (rfm["Monetary"] <= mon_cut)
        ].copy()

        print("이상치 제거 후 고객 수:", len(rfm))
        print("Frequency 상한:", freq_cut)
        print("Monetary 상한:", mon_cut)
        print(rfm.head())
        print(rfm.describe())

        # 4-2. 이상치 제거 후 남은 고객만 거래 데이터에 반영
        valid_customers = set(rfm["CustomerID"])
        df_filtered = df[df["CustomerID"].isin(valid_customers)].copy()

        return rfm, df_filtered


# =========================================================
# 4. 머신러닝 기반 고객 세분화 클래스
# =========================================================
# [발표용 설명]
# 이 단계에서는 RFM 데이터를 이용해 K-means 군집화를 수행한다.
# 목적은 고객을 비슷한 구매 행동을 가진 그룹으로 자동 분류하는 것이다.
#
# 주요 절차는 다음과 같다.
# 1) RFM 변수 추출
# 2) 스케일링
# 3) 적절한 k 탐색
# 4) 최종 K-means 적용
# 5) 군집별 평균값 해석
# 6) 사람이 이해하기 쉬운 등급명 부여
class CustomerSegmenter:
    def __init__(self, k_range, final_k, random_state=42, n_init=10):
        self.k_range = k_range
        self.final_k = final_k
        self.random_state = random_state
        self.n_init = n_init

        self.scaler = StandardScaler()
        self.kmeans = None
        self.inertia = []
        self.silhouette_scores = []
        self.cluster_to_grade = {}

    def evaluate_k(self, rfm):
        # 5-1. 군집화에 사용할 RFM 변수 추출
        X = rfm[["Recency", "Frequency", "Monetary"]]

        # 5-2. 스케일링
        # 변수 단위가 다르기 때문에 표준화 과정을 수행한다.
        X_scaled = self.scaler.fit_transform(X)

        # 5-3. 적절한 k 탐색
        self.inertia = []
        self.silhouette_scores = []

        print("\n[Silhouette Score 결과]")

        for k in self.k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=self.n_init)
            labels = km.fit_predict(X_scaled)

            self.inertia.append(km.inertia_)
            score = silhouette_score(X_scaled, labels)
            self.silhouette_scores.append(score)

            print(f"k={k}, silhouette={score:.3f}")

        return X_scaled

    def plot_k_result(self):
        # 5-3-1. k 평가 그래프
        # Elbow와 silhouette score를 동시에 확인하여 적절한 k를 판단한다.
        fig, ax1 = plt.subplots(figsize=(8, 5))
        line1, = ax1.plot(
            self.k_range,
            self.inertia,
            marker='o',
            color='blue',
            label='Elbow'
        )
        ax1.set_xlabel('k')
        ax1.set_ylabel('Elbow', color='blue')

        ax2 = ax1.twinx()
        line2, = ax2.plot(
            self.k_range,
            self.silhouette_scores,
            marker='s',
            linestyle='--',
            color='green',
            label='Silhouette Score'
        )
        ax2.set_ylabel('Silhouette Score', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        ax1.legend([line1, line2], ['Elbow', 'Silhouette Score'], fontsize=16)
        plt.title("Optimal k Analysis")
        plt.show()

    def fit_clusters(self, rfm, X_scaled, grade_names):
        # 5-4. 최종 K-means 적용
        self.kmeans = KMeans(
            n_clusters=self.final_k,
            random_state=self.random_state,
            n_init=self.n_init
        )
        rfm["Cluster"] = self.kmeans.fit_predict(X_scaled)

        # 5-5. 군집별 평균 비교
        cluster_summary = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
        print("\n[Cluster Summary]")
        print(cluster_summary)

        # 5-6. 최종 실루엣 점수 확인
        final_score = silhouette_score(X_scaled, rfm["Cluster"])
        print("\n최종 Silhouette Score:", final_score)

        # 5-7. 군집 중심 해석
        centers = pd.DataFrame(
            self.scaler.inverse_transform(self.kmeans.cluster_centers_),
            columns=["Recency", "Frequency", "Monetary"]
        )
        centers["Cluster"] = range(len(centers))
        print("\n[Cluster Centers]")
        print(centers)

        # 5-8. 군집에 등급명 부여
        # Monetary, Frequency가 높고 Recency가 낮을수록 우수 고객으로 해석
        centers = centers.sort_values(
            by=["Monetary", "Frequency", "Recency"],
            ascending=[True, True, False]
        ).reset_index(drop=True)

        self.cluster_to_grade = {}
        for i, row in centers.iterrows():
            self.cluster_to_grade[row["Cluster"]] = grade_names[i]

        rfm["Customer_Grade"] = rfm["Cluster"].map(self.cluster_to_grade)

        print("\n[Cluster -> Grade Mapping]")
        print(self.cluster_to_grade)
        print(rfm["Customer_Grade"].value_counts())

        return rfm

    def plot_cluster_result(self, rfm):
        # 5-9. 군집 시각화
        # Frequency와 Monetary를 log 변환하여 산점도로 표현하고,
        # 중심점도 함께 표시한다.
        centers = pd.DataFrame(
            self.scaler.inverse_transform(self.kmeans.cluster_centers_),
            columns=["Recency", "Frequency", "Monetary"]
        )
        centers["Cluster"] = range(len(centers))

        rfm_plot = rfm.copy()
        rfm_plot["log_Frequency"] = np.log1p(rfm_plot["Frequency"])
        rfm_plot["log_Monetary"] = np.log1p(rfm_plot["Monetary"])

        centers_plot = centers.copy()
        centers_plot["log_Frequency"] = np.log1p(centers_plot["Frequency"])
        centers_plot["log_Monetary"] = np.log1p(centers_plot["Monetary"])

        plt.figure(figsize=(10, 7))

        for grade in sorted(rfm_plot["Customer_Grade"].dropna().unique()):
            subset = rfm_plot[rfm_plot["Customer_Grade"] == grade]
            plt.scatter(
                subset["log_Frequency"],
                subset["log_Monetary"],
                label=grade,
                alpha=0.5,
                s=20
            )

        plt.scatter(
            centers_plot["log_Frequency"],
            centers_plot["log_Monetary"],
            c="black",
            marker="X",
            s=250,
            label="Centers"
        )

        for _, row in centers_plot.iterrows():
            plt.text(
                row["log_Frequency"] + 0.02,
                row["log_Monetary"] + 0.02,
                self.cluster_to_grade[row["Cluster"]]
            )

        plt.xlabel("Frequency (log)")
        plt.ylabel("Monetary (log)")
        plt.title("고객 등급별 세분화 결과")
        plt.legend(fontsize=16)
        plt.show()


# =========================================================
# 5. 상품 분석 클래스
# =========================================================
# [발표용 설명]
# 고객만 분석하는 것이 아니라 상품 자체의 매출 기여도도 함께 확인하기 위해
# ABC 분석과 Long Tail 분석을 수행한다.
#
# ABC 분석:
# - A: 상위 핵심 매출 상품
# - B: 중간 수준 기여 상품
# - C: 낮은 기여 상품
#
# Long Tail 분석:
# 소수의 핵심 상품이 큰 매출을 만들고,
# 다수의 상품은 상대적으로 낮은 매출을 갖는 구조를 확인한다.
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
        # 6-1. 상품별 총 판매량 및 총매출 집계
        product_sales = df_filtered.groupby("StockCode").agg(
            Quantity=("Quantity", "sum"),
            Revenue=("Revenue", "sum")
        ).reset_index()

        # 6-2. 매출 기준 내림차순 정렬
        product_sales = product_sales.sort_values(by="Revenue", ascending=False)

        # 6-3. 누적매출 및 누적비율 계산
        product_sales["CumulativeRevenue"] = product_sales["Revenue"].cumsum()
        product_sales["CumulativeRatio"] = (
            product_sales["CumulativeRevenue"] / product_sales["Revenue"].sum()
        )

        # 6-4. ABC 등급 분류
        product_sales["ABC_Class"] = product_sales["CumulativeRatio"].apply(self.abc_class)

        print("상위 매출 상품:\n", product_sales.head(10))
        print("A/B/C 상품 개수:\n", product_sales["ABC_Class"].value_counts())

        return product_sales

    def plot_abc_share(self, product_sales):
        # 6-5. ABC 등급별 매출 비중 확인
        abc_revenue_share = product_sales.groupby("ABC_Class")["Revenue"].sum()
        abc_revenue_share = abc_revenue_share / abc_revenue_share.sum()
        abc_revenue_share = abc_revenue_share.reindex(["A", "B", "C"])

        plt.figure(figsize=(6, 4))
        abc_revenue_share.plot(kind="bar")
        plt.title("ABC Class Revenue Share")
        plt.xlabel("ABC Class")
        plt.ylabel("Revenue Share")
        plt.show()

    def plot_long_tail(self, product_sales):
        # 6-6. 상품 랭크별 매출 분포 시각화
        product_rank = product_sales.sort_values(by="Revenue", ascending=False).reset_index(drop=True)

        plt.figure(figsize=(10, 6))
        plt.plot(product_rank.index, product_rank["Revenue"])
        plt.yscale("log")
        plt.title("Product Long Tail Distribution")
        plt.xlabel("Product Rank")
        plt.ylabel("Revenue (log scale)")
        plt.show()


# =========================================================
# 6. 고객 × 상품 통합 클래스
# =========================================================
# [발표용 설명]
# 고객 분석 결과와 상품 분석 결과를 연결하여,
# "어떤 고객이 어떤 등급의 상품을 소비하는지"를 함께 볼 수 있도록
# 고객-상품 통합 테이블을 생성한다.
#
# 이 테이블은 이후 VIP 고객의 상품 소비 패턴,
# 세그먼트별 상품 구성비 분석 등에 활용된다.
class CustomerProductIntegrator:
    def integrate(self, df_filtered, rfm, product_sales):
        # 7-1. 고객-상품 단위 집계
        customer_product = df_filtered.groupby(["CustomerID", "StockCode"]).agg(
            Quantity=("Quantity", "sum"),
            Revenue=("Revenue", "sum")
        ).reset_index()

        # 7-2. 고객 RFM/등급 정보 결합
        customer_product = customer_product.merge(
            rfm[["CustomerID", "Recency", "Frequency", "Monetary", "Customer_Grade"]],
            on="CustomerID",
            how="left"
        )

        # 7-3. 상품 ABC 등급 정보 결합
        customer_product = customer_product.merge(
            product_sales[["StockCode", "ABC_Class"]],
            on="StockCode",
            how="left"
        )

        print("통합 후 customer_product:\n", customer_product.head())

        return customer_product


# =========================================================
# 7. EDA 클래스
# =========================================================
# [발표용 설명]
# 본격적인 세분화 결과 해석 전에 데이터의 전체적인 특성을 먼저 확인하기 위해
# 기초 탐색적 데이터 분석(EDA)을 수행한다.
# 이를 통해 매출 분포, 수량/단가 분포, 월별 매출 추이, 변수 간 관계 등을 파악한다.
class EDAAnalyzer:
    def run_basic_eda(self, df, rfm):
        print("\n[매출 분포]")
        print(df["Revenue"].describe())

        print("\n[수량/가격 분포]")
        print(df[["Quantity", "UnitPrice"]].describe())

        # 9-1. 월별 총매출 추이
        monthly_sales = df.groupby("YearMonth")["Revenue"].sum()

        plt.figure(figsize=(10, 5))
        monthly_sales.plot(marker="o")
        plt.title("월별 총매출 추이")
        plt.xlabel("Year-Month")
        plt.ylabel("Revenue")
        plt.xticks(rotation=45)
        plt.show()

        print("\n[RFM 분포]")
        print(rfm.describe())

        print("\n[변수 관계]")
        print(df[["Quantity", "UnitPrice", "Revenue"]].corr())


# =========================================================
# 8. VIP 및 규칙 기반 세그먼트 분석 클래스
# =========================================================
# [발표용 설명]
# 머신러닝 군집화와 별도로, 규칙 기반 세그먼트 분석도 추가로 수행했다.
# 이유는 군집화 결과를 더 쉽게 해석하고,
# VIP 고객과 일반 고객의 차이를 직관적으로 설명하기 위해서이다.
#
# 이 단계에서는
# 1) VIP 고객 정의
# 2) VIP 고객 매출 기여도 확인
# 3) RFM 점수화
# 4) 규칙 기반 고객 세그먼트 생성
# 5) VIP 고객의 상품 소비 패턴 분석
# 을 수행한다.
class RuleBasedSegmentAnalyzer:
    @staticmethod
    def segment_customer(row):
        if int(row["R_score"]) >= 4 and int(row["F_score"]) >= 4 and int(row["M_score"]) >= 4:
            return "VIP"
        elif int(row["R_score"]) <= 2 and int(row["F_score"]) <= 2:
            return "At-risk"
        elif int(row["F_score"]) >= 4:
            return "Loyal"
        else:
            return "General"

    def analyze(self, rfm, df_filtered, customer_product):
        # 10-1. VIP 고객 정의
        # Monetary 기준 상위 25% 고객을 VIP로 정의
        vip_threshold = rfm["Monetary"].quantile(0.75)
        vip_customers = rfm[rfm["Monetary"] >= vip_threshold]

        vip_count = len(vip_customers)
        total_customers = len(rfm)
        vip_customer_ratio = vip_count / total_customers

        print("VIP 고객 수:", vip_count)
        print("VIP 고객 비율:", vip_customer_ratio)

        # 10-2. VIP 고객의 매출 기여도 확인
        vip_df = df_filtered[df_filtered["CustomerID"].isin(vip_customers["CustomerID"])]
        vip_revenue = vip_df["Revenue"].sum()
        total_revenue = df_filtered["Revenue"].sum()
        vip_ratio = vip_revenue / total_revenue

        print("VIP 매출 비율:", vip_ratio)

        # 10-3. RFM 점수화
        rfm["R_score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])
        rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        rfm["M_score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])

        rfm["RFM_score"] = (
            rfm["R_score"].astype(str) +
            rfm["F_score"].astype(str) +
            rfm["M_score"].astype(str)
        )

        print(rfm.head())

        # 10-4. 규칙 기반 고객 세그먼트 생성
        rfm["Segment2"] = rfm.apply(self.segment_customer, axis=1)
        print(rfm["Segment2"].value_counts())

        # 11-1. VIP 고객의 A상품 매출 비율
        vip_data = customer_product[
            customer_product["CustomerID"].isin(vip_customers["CustomerID"])
        ]

        vip_total = vip_data["Revenue"].sum()
        vip_A = vip_data[vip_data["ABC_Class"] == "A"]["Revenue"].sum()
        vip_A_ratio = vip_A / vip_total

        print("VIP 전체 매출:", vip_total)
        print("VIP의 A상품 매출:", vip_A)
        print("VIP의 A상품 매출 비율:", vip_A_ratio)

        # 11-2. 고객 세그먼트별 상품 소비 패턴 분석
        rfm["Segment"] = "Low"

        q75 = rfm["Monetary"].quantile(0.75)
        q50 = rfm["Monetary"].quantile(0.50)

        rfm.loc[rfm["Monetary"] >= q75, "Segment"] = "VIP"
        rfm.loc[
            (rfm["Monetary"] >= q50) &
            (rfm["Monetary"] < q75),
            "Segment"
        ] = "Mid"

        if "Segment" in customer_product.columns:
            customer_product = customer_product.drop(columns=["Segment"])

        customer_product = customer_product.merge(
            rfm[["CustomerID", "Segment"]],
            on="CustomerID",
            how="left"
        )

        segment_abc = customer_product.groupby(["Segment", "ABC_Class"])["Revenue"].sum().reset_index()

        segment_total = customer_product.groupby("Segment")["Revenue"].sum().reset_index()
        segment_total.columns = ["Segment", "TotalRevenue"]

        segment_abc = segment_abc.merge(segment_total, on="Segment", how="left")
        segment_abc["Ratio"] = segment_abc["Revenue"] / segment_abc["TotalRevenue"]

        print(segment_abc)

        return rfm, customer_product, vip_customers


# =========================================================
# 9. 세그먼트 요약/시각화 클래스
# =========================================================
# [발표용 설명]
# 마지막 단계에서는 머신러닝으로 만든 고객 등급을 기준으로
# 세그먼트별 특징을 더 명확하게 보여주는 요약 지표와 그래프를 생성한다.
#
# 주요 지표:
# - Customers: 고객 수
# - Orders: 주문 수
# - Revenue: 총매출
# - AOV: 객단가
# - Revenue_per_Customer: 고객당 매출
# - Customer_Share: 고객 비중
# - Revenue_Share: 매출 비중
class SegmentReportAnalyzer:
    def build_summary(self, df_filtered, rfm):
        # 12-1. 주문 단위 매출 요약
        order_summary = df_filtered.groupby(["CustomerID", "InvoiceNo"], as_index=False).agg(
            OrderRevenue=("Revenue", "sum"),
            YearMonth=("YearMonth", "max")
        )

        order_summary = order_summary.merge(
            rfm[["CustomerID", "Customer_Grade"]],
            on="CustomerID",
            how="left"
        )

        # 12-2. 세그먼트별 요약 지표 계산
        segment_summary = order_summary.groupby("Customer_Grade").agg(
            Customers=("CustomerID", "nunique"),
            Orders=("InvoiceNo", "nunique"),
            Revenue=("OrderRevenue", "sum")
        ).reset_index()

        segment_summary["AOV"] = segment_summary["Revenue"] / segment_summary["Orders"]
        segment_summary["Revenue_per_Customer"] = (
            segment_summary["Revenue"] / segment_summary["Customers"]
        )
        segment_summary["Customer_Share"] = (
            segment_summary["Customers"] / segment_summary["Customers"].sum()
        )
        segment_summary["Revenue_Share"] = (
            segment_summary["Revenue"] / segment_summary["Revenue"].sum()
        )

        segment_summary["GradeOrder"] = (
            segment_summary["Customer_Grade"].str.extract(r"Lv(\d)").astype(int)
        )
        segment_summary = segment_summary.sort_values("GradeOrder")

        print(segment_summary)

        return segment_summary

    def plot_reports(self, segment_summary, customer_product, rfm, df_filtered, grade_names):
        # 12-3. 세그먼트별 객단가(AOV)
        plt.figure(figsize=(8, 5))
        plt.bar(segment_summary["Customer_Grade"], segment_summary["AOV"])
        plt.title("세그먼트별 객단가(AOV)")
        plt.xlabel("Customer Grade")
        plt.ylabel("AOV")
        plt.show()

        # 12-4. 세그먼트별 고객당 매출
        plt.figure(figsize=(8, 5))
        plt.bar(segment_summary["Customer_Grade"], segment_summary["Revenue_per_Customer"])
        plt.title("세그먼트별 고객당 매출")
        plt.xlabel("Customer Grade")
        plt.ylabel("Revenue per Customer")
        plt.show()

        # 12-5. 세그먼트별 고객 비중 vs 매출 비중
        compare_df = segment_summary[
            ["Customer_Grade", "Customer_Share", "Revenue_Share"]
        ].copy().set_index("Customer_Grade")

        compare_df.plot(kind="bar", figsize=(9, 5))
        plt.title("세그먼트별 고객 비중 vs 매출 비중")
        plt.ylabel("Share")
        plt.xticks(rotation=0)
        plt.show()

        # 12-6. 세그먼트별 상품 ABC 매출 구성비
        if "Customer_Grade" not in customer_product.columns:
            customer_product = customer_product.merge(
                rfm[["CustomerID", "Customer_Grade"]],
                on="CustomerID",
                how="left"
            )

        segment_abc_grade = customer_product.groupby(
            ["Customer_Grade", "ABC_Class"]
        )["Revenue"].sum().reset_index()

        segment_abc_grade["GradeOrder"] = (
            segment_abc_grade["Customer_Grade"].str.extract(r"Lv(\d)").astype(int)
        )
        segment_abc_grade = segment_abc_grade.sort_values(["GradeOrder", "ABC_Class"])

        segment_abc_pivot = segment_abc_grade.pivot(
            index="Customer_Grade",
            columns="ABC_Class",
            values="Revenue"
        ).fillna(0)

        for col in ["A", "B", "C"]:
            if col not in segment_abc_pivot.columns:
                segment_abc_pivot[col] = 0

        segment_abc_pivot = segment_abc_pivot[["A", "B", "C"]]
        segment_abc_ratio = segment_abc_pivot.div(segment_abc_pivot.sum(axis=1), axis=0)

        segment_abc_ratio.plot(kind="bar", stacked=True, figsize=(9, 5))
        plt.title("세그먼트별 상품 ABC 매출 구성비")
        plt.xlabel("Customer Grade")
        plt.ylabel("Revenue Share within Segment")
        plt.xticks(rotation=0)
        plt.show()

        # 12-7. 세그먼트별 월별 매출 추이
        monthly_grade_sales = df_filtered.merge(
            rfm[["CustomerID", "Customer_Grade"]],
            on="CustomerID",
            how="left"
        )

        monthly_grade_sales = monthly_grade_sales.groupby(
            ["YearMonth", "Customer_Grade"]
        )["Revenue"].sum().reset_index()

        grade_month_pivot = monthly_grade_sales.pivot(
            index="YearMonth",
            columns="Customer_Grade",
            values="Revenue"
        ).fillna(0)

        ordered_cols = [g for g in grade_names if g in grade_month_pivot.columns]
        grade_month_pivot = grade_month_pivot[ordered_cols]

        plt.figure(figsize=(10, 6))
        for col in grade_month_pivot.columns:
            plt.plot(grade_month_pivot.index, grade_month_pivot[col], marker="o", label=col)

        plt.title("세그먼트별 월별 매출 추이")
        plt.xlabel("Year-Month")
        plt.ylabel("Revenue")
        plt.xticks(rotation=45)
        plt.legend(fontsize=16)
        plt.show()


# =========================================================
# 10. 전체 실행 클래스
# =========================================================
# [발표용 설명]
# 이 클래스는 전체 분석 흐름을 순서대로 실행하는 파이프라인 역할을 한다.
# 즉, "데이터 불러오기 → 전처리 → RFM → 군집화 → 상품 분석 → 통합 분석 → 시각화"
# 전 과정을 하나의 흐름으로 관리한다.
#
# 객체지향 구조의 장점은 각 기능이 클래스별로 분리되어 있으면서도,
# 필요할 때는 이 파이프라인을 통해 전체 과정을 한 번에 실행할 수 있다는 점이다.
class RetailAnalysisPipeline:
    def __init__(self, config):
        self.config = config

        self.loader = RetailDataLoader(config.FILE_PATH, config.SHEETS, config.CACHE_PATH)
        self.preprocessor = RetailPreprocessor(
            config.COUNTRY,
            config.START_DATE,
            config.END_DATE,
            config.NON_PRODUCT_CODES
        )
        self.rfm_analyzer = RFMAnalyzer()
        self.segmenter = CustomerSegmenter(
            config.K_RANGE,
            config.FINAL_K,
            config.RANDOM_STATE,
            config.N_INIT
        )
        self.product_analyzer = ProductAnalyzer()
        self.integrator = CustomerProductIntegrator()
        self.eda_analyzer = EDAAnalyzer()
        self.rule_analyzer = RuleBasedSegmentAnalyzer()
        self.report_analyzer = SegmentReportAnalyzer()

    def run(self):
        # =========================================================
        # 0. 시각화 환경 설정
        # =========================================================
        # [발표용 설명]
        # 한글 폰트가 깨지지 않도록 시각화 환경을 먼저 설정한다.
        plt.rcParams["font.family"] = "Malgun Gothic"
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["axes.titlesize"] = self.config.LABEL_FONT_SIZE
        plt.rcParams["axes.labelsize"] = self.config.LABEL_FONT_SIZE
        plt.rcParams["xtick.labelsize"] = self.config.LABEL_FONT_SIZE
        plt.rcParams["ytick.labelsize"] = self.config.LABEL_FONT_SIZE
        plt.rcParams["legend.fontsize"] = self.config.LABEL_FONT_SIZE

        # =========================================================
        # 1. 데이터 불러오기
        # =========================================================
        df = self.loader.load_data()

        # =========================================================
        # 2. 데이터 전처리
        # =========================================================
        df = self.preprocessor.preprocess(df)

        # =========================================================
        # 3. RFM 생성
        # =========================================================
        rfm = self.rfm_analyzer.create_rfm(df)

        # =========================================================
        # 4. 이상치 제거
        # =========================================================
        rfm, df_filtered = self.rfm_analyzer.remove_outliers(rfm, df)

        # =========================================================
        # 5. 머신러닝 기반 고객 세분화 (K-means)
        # =========================================================
        X_scaled = self.segmenter.evaluate_k(rfm)
        self.segmenter.plot_k_result()
        rfm = self.segmenter.fit_clusters(rfm, X_scaled, self.config.GRADE_NAMES)
        self.segmenter.plot_cluster_result(rfm)

        # =========================================================
        # 6. 상품 분석 (ABC / Long Tail)
        # =========================================================
        product_sales = self.product_analyzer.analyze_products(df_filtered)
        self.product_analyzer.plot_abc_share(product_sales)
        self.product_analyzer.plot_long_tail(product_sales)

        # =========================================================
        # 7. 고객 × 상품 통합 테이블
        # =========================================================
        customer_product = self.integrator.integrate(df_filtered, rfm, product_sales)

        # =========================================================
        # 8. 결과 확인
        # =========================================================
        # [발표용 설명]
        # 각 단계에서 생성된 핵심 데이터셋의 크기와 예시를 확인하여,
        # 전처리 및 분석 결과가 정상적으로 생성되었는지 점검한다.
        print("정리된 거래 데이터:", df.shape)
        print("RFM 테이블:", rfm.shape)
        print("상품 매출 테이블:", product_sales.shape)
        print("고객-상품 테이블:", customer_product.shape)

        print("\n[거래 데이터]")
        print(df.head())

        print("\n[RFM]")
        print(rfm.head())

        print("\n[상품 매출]")
        print(product_sales.head())

        print("\n[고객-상품]")
        print(customer_product.head())

        # =========================================================
        # 9. EDA
        # =========================================================
        self.eda_analyzer.run_basic_eda(df, rfm)

        # =========================================================
        # 10. VIP 및 규칙 기반 세그먼트 분석
        # 11. 고객 × 상품 통합 분석
        # =========================================================
        rfm, customer_product, vip_customers = self.rule_analyzer.analyze(
            rfm, df_filtered, customer_product
        )

        # =========================================================
        # 12. 세그먼트(머신러닝 등급) 시각화용 요약
        # =========================================================
        segment_summary = self.report_analyzer.build_summary(df_filtered, rfm)
        self.report_analyzer.plot_reports(
            segment_summary,
            customer_product,
            rfm,
            df_filtered,
            self.config.GRADE_NAMES
        )


# =========================================================
# 11. 실행
# =========================================================
# [발표용 설명]
# 프로그램 실행 시 설정 클래스를 기반으로 파이프라인 객체를 생성하고,
# run() 메서드를 호출하여 전체 분석을 수행한다.
if __name__ == "__main__":
    config = RetailConfig()
    pipeline = RetailAnalysisPipeline(config)
    pipeline.run()
