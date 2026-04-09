import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =========================================================
# 0. 시각화 환경 설정
# =========================================================
plt.rcParams["font.family"] = "AppleGothic"   # Mac
plt.rcParams["axes.unicode_minus"] = False

# =========================================================
# 1. 데이터 불러오기
# =========================================================
df_2009 = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2009-2010")
df_2010 = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")

# =========================================================
# 2. 두 시트 합치기
# =========================================================
df = pd.concat([df_2009, df_2010], ignore_index=True)

print("원본 shape:", df.shape)
print(df.head())
print(df.info())

# =========================================================
# 3. 데이터 전처리
# =========================================================

# 3-1. 컬럼명 정리
df.columns = df.columns.str.strip()

# 3-2. 컬럼명 통일
rename_dict = {}
if "Invoice" in df.columns:
    rename_dict["Invoice"] = "InvoiceNo"
if "Price" in df.columns:
    rename_dict["Price"] = "UnitPrice"
if "Customer ID" in df.columns:
    rename_dict["Customer ID"] = "CustomerID"

df = df.rename(columns=rename_dict)

# 3-3. 필수 컬럼 확인
required_cols = ["InvoiceNo", "StockCode", "Country", "UnitPrice", "CustomerID", "InvoiceDate"]
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"{col} 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")

# 3-4. 타입 변환
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

df["InvoiceNo"] = df["InvoiceNo"].astype(str)
df["StockCode"] = df["StockCode"].astype(str)
df["Country"] = df["Country"].astype(str)

if "Description" not in df.columns:
    df["Description"] = ""
df["Description"] = df["Description"].astype(str)

# 3-5. Revenue 생성
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

# 3-6. 결측 제거
print("CustomerID 결측 전:", df["CustomerID"].isna().sum())
df = df.dropna(subset=["CustomerID", "InvoiceDate", "StockCode"])
print("CustomerID 결측 제거 후:", df["CustomerID"].isna().sum())

# 3-7. CustomerID 타입 통일
df["CustomerID"] = df["CustomerID"].astype(float).astype(int).astype(str)

# 3-8. 취소 주문 제거
cancel_count = df["InvoiceNo"].str.upper().str.startswith("C").sum()
print("cancel_count:", cancel_count)
df = df[~df["InvoiceNo"].str.upper().str.startswith("C")]

# 3-9. 양수만 유지
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

# 3-10. 비상품 코드 제거
NON_PRODUCT_CODES = ["POST", "D", "M", "BANK CHARGES", "DOT", "C2"]
df = df[~df["StockCode"].isin(NON_PRODUCT_CODES)]

# 3-11. 국가 필터
df = df[df["Country"] == "United Kingdom"]

# 3-12. 기간 필터
df = df[
    (df["InvoiceDate"] >= "2011-01-01") &
    (df["InvoiceDate"] < "2012-01-01")
].copy()

# 3-13. 날짜 변수 생성
df["Year"] = df["InvoiceDate"].dt.year
df["Month"] = df["InvoiceDate"].dt.month
df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)

# 3-14. 중복은 제거하지 않고 개수만 확인
dup_count = df.duplicated().sum()
print("중복 개수(참고용):", dup_count)

print("전처리 후 shape:", df.shape)
print(df.head())
print(df.info())

# =========================================================
# 4. RFM 생성
# =========================================================
snapshot_date = df["InvoiceDate"].max()

customer_dates = df.groupby("CustomerID").agg(
    FirstPurchaseDate=("InvoiceDate", "min"),
    LastPurchaseDate=("InvoiceDate", "max")
).reset_index()

# 기본 RFM
rfm = df.groupby("CustomerID").agg(
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("Revenue", "sum")
).reset_index()

rfm = rfm.merge(customer_dates, on="CustomerID", how="left")

rfm["Recency"] = ((snapshot_date - rfm["LastPurchaseDate"]).dt.days/30).round(1)

rfm = rfm[
    ["CustomerID", "FirstPurchaseDate", "LastPurchaseDate", "Recency", "Frequency", "Monetary"]
]

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

valid_customers = set(rfm["CustomerID"])
df_filtered = df[df["CustomerID"].isin(valid_customers)].copy()
# =========================================================
# 5. 머신러닝 기반 고객 세분화 (K-means)
# =========================================================

# 5-1. RFM 데이터 준비
X = rfm[['Recency', 'Frequency', 'Monetary']]

# 5-2. 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5-3. 최적 k 찾기
inertia = []
silhouette_scores = []
k_range = range(2, 11)

print("\n[Silhouette Score 결과]")

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    inertia.append(km.inertia_)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

    print(f"k={k}, silhouette={score:.3f}")

# 5-3-1. 평가 그래프
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(k_range, inertia, 
         marker='o',
         color='blue',
         label='Inertia')
ax1.set_xlabel('k')
ax1.set_ylabel('Inertia', color='blue')

ax2 = ax1.twinx()
ax2.plot(k_range, silhouette_scores, 
         marker='s',
         linestyle='--',
         color='green',
         label='Silhouette Score')
ax2.set_ylabel('Silhouette Score', color='green')
ax2.tick_params(axis='y', labelcolor='red')

plt.title("Optimal k Analysis")
plt.show()

# 5-4. K-means 적용
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(X_scaled)

# 5-5. 클러스터 해석
cluster_summary = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
print("\n[Cluster Summary]")
print(cluster_summary)

# 5-6. 최종 평가
final_score = silhouette_score(X_scaled, rfm['Cluster'])
print("\n최종 Silhouette Score:", final_score)

# 5-7. 클러스터 중심
centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=['Recency', 'Frequency', 'Monetary']
)
centers['Cluster'] = range(len(centers))
print("\n[Cluster Centers]")
print(centers)

# 5-8. 등급 이름 부여
centers = centers.sort_values(
    by=['Monetary', 'Frequency', 'Recency'],
    ascending=[True, True, False]
).reset_index(drop=True)

grade_names = [
    "Lv1.이탈관리",
    "Lv2.관심필요",
    "Lv3.신규성장",
    "Lv4.우수충성",
    "Lv5.최우수VIP"
]

cluster_to_grade = {}
for i, row in centers.iterrows():
    cluster_to_grade[row['Cluster']] = grade_names[i]

rfm['Customer_Grade'] = rfm['Cluster'].map(cluster_to_grade)

print("\n[Cluster -> Grade Mapping]")
print(cluster_to_grade)
print("\n[고객 등급별 수]")
print(rfm['Customer_Grade'].value_counts())

# 5-9. 고객 군집 scatter
rfm_plot = rfm.copy()
rfm_plot['log_Frequency'] = np.log1p(rfm_plot['Frequency'])
rfm_plot['log_Monetary'] = np.log1p(rfm_plot['Monetary'])

centers_plot = centers.copy()
centers_plot['log_Frequency'] = np.log1p(centers_plot['Frequency'])
centers_plot['log_Monetary'] = np.log1p(centers_plot['Monetary'])

plt.figure(figsize=(10, 7))

for grade in grade_names:
    subset = rfm_plot[rfm_plot['Customer_Grade'] == grade]
    plt.scatter(
        subset['log_Frequency'],
        subset['log_Monetary'],
        label=grade,
        alpha=0.5,
        s=20
    )

plt.scatter(
    centers_plot['log_Frequency'],
    centers_plot['log_Monetary'],
    c='black',
    marker='X',
    s=250,
    label='Centers'
)

for _, row in centers_plot.iterrows():
    plt.text(
        row['log_Frequency'] + 0.02,
        row['log_Monetary'] + 0.02,
        cluster_to_grade[row['Cluster']]
    )

plt.xlabel("Frequency (log)")
plt.ylabel("Monetary (log)")
plt.title("고객 등급별 세분화 결과")
plt.legend()
plt.show()

# =========================================================
# 6. 상품 분석 (ABC)
# =========================================================
product_sales = df_filtered.groupby("StockCode").agg(
    Quantity=("Quantity", "sum"),
    Revenue=("Revenue", "sum")
).reset_index()

product_sales = product_sales.sort_values(by="Revenue", ascending=False)

product_sales["CumulativeRevenue"] = product_sales["Revenue"].cumsum()
product_sales["CumulativeRatio"] = product_sales["CumulativeRevenue"] / product_sales["Revenue"].sum()

def abc_class(x):
    if x <= 0.80:
        return "A"
    elif x <= 0.95:
        return "B"
    else:
        return "C"

product_sales["ABC_Class"] = product_sales["CumulativeRatio"].apply(abc_class)

print("상위 매출 상품:\n", product_sales.head(10))
print("A/B/C 상품 개수:\n", product_sales["ABC_Class"].value_counts())

# 6-1. ABC 매출 비중 그래프
abc_revenue_share = product_sales.groupby('ABC_Class')['Revenue'].sum()
abc_revenue_share = abc_revenue_share / abc_revenue_share.sum()
abc_revenue_share = abc_revenue_share.reindex(['A', 'B', 'C'])

plt.figure(figsize=(6, 4))
abc_revenue_share.plot(kind='bar')
plt.title("ABC Class Revenue Share")
plt.xlabel("ABC Class")
plt.ylabel("Revenue Share")
plt.show()

# 6-2. Long Tail 그래프
product_rank = product_sales.sort_values(by='Revenue', ascending=False).reset_index(drop=True)

plt.figure(figsize=(10, 6))
plt.plot(product_rank.index, product_rank['Revenue'])
plt.yscale('log')
plt.title("Product Long Tail Distribution")
plt.xlabel("Product Rank")
plt.ylabel("Revenue (log scale)")
plt.show()

# =========================================================
# 7. 고객 × 상품 통합 테이블
# =========================================================
customer_product = df_filtered.groupby(["CustomerID", "StockCode"]).agg(
    Quantity=("Quantity", "sum"),
    Revenue=("Revenue", "sum")
).reset_index()

customer_product = customer_product.merge(
    rfm[["CustomerID", "Recency", "Frequency", "Monetary", "Customer_Grade"]],
    on="CustomerID",
    how="left"
)

customer_product = customer_product.merge(
    product_sales[["StockCode", "ABC_Class"]],
    on="StockCode",
    how="left"
)

print("통합 후 customer_product:\n", customer_product.head())

# =========================================================
# 8. 결과 확인
# =========================================================
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
print("\n[매출 분포]")
print(df["Revenue"].describe())

print("\n[수량/가격 분포]")
print(df[["Quantity", "UnitPrice"]].describe())

monthly_sales = df.groupby('YearMonth')['Revenue'].sum()

plt.figure(figsize=(10, 5))
monthly_sales.plot(marker='o')
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
# 10. VIP 및 규칙 기반 세그먼트 분석
# =========================================================

# 10-1. VIP 고객 정의
vip_threshold = rfm["Monetary"].quantile(0.75)
vip_customers = rfm[rfm["Monetary"] >= vip_threshold]

print("VIP 고객 수:", len(vip_customers))
print("VIP 고객 비율:", len(vip_customers) / len(rfm))

# 10-2. VIP 매출 비율
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

# 10-4. 규칙 기반 세그먼트
def segment_customer(row):
    if int(row["R_score"]) >= 4 and int(row["F_score"]) >= 4 and int(row["M_score"]) >= 4:
        return "VIP"
    elif int(row["R_score"]) <= 2 and int(row["F_score"]) <= 2:
        return "At-risk"
    elif int(row["F_score"]) >= 4:
        return "Loyal"
    else:
        return "General"

rfm["Segment2"] = rfm.apply(segment_customer, axis=1)
print(rfm["Segment2"].value_counts())

# =========================================================
# 11. 고객 × 상품 통합 분석
# =========================================================

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

# 11-2. 고객 세그먼트별 상품 소비 패턴
rfm["Segment"] = "Low"
rfm.loc[rfm["Monetary"] >= rfm["Monetary"].quantile(0.75), "Segment"] = "VIP"
rfm.loc[
    (rfm["Monetary"] >= rfm["Monetary"].quantile(0.50)) &
    (rfm["Monetary"] < rfm["Monetary"].quantile(0.75)),
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

# =========================================================
# 12. 세그먼트(머신러닝 등급) 시각화용 요약
# =========================================================

# 주문 단위 요약
order_summary = df_filtered.groupby(['CustomerID', 'InvoiceNo'], as_index=False).agg(
    OrderRevenue=('Revenue', 'sum'),
    YearMonth=('YearMonth', 'max')
)

order_summary = order_summary.merge(
    rfm[['CustomerID', 'Customer_Grade']],
    on='CustomerID',
    how='left'
)

segment_summary = order_summary.groupby('Customer_Grade').agg(
    Customers=('CustomerID', 'nunique'),
    Orders=('InvoiceNo', 'nunique'),
    Revenue=('OrderRevenue', 'sum')
).reset_index()

segment_summary['AOV'] = segment_summary['Revenue'] / segment_summary['Orders']
segment_summary['Revenue_per_Customer'] = segment_summary['Revenue'] / segment_summary['Customers']
segment_summary['Customer_Share'] = segment_summary['Customers'] / segment_summary['Customers'].sum()
segment_summary['Revenue_Share'] = segment_summary['Revenue'] / segment_summary['Revenue'].sum()

segment_summary['GradeOrder'] = segment_summary['Customer_Grade'].str.extract(r'Lv(\d)').astype(int)
segment_summary = segment_summary.sort_values('GradeOrder')

print(segment_summary)

# 12-1. 세그먼트별 객단가(AOV)
plt.figure(figsize=(8, 5))
plt.bar(segment_summary['Customer_Grade'], segment_summary['AOV'])
plt.title("세그먼트별 객단가(AOV)")
plt.xlabel("Customer Grade")
plt.ylabel("AOV")
plt.show()

# 12-2. 세그먼트별 고객당 매출
plt.figure(figsize=(8, 5))
plt.bar(segment_summary['Customer_Grade'], segment_summary['Revenue_per_Customer'])
plt.title("세그먼트별 고객당 매출")
plt.xlabel("Customer Grade")
plt.ylabel("Revenue per Customer")
plt.show()

# 12-3. 세그먼트별 고객 비중 vs 매출 비중
compare_df = segment_summary[['Customer_Grade', 'Customer_Share', 'Revenue_Share']].copy()
compare_df = compare_df.set_index('Customer_Grade')
compare_df.plot(kind='bar', figsize=(9, 5))
plt.title("세그먼트별 고객 비중 vs 매출 비중")
plt.ylabel("Share")
plt.xticks(rotation=0)
plt.show()

# 12-4. 세그먼트별 상품 ABC 매출 구성비
if 'Customer_Grade' not in customer_product.columns:
    customer_product = customer_product.merge(
        rfm[['CustomerID', 'Customer_Grade']],
        on='CustomerID',
        how='left'
    )

segment_abc_grade = customer_product.groupby(['Customer_Grade', 'ABC_Class'])['Revenue'].sum().reset_index()

segment_abc_grade['GradeOrder'] = segment_abc_grade['Customer_Grade'].str.extract(r'Lv(\d)').astype(int)
segment_abc_grade = segment_abc_grade.sort_values(['GradeOrder', 'ABC_Class'])

segment_abc_pivot = segment_abc_grade.pivot(index='Customer_Grade', columns='ABC_Class', values='Revenue').fillna(0)
segment_abc_pivot = segment_abc_pivot[['A', 'B', 'C']]
segment_abc_ratio = segment_abc_pivot.div(segment_abc_pivot.sum(axis=1), axis=0)

segment_abc_ratio.plot(kind='bar', stacked=True, figsize=(9, 5))
plt.title("세그먼트별 상품 ABC 매출 구성비")
plt.xlabel("Customer Grade")
plt.ylabel("Revenue Share within Segment")
plt.xticks(rotation=0)
plt.show()

# 12-5. 세그먼트별 월별 매출 추이
monthly_grade_sales = df_filtered.merge(
    rfm[['CustomerID', 'Customer_Grade']],
    on='CustomerID',
    how='left'
)

monthly_grade_sales = monthly_grade_sales.groupby(['YearMonth', 'Customer_Grade'])['Revenue'].sum().reset_index()

grade_month_pivot = monthly_grade_sales.pivot(index='YearMonth', columns='Customer_Grade', values='Revenue').fillna(0)
ordered_cols = [g for g in grade_names if g in grade_month_pivot.columns]
grade_month_pivot = grade_month_pivot[ordered_cols]

plt.figure(figsize=(10, 6))
for col in grade_month_pivot.columns:
    plt.plot(grade_month_pivot.index, grade_month_pivot[col], marker='o', label=col)

plt.title("세그먼트별 월별 매출 추이")
plt.xlabel("Year-Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.legend()
plt.show()
