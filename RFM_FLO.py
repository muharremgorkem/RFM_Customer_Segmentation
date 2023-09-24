###########################################################
# FLO Customer Segmentation using the RFM Method
##############################################################

##############################################################
# 1. Business Problem
##############################################################
# FLO, an online shoe store, wants to segment its customers and determine
# marketing strategies based on these segments.

# Dataset story
# The dataset consists of information from customers who made their last purchases
# as OmniChannel (both online and offline) between the years 2020 and 2021.
#
# Variables
# master_id -- Unique customer number
# order_channel -- The channel used for shopping (Android, iOS, Desktop, Mobile)
# last_order_channel -- The channel used for the last purchase
# first_order_date -- The date of the customer's first purchase
# last_order_date -- The date of the customer's last purchase
# last_order_date_online -- The date of the customer's last online purchase
# last_order_date_offline -- The date of the customer's last offline purchase
# order_num_total_ever_online -- Total number of purchases made by the customer online
# order_num_total_ever_offline -- Total number of purchases made by the customer offline
# customer_value_total_ever_offline -- Total amount paid by the customer for offline purchases
# customer_value_total_ever_online -- Total amount paid by the customer for online purchases

###############################################################
# 2. Data Preparation
###############################################################

# Importing libraries
##############################################
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 200)

df_ = pd.read_csv('Datasets/flo_data_20k.csv')
df = df_.copy()

# Data Understanding
#############################
def check_df(dataframe, head=5):
    print('################# Columns ################# ')
    print(dataframe.columns)
    print('################# Types  ################# ')
    print(dataframe.dtypes)
    print('##################  Head ################# ')
    print(dataframe.head(head))
    print('#################  Shape ################# ')
    print(dataframe.shape)
    print('#################  NA ################# ')
    print(dataframe.isnull().sum())
    print('#################  Quantiles ################# ')
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99]).T)
check_df(df)


# Creating new variables for each customer's total number of purchases and total spending
##########################################################################################
df['order_num_total'] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']
df['customer_value_total'] = df['customer_value_total_ever_offline'] + df['customer_value_total_ever_online']
df.head()


# Converting the variables representing dates to the 'date' data type
#####################################################################
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.info()

# Analyzing the distribution of the number of customers in each shopping channel,
# the total number of products purchased, and the total spending
#################################################################################
df.groupby("order_channel").agg({"order_num_total": "count", "customer_value_total": "mean"})

#Graphs:
fig, ax = plt.subplots(nrows=1, ncols=3)
sns.countplot(df,
              y="order_channel",
              ax=ax[0],
              order=df["order_channel"].value_counts(ascending=False).index)
ax[0].set_xlabel("Total Number of Customer")
ax[0].set_ylabel("Order Chanel")
ax[0].set_title("Customer Distribution by Order Chanel")

sns.kdeplot(df,
             x="order_num_total",
             ax=ax[1],
             hue="order_channel")
ax[1].get_legend().set_title("Order Chanel")
ax[1].set_xlabel("Total Number of Purchase")
ax[1].set_ylabel("Total Number of Customer")
ax[1].set_title("Customer Distribution by Order Chanel")
ax[1].set_xlim([0, 75])

sns.kdeplot(df,
             x="customer_value_total",
             ax=ax[2],
             hue="order_channel")
ax[2].get_legend().set_title("Order Channel")
ax[2].set_xlabel("Total Number of Purchases")
ax[2].set_ylabel("Total Number of Customer")
ax[2].set_title("Total Number of Purchase by Order Chanel")
ax[2].set_xlim([0, 15000])

fig.set_size_inches(24, 7)
plt.show(block=True)

# Ranking the top 10 customers who has the highest revenue
###################################################################
top10_value = df.sort_values(by='customer_value_total', ascending=False).head(10)
top10_value[['master_id', 'customer_value_total']]

# Ranking the top 10 customers who have placed the most orders
##############################################################
top10_order = df.sort_values(by='order_num_total', ascending=False).head(10)
top10_order[['master_id', 'order_num_total']]

# # The function for data preparation - includes all the steps above
def data_prep(dataframe):
    df['order_num_total'] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']
    df['customer_value_total'] = df['customer_value_total_ever_offline'] + df['customer_value_total_ever_online']
    date_columns = df.columns[df.columns.str.contains("date")]
    df[date_columns] = df[date_columns].apply(pd.to_datetime)
    return dataframe
data_prep(df)


###############################################################
# 3. Calculation of RFM Metrics
###############################################################

# Determining the analysis date as 2 days after the last order date
###################################################################
df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)
print(today_date)

# Assigning the calculated metrics to a new variable called "RFM"
#################################################################
rfm = df.groupby('master_id').agg({'last_order_date': lambda last_order_date: (today_date - last_order_date.max()).days,
                                   'order_num_total': lambda order_num_total: order_num_total,
                                   'customer_value_total': lambda customer_value_total: customer_value_total})
rfm.head()

# Changing the variable names to "Recency," "Frequency," and "Monetary"
#######################################################################
rfm.columns = ['recency', 'frequency', 'monetary']
rfm.head()

###############################################################
# 4. Calculation of RF Score
###############################################################

# Converting the Recency, Frequency, and Monetary metrics into scores ranging from 1 to 5
##########################################################################################
rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]) # (method='first') --> order-based ranking
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

# Assigning "recency_score" and "frequency_score" as a single variable named "RF_SCORE"
#######################################################################################
rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))

###############################################################
# 5. Defining RF Score as Segments
###############################################################

# Defining segments for the created RF scores
##############################################
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'}

rfm_segment = rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

# Examining the mean values of recency, frequency, and monetary values for each segment
#######################################################################################
means = rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean"])

# Case 1:
# FLO is investing in a new women's shoe brand. Products are priced above the general customer
# preferences. Therefore, they want to reach out to specific customer profiles for the promotion
# and sales of this new brand. They aim to contact loyal customers (champions, loyal_customers)
# and those who have made purchases in the women's category. The goal is to identify the customer IDs
# of these specific customers and save them to a CSV file.
#######################################################################################################
rfm_segment = rfm_segment.reset_index()
rfm_segment = rfm_segment.rename(columns={"master_id":"customer_id"})
df = pd.merge(df, rfm_segment, left_on= "master_id", right_on="customer_id", how="left")
df.drop(columns="customer_id", inplace=True)

flo_kadin = df.loc[(df["RF_SCORE"].isin(["champions", 'loyal_customers'])) &
                       (df["interested_in_categories_12"].str.contains("KADIN"))] # KADIN --> woman

flo_woman_ids = flo_kadin[["master_id"]]# --> 2.487 customer ids
flo_woman_ids.to_csv("woman_ids.csv", index=False)

# Case 2:
# A discount of nearly 40% is planned for men's and children's products.
# Customers who are interested in these discounted categories and have been good customers
# in the past but haven't made purchases for a long time, newly acquired customers,
# and those who are considered "at risk" of being lost should be specifically targeted.
# The goal is to save the IDs of the eligible customers into a CSV file for targeted marketing.
################################################################################################

flo_male_child = df.loc[(df["RF_SCORE"].isin(["Hibernating", 'New Customers'])) \
                         & (df["interested_in_categories_12"].str.contains("ERKEK")) \
                            | (df["interested_in_categories_12"].str.contains("COCUK"))]
# COCUK -> children, ERKEK -> man

flo_male_child_ids = flo_male_child[["master_id"]]
flo_male_child_ids.to_csv("mc_ids.csv", index=False)
