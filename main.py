#جمع آوری پست و پاکسازی متن ها و تحلیل احساسات

import praw
import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from dotenv import load_dotenv
import os


# مقادیر خودت رو جایگزین کن
load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# انتخاب subredditها
subreddits = ["Bitcoin", "CryptoCurrency"]

# تعداد پست‌ها برای هر subreddit
post_limit = 200

all_posts = []

for subreddit_name in subreddits:
    subreddit = reddit.subreddit(subreddit_name)
    for post in subreddit.new(limit=post_limit):
        all_posts.append({
            "subreddit": subreddit_name,
            "title": post.title,  #عنوان پست
            "selftext": post.selftext,  #متن پست
            "score": post.score,  #امتیاز پست
            "comments": post.num_comments,  #تعداد کامنت ها
            "url": post.url
        })

# ذخیره در DataFrame
df = pd.DataFrame(all_posts)
print(df.head())

# ذخیره در CSV
df.to_csv("reddit_crypto_posts.csv", index=False, encoding="utf-8-sig")

#print(len(df))  # تعداد کل ردیف‌ها
#print(df["subreddit"].value_counts())  # تعداد پست‌ها برای هر subreddit


# فایل CSV که قبلا ذخیره کردی
df = pd.read_csv("reddit_crypto_posts.csv")

# تابع پاک‌سازی متن
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()  # تبدیل به حروف کوچک
    text = re.sub(r"http\S+", "", text)  # حذف لینک‌ها
    text = re.sub(r"www\S+", "", text)   # حذف لینک‌های دیگر
    text = re.sub(r"@\S+", "", text)     # حذف منشن‌ها
    text = re.sub(r"#\S+", "", text)     # حذف هشتگ‌ها
    text = re.sub(r"[^\w\s]", "", text)   # حذف علامت‌های اضافی
    text = re.sub(r"\s+", " ", text).strip()  # حذف فاصله‌های اضافی
    return text

# ترکیب title و selftext و پاک‌سازی
df["text"] = (df["title"].fillna("") + " " + df["selftext"].fillna("")).apply(clean_text)

# فقط ستون متن و زبان (فارسی یا انگلیسی) رو نگه داریم
df_clean = df[["text"]]

# نمایش نمونه
print(df_clean.head())

# ذخیره نهایی برای تحلیل
df_clean.to_csv("reddit_crypto_posts_clean.csv", index=False, encoding="utf-8-sig")



# خواندن داده پاک‌سازی‌شده
df = pd.read_csv("reddit_crypto_posts_clean.csv")

analyzer = SentimentIntensityAnalyzer()

# تابع تشخیص احساسات
def get_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

# اعمال روی ستون متن
df["sentiment"] = df["text"].apply(get_sentiment)

# نمایش نمونه
print(df.head())

# ذخیره نهایی
df.to_csv("reddit_crypto_posts_sentiment.csv", index=False, encoding="utf-8-sig")


# خواندن داده پاک‌سازی‌شده
df = pd.read_csv("reddit_crypto_posts_clean.csv")

# مدل تحلیل احساسات چندزبانه
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

# اعمال روی ستون متن
def get_sentiment_ai(text):
    result = sentiment_pipeline(text[:512])[0]  # خروجی مدل
    return result['label']

df["sentiment_ai"] = df["text"].apply(get_sentiment_ai)

# ---------- اینجا کد تبدیل ستاره به positive/negative/neutral ----------
def stars_to_sentiment(stars_label):
    stars = int(stars_label.split()[0])  # گرفتن عدد اول از "2 stars"
    if stars <= 2:
        return "negative"
    elif stars == 3:
        return "neutral"
    else:
        return "positive"

df["sentiment_ai_simple"] = df["sentiment_ai"].apply(stars_to_sentiment)
# ---------------------------------------------------------------------------

# ذخیره نهایی
df.to_csv("reddit_crypto_posts_sentiment_ai_final.csv", index=False, encoding="utf-8-sig")


#ترنسفورمرز مدل و رابطش رو میاره
#ترنسفورمرز رابط راحت برای استفاده از مدل‌های NLP آماده و اجرای آن


# ترچ عملیات پشت صحنه رو اجرا می‌کنه و موتور محاسباتی و اجرای شبکه عصبی



