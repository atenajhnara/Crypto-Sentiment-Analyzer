# 📊 Crypto Reddit Sentiment Analyzer | تحلیل احساسات کریپتو در Reddit

A Python project that fetches Reddit posts related to cryptocurrencies, cleans the text, and performs sentiment analysis using VADER and multilingual BERT.  
It generates CSV files with cleaned text and sentiment labels for further analysis.

پروژه پایتون برای جمع‌آوری پست‌های Reddit در حوزه ارز دیجیتال، پاکسازی متن‌ها و تحلیل احساسات با استفاده از VADER و مدل چندزبانه BERT.  
خروجی شامل فایل‌های CSV با متن پاک‌شده و برچسب‌های احساسات است.

---

## 🧠 Technologies Used | تکنولوژی‌های استفاده‌شده

- Python 3.10+  
- PRAW (Python Reddit API Wrapper)  
- pandas / NumPy  
- VADER Sentiment Analyzer  
- transformers (BERT multilingual)  
- dotenv (مدیریت کلیدهای محیطی)  
- re (Regular Expressions برای پاکسازی متن)

---

## ⚙️ How It Works | نحوه کار

1. Fetch Reddit posts  
   Collect posts from multiple subreddits (e.g., Bitcoin, CryptoCurrency).  
   جمع‌آوری پست‌ها از چند subreddit مرتبط با کریپتو.

2. Clean text  
   Remove links, mentions, hashtags, special characters and normalize text.  
   حذف لینک‌ها، منشن‌ها، هشتگ‌ها و علامت‌های اضافی، و استانداردسازی متن.

3. Sentiment Analysis (VADER)  
   Calculate sentiment polarity and label as positive, negative, or neutral.  
   تحلیل احساسات با VADER و تعیین برچسب مثبت، منفی یا خنثی.

4. Sentiment Analysis (BERT multilingual)  
   Use pre-trained multilingual BERT model for more advanced sentiment scoring.  
   استفاده از مدل BERT چندزبانه برای تحلیل احساسات دقیق‌تر.

5. Simplify BERT output  
   Convert star ratings to positive/neutral/negative labels.  
   تبدیل خروجی ستاره‌ای مدل BERT به برچسب‌های ساده مثبت/منفی/خنثی.

6. Save to CSV  
   Final results are stored in CSV for further analysis.  
   ذخیره نتایج نهایی در فایل CSV برای تحلیل‌های بعدی.

---

## 🧩 Key Code Structure | ساختار اصلی کد

```python
# Connect to Reddit
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# Collect posts from subreddits
subreddits = ["Bitcoin", "CryptoCurrency"]
all_posts = []
for subreddit_name in subreddits:
    subreddit = reddit.subreddit(subreddit_name)
    for post in subreddit.new(limit=200):
        all_posts.append({
            "title": post.title,
            "selftext": post.selftext,
            "score": post.score,
            "comments": post.num_comments
        })

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|@\S+|#\S+|[^\w\s]", "", text)
    return text.strip()

df["text"] = (df["title"].fillna("") + " " + df["selftext"].fillna("")).apply(clean_text)

# Sentiment Analysis with VADER
analyzer = SentimentIntensityAnalyzer()
df["sentiment"] = df["text"].apply(
    lambda x: "positive" if analyzer.polarity_scores(x)["compound"]>=0.05
              else ("negative" if analyzer.polarity_scores(x)["compound"]<=-0.05 else "neutral")
)

# Sentiment Analysis with multilingual BERT
sentiment_pipeline = pipeline("sentiment-analysis",
                              model="nlptown/bert-base-multilingual-uncased-sentiment")
df["sentiment_ai_simple"] = df["text"].apply(lambda x: stars_to_sentiment(sentiment_pipeline(x[:512])[0]['label']))
