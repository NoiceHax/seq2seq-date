from datetime import datetime, timedelta
import random
import pandas as pd

FORMATS = [
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%Y/%m/%d",
    "%d %B, %Y",
    "%B %d %Y"
]

def random_date():
    start = datetime(1900, 1, 1)
    end = datetime(2100, 12, 31)

    days = random.randint(0, (end - start).days)
    return start + timedelta(days=days)

data = []

for _ in range(100000):
    date = random_date()

    input_format = random.choice(FORMATS)

    input_text = date.strftime(input_format)
    target_text = date.strftime("%Y-%m-%d")

    data.append([input_text, target_text])

df = pd.DataFrame(data, columns=["input", "target"])

df.to_csv("./data/dataset.csv", index=False)

print("Dataset generated!")