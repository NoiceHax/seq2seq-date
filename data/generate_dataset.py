from datetime import date
import random
import calendar
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
    "%B %d %Y",
    "%b %d %Y",
    "%d/%m/%Y"
]


def random_case(text):

    mode = random.choice([
        "original",
        "lower",
        "upper"
    ])

    if mode == "lower":
        return text.lower()

    if mode == "upper":
        return text.upper()

    return text


def random_date():

    year = random.randint(1, 3000)

    month = random.randint(1, 12)

    max_day = calendar.monthrange(
        year,
        month
    )[1]

    day = random.randint(
        1,
        max_day
    )

    return date(
        year,
        month,
        day
    )


data = []

for _ in range(300000):

    current_date = random_date()

    input_format = random.choice(
        FORMATS
    )

    input_text = current_date.strftime(
        input_format
    )

    input_text = random_case(
        input_text
    )

    target_text = (
        f"{current_date.year:04d}-"
        f"{current_date.month:02d}-"
        f"{current_date.day:02d}"
    )

    data.append([
        input_text,
        target_text
    ])


df = pd.DataFrame(
    data,
    columns=["input", "target"]
)

df.to_csv(
    "data/dataset.csv",
    index=False
)

print("Dataset generated!")