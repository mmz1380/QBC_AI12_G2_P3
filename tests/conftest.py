"""Shared fixtures. Adds src/ to the path and builds a tiny synthetic catalog so
router/aggregate tests run in milliseconds with no models or data files."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def products():
    return pd.DataFrame({
        "product_id": [100101, 100102, 100103],   # 6-9 digits, like real Digikala ids
        "title_fa": ["کیف چرم زنانه", "گوشی موبایل سامسونگ", "کفش ورزشی مردانه"],
        "title_norm": ["کیف چرم زنانه", "گوشی موبایل سامسونگ", "کفش ورزشی مردانه"],
        "product_text_norm": ["کیف چرم زنانه اکسسوری چرم", "گوشی موبایل سامسونگ",
                              "کفش ورزشی مردانه کفش"],
        "brand_norm": ["چرم", "سامسونگ", "نایک"],
        "category1_norm": ["اکسسوری زنانه", "موبایل", "کفش"],
        "category2_norm": ["کیف", "گوشی", "کفش ورزشی"],
        "sub_category_norm": ["کیف زنانه", "موبایل", "کفش"],
        "price_clean": [2_000_000, 90_000_000, 5_000_000],
        "product_rate_clean": [88, 92, 80],
        "rate_count": [10, 500, 30],
        "is_fake": [False, False, False],
    })


@pytest.fixture
def comments():
    return pd.DataFrame({
        "comment_id": [1, 2, 3, 4],
        "product_id": [100101, 100101, 100102, 100102],
        "comment_text_norm": ["کیف با کیفیت و زیبا", "بند کیف پاره شد افتضاح",
                              "گوشی عالی و سریع", "باتری ضعیف است"],
        "body_norm": ["کیف با کیفیت", "بند پاره شد", "گوشی عالی", "باتری ضعیف"],
        "advantages_norm": ["زیبا", "", "سرعت", ""],
        "disadvantages_norm": ["", "بند پاره", "", "باتری"],
        "recommendation_status": ["recommended", "not_recommended", "recommended", "no_idea"],
        "rate_clean": [5, 1, 5, 3],
        "likes": [10, 2, 40, 1],
        "is_buyer": [True, True, True, False],
        "has_text": [True, True, True, True],
    })


@pytest.fixture
def catalog(products, comments):
    from digikala.phase2_assistant.retrieval import GroupedComments
    from digikala.phase2_assistant.router import Catalog
    gc = GroupedComments(comments)
    return Catalog.build(products, gc)
