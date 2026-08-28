from digikala.core import persian_text as pt


def test_normalize_folds_digits_and_arabic():
    assert pt.normalize("۱۲۳ كيفيت") == "123 کیفیت"


def test_price_constraint_max_toman_to_rial():
    # "under 500 thousand Toman" -> 500,000 Toman = 5,000,000 Rials
    assert pt.extract_price_constraint("زیر ۵۰۰ هزار تومان", 10) == {"price_max": 5_000_000}


def test_price_constraint_min():
    assert pt.extract_price_constraint("بالای ۲ میلیون", 10) == {"price_min": 20_000_000}


def test_bare_number_is_not_a_constraint():
    assert pt.extract_price_constraint("یک کیف ۵۰۰ عددی", 10) == {}


def test_format_toman_rials_to_toman():
    assert pt.format_toman(5_000_000) == "500,000"
    assert pt.format_toman(None) == "نامشخص"


def test_tokenize_norm_skips_empty():
    assert pt.tokenize_norm("") == []
    assert pt.tokenize_norm("کیف چرم") == ["کیف", "چرم"]
