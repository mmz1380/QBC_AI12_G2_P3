from digikala.phase2_assistant.router import IntentRouter, extract_filters


def test_discovery_with_price_filter(catalog):
    r = IntentRouter(catalog).route("یک کیف ارزان زیر ۵۰۰ هزار تومان می‌خواهم")
    assert r.intent == "discovery"
    assert r.filters.get("price_max") == 5_000_000


def test_comparison_two_ids(catalog):
    r = IntentRouter(catalog).route("محصول 100101 و محصول 100102 را مقایسه کن")
    assert r.intent == "comparison"
    assert set(r.product_ids) == {100101, 100102}


def test_product_qa_with_id_and_cue(catalog):
    r = IntentRouter(catalog).route("آیا کیفیت محصول 100101 خوب بود؟")
    assert r.intent == "product_qa"
    assert r.product_id == 100101


def test_managerial_scope(catalog):
    r = IntentRouter(catalog).route("پرتکرارترین شکایت‌ها در دستهٔ موبایل چیست؟")
    assert r.intent == "managerial"
    assert r.scope.get("value")


def test_extract_filters_brand(catalog):
    f = extract_filters(catalog, "گوشی سامسونگ می‌خواهم")
    assert f.get("brand") == "سامسونگ"
