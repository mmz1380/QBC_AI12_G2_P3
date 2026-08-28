"""Grounded Persian prompts (zero-hallucination contract) + evidence formatters.

The system prompt forces the model to answer only from the supplied evidence and to
tag every claim with a machine-checkable citation — [محصول <id>] for a product,
[بازبینی <id>] for a review — which `assistant.verify_citations` then audits.
"""
from __future__ import annotations

from ..core.persian_text import format_toman

SYSTEM_CORE = (
    "تو یک دستیار خرید دیجی‌کالا هستی که فقط بر پایهٔ داده‌های واقعی پاسخ می‌دهی.\n"
    "۱) فقط از «مدارک» استفاده کن؛ دانش قبلی یا عددسازی ممنوع.\n"
    "۲) هر ادعا باید با [محصول شناسه] یا [بازبینی شناسه] ارجاع داده شود.\n"
    "۳) اگر مدارک کافی نیست، صریحاً بنویس «اطلاعات کافی موجود نیست».\n"
    "۴) پاسخ فارسی، کوتاه و ساختاریافته باشد.\n"
    "۵) عددها را عیناً از مدارک کپی کن.\n"
)
DISCOVERY_SYSTEM = SYSTEM_CORE + "\nوظیفه: پیشنهاد و رتبه‌بندی محصول. حداکثر {k} پیشنهاد، هرکدام با [محصول ...] و دلیل کوتاه."
QA_SYSTEM = SYSTEM_CORE + "\nوظیفه: پاسخ به پرسش دربارهٔ یک محصول، فقط از بازبینی‌ها. هر ادعا با [بازبینی ...]. حداکثر {max_lines} خط."
COMPARISON_SYSTEM = SYSTEM_CORE + "\nوظیفه: مقایسهٔ محصولات از جدول واقعیت و بازبینی‌ها؛ نبودِ داده را در «اطلاعات موجود نیست» بگو."
MANAGERIAL_SYSTEM = SYSTEM_CORE + "\nوظیفه: تحلیل مدیریتی (شکایت‌ها، محصولات با نرخ توصیهٔ پایین، رضایت برندها). حداکثر {max_paragraphs} پاراگراف."

# LLM-as-judge prompts (Phase 4)
JUDGE_FAITH_SYS = ("تو یک ارزیاب هستی. «پایبندی به منبع» را بسنج: آیا هر ادعای پاسخ در منابع "
                   "ارجاع‌شده پشتیبانی می‌شود؟ فقط یک عدد ۰ تا ۵ بنویس.")
JUDGE_REL_SYS = ("تو یک ارزیاب هستی. «مفید و مرتبط بودن پاسخ» را بسنج: آیا مستقیم و مرتبط به "
                 "سؤال جواب داده؟ فقط یک عدد ۰ تا ۵ بنویس.")
JUDGE_USER = "سؤال: {query}\n\nپاسخ سیستم:\n{answer}\n\nمنابع:\n{sources}\n\nنمره (۰ تا ۵):"


def evidence_products(rows) -> str:
    out = []
    for r in rows:
        out.append(f"[محصول {r['product_id']}] {r['title']} | برند: {r['brand'] or 'نامشخص'} | "
                   f"قیمت: {format_toman(r['price'])} تومان | امتیاز: {r['rate'] if r['rate'] is not None else 'نامشخص'} | "
                   f"تعداد نظر: {r['comment_count']}")
    return "\n".join(out)


def evidence_reviews(rows) -> str:
    m = {"recommended": "توصیه", "not_recommended": "عدم توصیه", "no_idea": "نظری ندارد"}
    out = []
    for r in rows:
        out.append(f"[بازبینی {r['comment_id']}] (محصول {r['product_id']}) "
                   f"امتیاز {r['rate'] if r['rate'] is not None else '?'} | "
                   f"{m.get(r['recommendation_status'], 'نامشخص')} | پسند {r['likes']}\nمتن: {r['text']}")
    return "\n".join(out)
