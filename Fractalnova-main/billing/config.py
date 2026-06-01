from __future__ import annotations

import os


class BillingSettings:
    stripe_secret_key: str = os.getenv("STRIPE_SECRET_KEY", "")
    stripe_webhook_secret: str = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    success_url: str = os.getenv("BILLING_SUCCESS_URL", "https://fractalnova.app/billing/success")
    cancel_url: str = os.getenv("BILLING_CANCEL_URL", "https://fractalnova.app/billing/cancel")
    trial_days: int = int(os.getenv("BILLING_TRIAL_DAYS", "7"))


settings = BillingSettings()
