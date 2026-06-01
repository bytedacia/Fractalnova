from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class SubscriptionStatus(str, Enum):
    trialing = "trialing"
    active = "active"
    past_due = "past_due"
    canceled = "canceled"
    incomplete = "incomplete"
    incomplete_expired = "incomplete_expired"


@dataclass
class Price:
    stripe_price_id: str
    currency: str = "usd"
    unit_amount: int = 0
    interval: str = "month"
    id: Optional[str] = None

    @property
    def display_amount(self) -> str:
        return f"${self.unit_amount / 100:.2f}"


@dataclass
class Plan:
    id: str
    name: str
    description: str
    stripe_price_id: str
    price: Price
    features: List[str] = field(default_factory=list)
    max_books_per_month: int = 5
    max_pages_per_book: int = 200
    max_export_formats: int = 2
    priority_generation: bool = False
    api_access: bool = False
    custom_model: bool = False

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "price": {
                "stripe_price_id": self.price.stripe_price_id,
                "currency": self.price.currency,
                "unit_amount": self.price.unit_amount,
                "display": self.price.display_amount,
                "interval": self.price.interval,
            },
            "features": self.features,
            "limits": {
                "max_books_per_month": self.max_books_per_month,
                "max_pages_per_book": self.max_pages_per_book,
                "max_export_formats": self.max_export_formats,
                "priority_generation": self.priority_generation,
                "api_access": self.api_access,
                "custom_model": self.custom_model,
            },
        }


PLANS: List[Plan] = [
    Plan(
        id="free",
        name="Free",
        description="Per iniziare a scrivere con l'IA",
        stripe_price_id="",
        price=Price(stripe_price_id="", unit_amount=0, interval="month"),
        features=["5 libri/mese", "Fino a 100 pagine", "Export EPUB + TXT"],
        max_books_per_month=5,
        max_pages_per_book=100,
        max_export_formats=2,
    ),
    Plan(
        id="starter",
        name="Starter",
        description="Per scrittori indipendenti",
        stripe_price_id="price_starter_monthly",
        price=Price(stripe_price_id="price_starter_monthly", unit_amount=999, interval="month"),
        features=["50 libri/mese", "Fino a 500 pagine", "Export EPUB + PDF + Word", "Generazione prioritaria"],
        max_books_per_month=50,
        max_pages_per_book=500,
        max_export_formats=3,
        priority_generation=True,
    ),
    Plan(
        id="pro",
        name="Professional",
        description="Per autori professionisti e piccoli editori",
        stripe_price_id="price_pro_monthly",
        price=Price(stripe_price_id="price_pro_monthly", unit_amount=2999, interval="month"),
        features=["Libri illimitati", "Fino a 2000 pagine", "Tutti i formati di export", "API access", "Copertina con IA"],
        max_books_per_month=999,
        max_pages_per_book=2000,
        max_export_formats=5,
        priority_generation=True,
        api_access=True,
        custom_model=False,
    ),
    Plan(
        id="enterprise",
        name="Enterprise",
        description="Per case editrici e piattaforme",
        stripe_price_id="price_enterprise_monthly",
        price=Price(stripe_price_id="price_enterprise_monthly", unit_amount=9999, interval="month"),
        features=["Tutto illimitato", "Modello custom fine-tuned", "SSO/SAML", "SLA 99.99%", "Supporto dedicato"],
        max_books_per_month=99999,
        max_pages_per_book=99999,
        max_export_formats=10,
        priority_generation=True,
        api_access=True,
        custom_model=True,
    ),
]
