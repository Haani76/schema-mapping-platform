import random
import pandas as pd
import json
import os
from configs.config import config

# Column header variations per semantic type
COLUMN_VARIANTS = {
    "CUSTOMER_ID": [
        "customer_id", "cust_id", "client_id", "customer_number", "cust_no",
        "clientid", "customer_code", "cust_code", "account_id", "acct_id",
        "customer_ref", "cust_ref", "client_number", "client_code"
    ],
    "PRODUCT_ID": [
        "product_id", "prod_id", "item_id", "sku", "product_code", "prod_code",
        "item_code", "article_id", "article_code", "product_number", "prod_no",
        "item_number", "stock_id", "catalog_id"
    ],
    "REVENUE": [
        "revenue", "sales", "amount", "total", "price", "value", "income",
        "earnings", "gross_sales", "net_sales", "total_sales", "sale_amount",
        "transaction_amount", "order_value", "total_amount", "rev", "sales_amount"
    ],
    "DATE": [
        "date", "order_date", "sale_date", "transaction_date", "created_at",
        "updated_at", "purchase_date", "invoice_date", "ship_date", "delivery_date",
        "date_of_purchase", "order_dt", "sale_dt", "trans_date", "record_date"
    ],
    "QUANTITY": [
        "quantity", "qty", "units", "count", "amount", "volume", "num_items",
        "number_of_units", "units_sold", "qty_sold", "quantity_sold", "pieces",
        "stock_quantity", "order_qty", "purchase_qty"
    ],
    "LOCATION": [
        "location", "city", "state", "country", "region", "address", "zip",
        "postal_code", "territory", "area", "district", "zone", "market",
        "store_location", "warehouse", "branch"
    ],
    "EMAIL": [
        "email", "email_address", "e_mail", "customer_email", "client_email",
        "contact_email", "user_email", "mail", "email_id", "correspondence_email"
    ],
    "PHONE": [
        "phone", "phone_number", "telephone", "mobile", "cell", "contact_number",
        "phone_no", "tel", "mobile_number", "contact_phone", "fax"
    ],
    "NAME": [
        "name", "customer_name", "client_name", "full_name", "first_name",
        "last_name", "contact_name", "account_name", "person_name", "rep_name",
        "sales_rep", "agent_name"
    ],
    "STATUS": [
        "status", "order_status", "account_status", "customer_status", "state",
        "condition", "stage", "phase", "flag", "active", "is_active", "enabled"
    ],
    "CATEGORY": [
        "category", "product_category", "type", "product_type", "class",
        "segment", "group", "department", "division", "product_group",
        "item_category", "item_type", "classification"
    ],
}

# Sample values per semantic type
SAMPLE_VALUES = {
    "CUSTOMER_ID": ["C001", "CUST-1234", "10045", "CLI_9988", "A-00231"],
    "PRODUCT_ID": ["P001", "SKU-4521", "PROD_99", "ITM-001", "ART-5523"],
    "REVENUE": ["1500.00", "250.99", "10000", "99.5", "3200.75"],
    "DATE": ["2023-01-15", "01/15/2023", "Jan 15 2023", "2023/01/15", "15-01-2023"],
    "QUANTITY": ["10", "250", "1", "500", "75"],
    "LOCATION": ["New York", "California", "US", "90210", "North Region"],
    "EMAIL": ["john@example.com", "user@domain.org", "contact@company.co"],
    "PHONE": ["555-1234", "+1-800-555-0199", "9876543210", "(555) 123-4567"],
    "NAME": ["John Smith", "Jane Doe", "Acme Corp", "Bob Johnson"],
    "STATUS": ["active", "pending", "closed", "1", "true", "inactive"],
    "CATEGORY": ["Electronics", "Apparel", "Type A", "Group 1", "Segment B"],
}


def generate_training_sample(label: str, column_name: str) -> dict:
    """Generate a single training sample."""
    sample_value = random.choice(SAMPLE_VALUES[label])
    tokens = column_name.replace("_", " ").replace("-", " ").split()
    ner_tags = [f"B-{label}"] + [f"I-{label}"] * (len(tokens) - 1)

    return {
        "column_name": column_name,
        "sample_value": sample_value,
        "label": label,
        "tokens": tokens,
        "ner_tags": ner_tags,
    }


def generate_dataset(num_samples: int = 2000) -> list:
    """Generate a full training dataset."""
    dataset = []

    for label, variants in COLUMN_VARIANTS.items():
        samples_per_label = num_samples // len(COLUMN_VARIANTS)
        for _ in range(samples_per_label):
            column_name = random.choice(variants)
            sample = generate_training_sample(label, column_name)
            dataset.append(sample)

    random.shuffle(dataset)
    return dataset


def save_dataset(dataset: list, split: str = "train"):
    """Save dataset to disk."""
    os.makedirs(config.TRAINING_DATA_DIR, exist_ok=True)
    output_path = os.path.join(config.TRAINING_DATA_DIR, f"{split}.json")

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved {len(dataset)} samples to {output_path}")
    return output_path


def generate_and_save_all():
    """Generate train, validation, and test splits."""
    full_dataset = generate_dataset(num_samples=3000)

    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))

    train_data = full_dataset[:train_size]
    val_data = full_dataset[train_size:train_size + val_size]
    test_data = full_dataset[train_size + val_size:]

    save_dataset(train_data, "train")
    save_dataset(val_data, "val")
    save_dataset(test_data, "test")

    print(f"\nDataset Summary:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")
    print(f"  Test:  {len(test_data)} samples")


if __name__ == "__main__":
    generate_and_save_all()