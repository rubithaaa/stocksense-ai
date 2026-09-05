TRACK_ID=PS6 - nothing else on that line

# StockSense AI

## Retail Sales & Inventory Copilot

StockSense AI is an intelligent retail decision-support system that converts sales and inventory data into actionable insights.

It helps users identify:
- Critical stock-out risks
- High-risk inventory
- Top-selling products
- Sales trends
- Non-moving products
- Evidence-backed inventory decisions

## Key Idea

**From raw retail data to decisions you can defend.**

The system follows:

CSV Data → Validation → Analytics → Evidence → AI Reasoning → Recommendation → Human Decision

## Features

- CSV dataset validation and ingestion
- Inventory risk detection
- Days-of-inventory-cover calculation
- Product sales ranking
- Sales trend analysis
- Non-moving product detection
- Natural-language AI queries
- Evidence-grounded Gemini responses
- Graceful fallback when AI reasoning is unavailable
- Human-review escalation for insufficient evidence
- Single-command application startup

## Technology Stack

- Python 3.11
- Flask
- Pandas
- Google Gemini API
- HTML
- CSS
- JavaScript
- python-dotenv

## Project Structure

```text
stocksense-ai/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── data/
│   ├── products.csv
│   ├── sales.csv
│   └── retail_data.csv
├── src/
│   ├── analytics.py
│   ├── ai_engine.py
│   ├── data_engine.py
│   ├── rules.py
│   └── validators.py
└── frontend/
    └── index.html