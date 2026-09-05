"""
StockSense AI backend package.

Pipeline: Retail CSV -> validators -> data_engine -> analytics (+ rules)
          -> evidence -> ai_engine (Gemini) -> grounded answer -> app.py routes.
"""

__version__ = "0.1.0"
