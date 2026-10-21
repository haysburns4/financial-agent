"""The database-backed watchlist: what the user follows, and what is monitored.

- resolver.py: watchlist ∪ held positions, computed per call (read by everything)
- service.py: user-driven mutations, their events, and backfill-on-add
"""
