"""add position_snapshots

Revision ID: be92fb786e44
Revises:
Create Date: 2026-10-08 09:07:13.917288
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'be92fb786e44'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "position_snapshots",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("snapshot_date", sa.Date(), nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("capture_source", sa.String(length=20), nullable=False),
        sa.Column("market_state", sa.String(length=20), nullable=False),
        sa.Column("account_id", sa.String(length=50), nullable=False),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("cost_basis", sa.Float(), nullable=False),
        sa.Column("market_value", sa.Float(), nullable=False),
        sa.Column("pnl", sa.Float(), nullable=False),
        sa.Column("pnl_pct", sa.Float(), nullable=True),
        sa.Column("positions_as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "snapshot_date", "account_id", "ticker", name="uq_position_snapshots_date_account_ticker"
        ),
    )
    with op.batch_alter_table("position_snapshots") as batch_op:
        batch_op.create_index("ix_position_snapshots_snapshot_date", ["snapshot_date"])
        batch_op.create_index("ix_position_snapshots_account_id", ["account_id"])
        batch_op.create_index("ix_position_snapshots_ticker", ["ticker"])
        batch_op.create_index("ix_position_snapshots_date_account", ["snapshot_date", "account_id"])


def downgrade() -> None:
    with op.batch_alter_table("position_snapshots") as batch_op:
        batch_op.drop_index("ix_position_snapshots_date_account")
        batch_op.drop_index("ix_position_snapshots_ticker")
        batch_op.drop_index("ix_position_snapshots_account_id")
        batch_op.drop_index("ix_position_snapshots_snapshot_date")
    op.drop_table("position_snapshots")
