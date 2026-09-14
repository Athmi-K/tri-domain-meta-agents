"""add optional finance analysis fields

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-14 00:00:00
"""
from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("finance_profiles", sa.Column("current_savings", sa.Float(), nullable=True))
    op.add_column("finance_profiles", sa.Column("portfolio", sa.JSON(), nullable=True))
    op.add_column("finance_profiles", sa.Column("debts", sa.JSON(), nullable=True))
    op.add_column("finance_profiles", sa.Column("total_debt", sa.Float(), nullable=True))
    op.add_column("finance_profiles", sa.Column("monthly_debt_payment", sa.Float(), nullable=True))
    op.add_column("finance_profiles", sa.Column("retirement_age", sa.Integer(), nullable=True))
    op.add_column("finance_profiles", sa.Column("retirement_savings", sa.Float(), nullable=True))
    op.add_column("finance_profiles", sa.Column("monthly_contribution", sa.Float(), nullable=True))
    op.add_column("finance_profiles", sa.Column("annual_income", sa.Float(), nullable=True))
    op.add_column("finance_profiles", sa.Column("tax_deductions", sa.JSON(), nullable=True))


def downgrade():
    op.drop_column("finance_profiles", "tax_deductions")
    op.drop_column("finance_profiles", "annual_income")
    op.drop_column("finance_profiles", "monthly_contribution")
    op.drop_column("finance_profiles", "retirement_savings")
    op.drop_column("finance_profiles", "retirement_age")
    op.drop_column("finance_profiles", "monthly_debt_payment")
    op.drop_column("finance_profiles", "total_debt")
    op.drop_column("finance_profiles", "debts")
    op.drop_column("finance_profiles", "portfolio")
    op.drop_column("finance_profiles", "current_savings")
