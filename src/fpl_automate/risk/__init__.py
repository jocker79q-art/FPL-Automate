"""Player- and squad-level risk classification, and (see
`optimization/lineup.py`'s `"risk_adjusted"` strategy) a mean-variance
optimizer built on top of it. Distinct from `projections/`: this module
never produces a points estimate, only interprets the uncertainty already
attached to one (`PlayerProjection.floor_points`/`ceiling_points`/
`confidence`/`risk_flags`).
"""
