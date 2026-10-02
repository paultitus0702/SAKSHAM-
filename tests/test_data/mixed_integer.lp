\ Mixed-Integer Linear Program (MILP)
Minimize
  total_cost: 100 y_setup + 5 x_prod + 2 s_inv
Subject To
  demand: x_prod - s_inv >= 50
  capacity: x_prod - 200 y_setup <= 0
  holding_limit: s_inv <= 80
Bounds
  x_prod >= 0
  s_inv >= 0
Binary
  y_setup
Generals
  s_inv
End
