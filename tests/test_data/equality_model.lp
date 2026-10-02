\ LP with equality constraint and free variable
Minimize
  cost: 2 x1 + 3 x2 - x3
Subject To
  balance: x1 + x2 - x3 = 5
  limit: 2 x1 + 4 x2 <= 20
Bounds
  x1 free
  x2 >= 0
  0 <= x3 <= 15
End
