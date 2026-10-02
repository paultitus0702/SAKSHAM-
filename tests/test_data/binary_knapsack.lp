\ 0-1 Binary Knapsack model
Maximize
  value: 10 y1 + 15 y2 + 25 y3 + 8 y4
Subject To
  weight_cap: 3 y1 + 4 y2 + 7 y3 + 2 y4 <= 10
Binary
  y1 y2 y3 y4
End
