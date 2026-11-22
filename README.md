# Sokoban Solver

A push-optimal Sokoban solver for the
[Build the best Sokoban solver, 21 November 2026](https://make.fun/bounty/sokoban-solver-01-run-20261121)
bounty on [make.fun](https://make.fun).

![Bounty](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)
![Rank](https://make.fun/badge/sokoban-solver-01-run-20261121/iwankejar-ops/sokoban-solver-20261121/rank.svg)
![Reward](https://make.fun/badge/sokoban-solver-01-run-20261121/iwankejar-ops/sokoban-solver-20261121/reward.svg)

## How it works

The board splits into independent rooms off one hallway: a box pushed
out of its room can never be pushed back, so a level's push-optimal
count is the sum of its rooms' optima, and each room is solved on its
own. Inside a room the search runs on pushes — a state is the boxes
plus the region the player can reach — and A* uses the sum of per-box
push distances to the nearest goal as its heuristic. The room solutions
are stitched together with walks along the hallway.

## Run

```sh
node main.ts
```

Reads `/input/<id>/level.txt` and writes `/output/<id>/solution.txt`
for every instance. Deterministic: no clock, no environment, no
randomness, so two runs write the same bytes.
