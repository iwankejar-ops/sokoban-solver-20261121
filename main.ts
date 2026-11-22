// Sokoban solver for make.fun's "Build the best Sokoban solver, 21 November 2026".
//
// The board splits into independent rooms off one hallway: a box pushed out
// of its room can never be pushed back, so a level's push-optimal count is
// the sum of its rooms' optima, and each room can be solved on its own.
// Inside a room the search runs on pushes: a state is the boxes plus the
// region the player can reach, and A* uses the sum of per-box push
// distances to the nearest goal (a reverse push BFS that ignores other
// boxes, so it is admissible and also marks dead cells). The room
// solutions are stitched together with walks along the hallway, and the
// player must be able to walk back out of every room it finishes, except
// a room that can only be solved by staying inside, which goes last.
//
// The program is deterministic: no clock, no environment, no randomness.
// Work is bounded by node counts, never by time, so two runs write the
// same bytes.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Four steps: row delta, column delta, move letter.
const DIRS = [[-1, 0, 'u'], [1, 0, 'd'], [0, -1, 'l'], [0, 1, 'r']] as const;

// A parsed level. Cells are flat indices r * w + c; '#' is a wall.
type Grid = {
  w: number;
  h: number;
  wall: boolean[];
  goal: boolean[];
  boxes: number[];
  player: number;
};

function parseLevel(text: string): Grid {
  const lines = text.split('\n').filter((l) => l.length > 0);
  const h = lines.length;
  const w = Math.max(...lines.map((l) => l.length));
  const wall = new Array<boolean>(w * h).fill(false);
  const goal = new Array<boolean>(w * h).fill(false);
  const boxes: number[] = [];
  let player = -1;
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const ch = lines[r][c] ?? '#';
      const i = r * w + c;
      if (ch === '#') wall[i] = true;
      else if (ch === '.') goal[i] = true;
      else if (ch === '$') boxes.push(i);
      else if (ch === '*') {
        goal[i] = true;
        boxes.push(i);
      } else if (ch === '@') player = i;
      else if (ch === '+') {
        goal[i] = true;
        player = i;
      }
    }
  }
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goal, boxes, player };
}

// One room: its floor cells (including its door cell), the door cell itself,
// and the hallway cell the player enters from.
type Room = {
  cells: Set<number>;
  door: number;
  entry: number;
  boxes: number[];
};

// The player starts in the hallway, so the hallway is the player's row.
// Every room is one connected component of passable cells off that row.
function findRooms(g: Grid): Room[] {
  const { w, h, wall } = g;
  const hr = Math.floor(g.player / w);
  const seen = new Array<boolean>(w * h).fill(false);
  const rooms: Room[] = [];
  for (let start = 0; start < w * h; start++) {
    const r = Math.floor(start / w);
    if (r === hr || wall[start] || seen[start]) continue;
    const cells = new Set<number>();
    const stack = [start];
    seen[start] = true;
    while (stack.length > 0) {
      const cur = stack.pop()!;
      cells.add(cur);
      const cr = Math.floor(cur / w);
      const cc = cur % w;
      for (const [dr, dc] of DIRS) {
        const nr = cr + dr;
        const nc = cc + dc;
        if (nr < 0 || nr >= h || nc < 0 || nc >= w) continue;
        const ni = nr * w + nc;
        if (nr === hr || wall[ni] || seen[ni]) continue;
        seen[ni] = true;
        stack.push(ni);
      }
    }
    // The door is the room cell in the row next to the hallway.
    let door = -1;
    for (const cell of cells) {
      const cr = Math.floor(cell / w);
      if (cr === hr - 1 || cr === hr + 1) door = cell;
    }
    if (door < 0) continue; // a pocket with no door is not a room
    // The entry cell is the hallway cell in the door's column.
    const entryCell = hr * w + (door % w);
    rooms.push({ cells, door, entry: entryCell, boxes: g.boxes.filter((b) => cells.has(b)) });
  }
  // Left to right by door column; the order between rooms does not change
  // the push count, only the walk.
  rooms.sort((a, b) => (a.door % w) - (b.door % w));
  return rooms;
}

// Walkable cells from `from`, with `boxes` as obstacles, over `allowed`
// cells. Returns the parent map (cell -> [previous cell, move letter]) so
// a walk to any reached cell can be spelled out.
function walkParents(
  g: Grid,
  allowed: Set<number>,
  boxes: Set<number>,
  from: number,
): Map<number, [number, string]> {
  const parent = new Map<number, [number, string]>();
  parent.set(from, [-1, '']);
  const queue = [from];
  let head = 0;
  while (head < queue.length) {
    const cur = queue[head++];
    const cr = Math.floor(cur / g.w);
    const cc = cur % g.w;
    for (const [dr, dc, letter] of DIRS) {
      const nr = cr + dr;
      const nc = cc + dc;
      if (nr < 0 || nr >= g.h || nc < 0 || nc >= g.w) continue;
      const ni = nr * g.w + nc;
      if (!allowed.has(ni) || boxes.has(ni) || parent.has(ni)) continue;
      parent.set(ni, [cur, letter]);
      queue.push(ni);
    }
  }
  return parent;
}

// The walk from `from` to `to`, or null when it cannot be walked.
function walkPath(
  g: Grid,
  allowed: Set<number>,
  boxes: number[],
  from: number,
  to: number,
): string | null {
  if (from === to) return '';
  const parent = walkParents(g, allowed, new Set(boxes), from);
  if (!parent.has(to)) return null;
  let moves = '';
  let cur = to;
  while (cur !== from) {
    const [prev, letter] = parent.get(cur)!;
    moves = letter + moves;
    cur = prev;
  }
  return moves;
}

// A small binary heap, ordered by f, then by h, then by insertion order,
// so the search is deterministic.
type HeapItem = { f: number; h: number; order: number; node: SearchNode };
class Heap {
  private a: HeapItem[] = [];
  get size(): number {
    return this.a.length;
  }
  push(item: HeapItem): void {
    this.a.push(item);
    let i = this.a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.less(this.a[i], this.a[p])) {
        [this.a[i], this.a[p]] = [this.a[p], this.a[i]];
        i = p;
      } else break;
    }
  }
  pop(): HeapItem {
    const top = this.a[0];
    const last = this.a.pop()!;
    if (this.a.length > 0) {
      this.a[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < this.a.length && this.less(this.a[l], this.a[m])) m = l;
        if (r < this.a.length && this.less(this.a[r], this.a[m])) m = r;
        if (m === i) break;
        [this.a[i], this.a[m]] = [this.a[m], this.a[i]];
        i = m;
      }
    }
    return top;
  }
  private less(x: HeapItem, y: HeapItem): boolean {
    if (x.f !== y.f) return x.f < y.f;
    if (x.h !== y.h) return x.h < y.h;
    return x.order < y.order;
  }
}

type SearchNode = {
  boxes: number[];
  player: number;
  g: number;
  parent: SearchNode | null;
  step: string;
};

type RoomSolution = {
  moves: string;
  pushes: number;
  endPlayer: number;
  endBoxes: number[];
  exitPossible: boolean;
};

// Minimum pushes for one box to reach any goal, ignoring other boxes,
// by a reverse push BFS from the goals. A cell with no distance is dead:
// a box pushed there can never reach a goal.
function pushDistances(g: Grid, allowed: Set<number>): Map<number, number> {
  const dist = new Map<number, number>();
  const queue: number[] = [];
  for (const cell of allowed) {
    if (g.goal[cell]) {
      dist.set(cell, 0);
      queue.push(cell);
    }
  }
  let head = 0;
  while (head < queue.length) {
    const z = queue[head++];
    const zr = Math.floor(z / g.w);
    const zc = z % g.w;
    const dz = dist.get(z)!;
    for (const [dr, dc] of DIRS) {
      const b = (zr - dr) * g.w + (zc - dc);
      const p = (zr - 2 * dr) * g.w + (zc - 2 * dc);
      if (!allowed.has(b) || !allowed.has(p) || dist.has(b)) continue;
      dist.set(b, dz + 1);
      queue.push(b);
    }
  }
  return dist;
}

// True when a push that lands a box on `dest` builds a 2 by 2 block of
// walls and boxes with no goal in it: those boxes can never move again.
function makesFrozenBlock(
  g: Grid,
  boxes: Set<number>,
  dest: number,
): boolean {
  const dr = Math.floor(dest / g.w);
  const dc = dest % g.w;
  for (const [or, oc] of [[0, 0], [-1, 0], [0, -1], [-1, -1]] as const) {
    const cells = [
      (dr + or) * g.w + (dc + oc),
      (dr + or) * g.w + (dc + oc + 1),
      (dr + or + 1) * g.w + (dc + oc),
      (dr + or + 1) * g.w + (dc + oc + 1),
    ];
    let blocked = true;
    let hasGoal = false;
    for (const cell of cells) {
      const r = Math.floor(cell / g.w);
      const c = cell % g.w;
      if (r < 0 || r >= g.h || c < 0 || c >= g.w || g.wall[cell]) continue;
      if (g.goal[cell]) hasGoal = true;
      if (!boxes.has(cell)) blocked = false;
    }
    if (blocked && !hasGoal) return true;
  }
  return false;
}

// Solve one room push-optimally, starting with the player on its entry
// cell. When requireExit is set, the player must be able to walk back out
// of the room once every box is on a goal; without it, any solved state
// counts. Returns null when the room cannot be solved within the node cap.
function solveRoom(
  g: Grid,
  room: Room,
  startBoxes: number[],
  requireExit: boolean,
  nodeCap: number,
): RoomSolution | null {
  const allowed = new Set<number>(room.cells);
  allowed.add(room.entry);
  const dist = pushDistances(g, allowed);
  const start = startBoxes.filter((b) => room.cells.has(b)).sort((a, b) => a - b);
  for (const b of start) {
    if (!dist.has(b)) return null; // a box already stands on a dead cell
  }
  const goals = new Set<number>();
  for (const cell of room.cells) if (g.goal[cell]) goals.add(cell);

  const heuristic = (boxes: number[]): number => {
    let s = 0;
    for (const b of boxes) {
      const d = dist.get(b);
      if (d === undefined) return Infinity;
      s += d;
    }
    return s;
  };

  const startNode: SearchNode = {
    boxes: start,
    player: room.entry,
    g: 0,
    parent: null,
    step: '',
  };
  const heap = new Heap();
  let order = 0;
  const h0 = heuristic(start);
  heap.push({ f: h0, h: h0, order: order++, node: startNode });
  const best = new Map<string, number>();
  const startRegion = walkParents(g, allowed, new Set(start), room.entry);
  let startMin = Infinity;
  for (const cell of startRegion.keys()) if (cell < startMin) startMin = cell;
  best.set(start.join(',') + '|' + startMin, 0);
  let expanded = 0;

  while (heap.size > 0) {
    const { node } = heap.pop();
    const nodeKey = node.boxes.join(',') + '|' + regionMinOf(g, allowed, node);
    const recorded = best.get(nodeKey);
    if (recorded !== undefined && recorded < node.g) continue;
    expanded++;
    if (expanded > nodeCap) return null;

    // Goal: every box on a goal, and (when required) the player able to
    // walk back out to the entry cell.
    let solved = true;
    for (const b of node.boxes) {
      if (!goals.has(b)) {
        solved = false;
        break;
      }
    }
    if (solved) {
      if (!requireExit) {
        return finishRoom(node, false);
      }
      const region = walkParents(g, allowed, new Set(node.boxes), node.player);
      if (region.has(room.entry)) return finishRoom(node, true);
      // Solved but the player is trapped inside: keep searching for a
      // solved state it can walk out of.
      continue;
    }

    // The player's reachable region, with the walk to every cell in it.
    const region = walkParents(g, allowed, new Set(node.boxes), node.player);
    const boxSet = new Set(node.boxes);
    for (const box of node.boxes) {
      const br = Math.floor(box / g.w);
      const bc = box % g.w;
      for (const [dr, dc, letter] of DIRS) {
        const pushFrom = (br - dr) * g.w + (bc - dc);
        const dest = (br + dr) * g.w + (bc + dc);
        if (!region.has(pushFrom)) continue;
        if (!room.cells.has(dest) || boxSet.has(dest)) continue;
        if (!dist.has(dest)) continue; // dead cell
        const next = node.boxes.map((b) => (b === box ? dest : b));
        next.sort((a, b) => a - b);
        const nextSet = new Set(next);
        if (makesFrozenBlock(g, nextSet, dest)) continue;
        const h = heuristic(next);
        if (h === Infinity) continue;
        // The walk to the push position, then the push itself.
        let walk = '';
        let cur = pushFrom;
        while (cur !== node.player) {
          const [prev, stepLetter] = region.get(cur)!;
          walk = stepLetter + walk;
          cur = prev;
        }
        const child: SearchNode = {
          boxes: next,
          player: box,
          g: node.g + 1,
          parent: node,
          step: walk + letter,
        };
        const childRegion = walkParents(g, allowed, nextSet, box);
        let childMin = Infinity;
        for (const cell of childRegion.keys()) if (cell < childMin) childMin = cell;
        const childKey = next.join(',') + '|' + childMin;
        const prevBest = best.get(childKey);
        if (prevBest !== undefined && prevBest <= child.g) continue;
        best.set(childKey, child.g);
        heap.push({ f: child.g + h, h, order: order++, node: child });
      }
    }
  }
  return null;
}

// The smallest cell in the player's reachable region for a search node;
// two nodes with the same boxes and the same region are the same state.
function regionMinOf(g: Grid, allowed: Set<number>, node: SearchNode): number {
  const region = walkParents(g, allowed, new Set(node.boxes), node.player);
  let m = Infinity;
  for (const cell of region.keys()) if (cell < m) m = cell;
  return m;
}

function finishRoom(node: SearchNode, exitPossible: boolean): RoomSolution {
  let moves = '';
  let pushes = 0;
  let cur: SearchNode | null = node;
  const parts: string[] = [];
  while (cur !== null && cur.parent !== null) {
    parts.push(cur.step);
    pushes++;
    cur = cur.parent;
  }
  parts.reverse();
  moves = parts.join('');
  return { moves, pushes, endPlayer: node.player, endBoxes: node.boxes, exitPossible };
}

// Solve one whole level: walk between rooms, solve each room in turn.
function solveLevel(text: string): string {
  const g = parseLevel(text);
  const rooms = findRooms(g);
  if (rooms.length === 0) return '';
  // Solve every room on its own first, so a room that can only be solved
  // by staying inside can be moved to the end of the order.
  const solved: { room: Room; sol: RoomSolution }[] = [];
  const deferred: { room: Room; sol: RoomSolution }[] = [];
  for (const room of rooms) {
    const roomBoxes = g.boxes.filter((b) => room.cells.has(b));
    let sol = solveRoom(g, room, roomBoxes, true, 200000);
    if (sol !== null) {
      solved.push({ room, sol });
    } else {
      const loose = solveRoom(g, room, roomBoxes, false, 200000);
      if (loose === null) return '';
      deferred.push({ room, sol: loose });
    }
  }
  const order = [...solved, ...deferred];
  // Stitch: walk to each room's entry, then play its solution. Boxes in
  // rooms already solved sit on their goals and stay out of the way.
  const allCells = new Set<number>();
  for (let i = 0; i < g.w * g.h; i++) if (!g.wall[i]) allCells.add(i);
  let boxes = [...g.boxes];
  let player = g.player;
  let moves = '';
  for (const { room, sol } of order) {
    const walk = walkPath(g, allCells, boxes, player, room.entry);
    if (walk === null) return '';
    moves += walk;
    player = room.entry;
    moves += sol.moves;
    // Apply the room's solution to the global state: its boxes end where
    // the room solution leaves them, the player where it leaves it.
    const roomBoxSet = new Set(room.cells);
    boxes = boxes.filter((b) => !roomBoxSet.has(b)).concat(sol.endBoxes);
    boxes.sort((a, b) => a - b);
    player = sol.endPlayer;
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
