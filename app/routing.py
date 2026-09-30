"""危险品运输路线编排：带禁行窗口与节点等待的时间相关多标号最短路。

输入为本地有向道路图：
- 路段：行驶分钟、禁运危险类别（恒闭）、UTC 禁行窗口 [start, end)（半开）；
- 节点：是否允许等待。

通行规则：
- 车辆在时刻 t 进入行驶时长 d 的路段，通行区间为 [t, t+d)；
  与任一禁行窗口 [s, e) 相交（t < e 且 t+d > s）即不可进入；
  t == e（窗口结束瞬间进入）或 t+d == s（窗口开始瞬间到达）均允许；
- 节点允许等待时，等待到该路段下一可行时刻再进入；不允许等待则该走法不可行；
- 禁运类别为恒闭限制，等待不能解除。

唯一结果按字典序目标确定：最早到达 → 总等待最少 → 路段 ID 序列字典序。
由于"在禁行窗口前等待"会抹平不同走法的到达时间差（更长的行驶路径可能用更少
等待追平到达时刻），同一节点需要保留多个互不支配的标号。标号状态为
(到达时刻 t, 累计行驶 drive)，t == drive + 总等待；对任意后续路段序列：

- 到达时刻是关于 t 的非递减函数（FIFO），t 更小者不会更晚到达；
- 被窗口"抹平"到达时刻时，drive 更大者总等待更少；

支配判定分两种节点（还要求支配者的已用边集合包含于被支配者，保证后续可选边
不更少；相同 `(t, drive)` 但已用边集合不同的标号不可合并——它们对后续边的
可用性不同，最优路线可能合法地重访节点走不同边）：

- **允许等待**的节点：早到标号可等待到晚到标号的时刻，故 A 支配 B 当且仅当
  t_A < t_B 且 drive_A **严格大于** drive_B（drive 相等时等待差会被追平，
  到达与等待都相同，必须靠路段序列字典序裁决，双方都保留）；
- **禁止等待**的节点：t 不同的标号不可比——晚到标号可能恰在窗口结束后通过，
  而早到标号被窗口关闭且无法等待。

两类节点共有的安全规则：t 相同则 drive 更大者支配（同一边可行性只取决于
节点与到达时刻，后续每个时刻都更优且等待更少）；相同 `(t, drive)` 且
已用边集合相同时，只保留路段序列字典序最小者。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import heapq


@dataclass(frozen=True)
class Window:
    """禁行窗口 [start, end)，时刻为相对出发时刻的分钟偏移。"""

    start: int
    end: int


@dataclass(frozen=True)
class RoadNode:
    node_id: str
    allow_waiting: bool


@dataclass(frozen=True)
class RoadSegment:
    segment_id: str
    from_node: str
    to_node: str
    drive_minutes: int
    embargo_classes: frozenset[str]
    windows: tuple[Window, ...]  # 已按 start 升序排列


@dataclass(frozen=True)
class Step:
    """逐段时刻记录。"""

    segment_id: str
    from_node: str
    to_node: str
    departure: int  # 进入路段的时刻（相对出发的分钟偏移）
    arrival: int  # 到达下游节点的时刻
    drive_minutes: int
    wait_before_minutes: int  # 在上游节点为本段等待的分钟数


@dataclass(frozen=True)
class Wait:
    """节点等待记录。"""

    node_id: str
    for_segment_id: str
    start: int  # 到达节点（开始等待）的时刻
    end: int  # 实际驶离的时刻
    wait_minutes: int
    blocking_windows: tuple[Window, ...]  # 导致等待的禁行窗口（按开始时刻升序）


@dataclass(frozen=True)
class EmbargoBlockage:
    """阻断原因：路段对本车载货类别恒闭。"""

    type: str
    node_id: str
    segment_id: str
    to_node: str
    embargo_classes: tuple[str, ...]  # 该路段全部禁运类别
    matched_classes: tuple[str, ...]  # 与本车载货类别相交的部分


@dataclass(frozen=True)
class NoWaitingBlockage:
    """阻断原因：最早到达时路段被窗口关闭，而节点不允许等待。"""

    type: str
    node_id: str
    segment_id: str
    to_node: str
    arrival: int
    drive_minutes: int
    windows: tuple[Window, ...]  # 该到达时刻下与通行区间相交的窗口


@dataclass(frozen=True)
class UnreachableBlockage:
    """兜底阻断原因：起止点之间不存在任何可行路线。"""

    type: str
    origin: str
    destination: str


@dataclass
class RoutePlan:
    status: str  # success / infeasible
    steps: list[Step] = field(default_factory=list)
    waits: list[Wait] = field(default_factory=list)
    arrival: int | None = None
    total_drive: int = 0
    total_wait: int = 0
    blockages: list = field(default_factory=list)


@dataclass(frozen=True)
class _Label:
    """搜索标号：节点、到达时刻、累计行驶/等待分钟、完整路径与记录。"""

    node: str
    t: int
    drive: int
    total_wait: int
    path: tuple[str, ...]
    steps: tuple[Step, ...]
    waits: tuple[Wait, ...]
    visited: frozenset[str] = frozenset()  # 本路径已用路段（不重复走同一边）


def earliest_entry(
    t: int, drive: int, windows: tuple[Window, ...], allow_waiting: bool
) -> tuple[int | None, int, tuple[Window, ...]]:
    """计算在时刻 t 进入时长 drive 的路段的最早可行进入时刻。

    返回 (进入时刻, 等待分钟, 导致等待的窗口)；不允许等待且当前不可进入时
    进入时刻为 None，第三个返回值为当前与通行区间相交的窗口。
    """
    cand = t
    forced: list[Window] = []
    seen: set[Window] = set()
    while True:
        # [cand, cand+drive) 与 [w.start, w.end) 相交
        hits = [w for w in windows if cand < w.end and cand + drive > w.start]
        if not hits:
            return cand, cand - t, tuple(forced)
        if not allow_waiting:
            return None, 0, tuple(sorted(hits, key=lambda w: (w.start, w.end)))
        for w in hits:
            if w not in seen:
                seen.add(w)
                forced.append(w)
        # 跳到所有相交窗口中最晚的结束时刻（相邻/重叠窗口可能需要多跳）
        cand = max(cand, max(w.end for w in hits))


def plan_route(
    segments: list[RoadSegment],
    nodes: list[RoadNode],
    origin: str,
    destination: str,
    carried_classes: set[str],
) -> RoutePlan:
    """编排唯一最优路线；不可行时返回稳定排序的阻断原因且不含任何行程段。"""
    if origin == destination:
        return RoutePlan(status="success", arrival=0)  # 零段行程：到达时刻即出发时刻

    adj: dict[str, list[RoadSegment]] = {}
    for seg in segments:
        adj.setdefault(seg.from_node, []).append(seg)
    for outs in adj.values():
        outs.sort(key=lambda s: s.segment_id)  # 放松顺序确定，保证记录可复现
    allow_wait = {n.node_id: n.allow_waiting for n in nodes}

    root = _Label(node=origin, t=0, drive=0, total_wait=0, path=(), steps=(), waits=(),
                  visited=frozenset())
    buckets: dict[str, list[_Label]] = {origin: [root]}
    heap: list[tuple[int, int, _Label]] = [(0, 0, root)]
    counter = 1
    best_dest: _Label | None = None

    # 阻断原因收集（同一“节点 × 路段”只记一条；no_waiting 取最早到达尝试）
    embargo_hits: dict[tuple[str, str], EmbargoBlockage] = {}
    nowait_hits: dict[tuple[str, str], NoWaitingBlockage] = {}
    # 至少被一个标号成功通过的边：这类边不算阻断（另一标号可能因禁等被窗口挡住，
    # 但更晚到达的标号恰可在窗口结束后通过）
    passed_edges: set[tuple[str, str]] = set()

    def insert(label: _Label) -> None:
        nonlocal counter
        # 到达终点不可能比已知最优到达“更早”（后续行驶分钟为正），故仅严格更晚才丢弃；
        # t 相同的标号必须保留——总等待可能更少、路段序列可能字典序更小
        if best_dest is not None and label.t > best_dest.t:
            return
        # 禁等节点上“更早到达”不能等待到晚到时刻，晚到标号可能恰好越过窗口，
        # 故 t 更小的标号不具支配力；仅相同 t 可比（边可行性只取决于节点与 t）。
        # 跨标号支配还要求已用边集合包含（visited_A ⊆ visited_B）：A 能走的后续边
        # 不比 B 少，否则即便时刻更优也可能因边已被占用而缺少 B 的可行延续。
        waiting_node = allow_wait.get(label.node, False)
        bucket = buckets.setdefault(label.node, [])
        survivors: list[_Label] = []
        for old in bucket:
            if old.t == label.t and old.drive == label.drive \
                    and old.visited == label.visited:
                # 相同状态：只保留路段序列字典序最小者
                if label.path < old.path:
                    continue  # old 被替换
                return
            # 新标号被旧标号支配
            if (old.t == label.t and old.drive > label.drive
                    and old.visited <= label.visited):
                return
            if (waiting_node and old.t < label.t and old.drive > label.drive
                    and old.visited <= label.visited):
                return
            # 旧标号被新标号支配
            if (old.t == label.t and label.drive > old.drive
                    and label.visited <= old.visited):
                continue
            if (waiting_node and label.t < old.t and label.drive > old.drive
                    and label.visited <= old.visited):
                continue
            survivors.append(old)
        survivors.append(label)
        buckets[label.node] = survivors
        heapq.heappush(heap, (label.t, counter, label))
        counter += 1

    while heap:
        _, _, label = heapq.heappop(heap)
        # 可能已在出堆前被更优标号淘汰
        if not any(l is label for l in buckets.get(label.node, ())):
            continue
        if best_dest is not None and label.t > best_dest.t:
            continue
        if label.node == destination:
            if best_dest is None or (label.t, label.total_wait, label.path) < (
                best_dest.t,
                best_dest.total_wait,
                best_dest.path,
            ):
                best_dest = label
            continue

        for seg in adj.get(label.node, ()):
            key = (label.node, seg.segment_id)
            if seg.segment_id in label.visited:
                # 同一路径不重复走同一边：含重复边必带环，删环后首达不晚（FIFO）、
                # 等待/行驶不增、序列字典序更小，故最优路线必为简单边路径
                continue
            matched = seg.embargo_classes & carried_classes
            if matched:
                # 禁运类别为恒闭阻断，与到达时刻无关
                embargo_hits.setdefault(
                    key,
                    EmbargoBlockage(
                        type="embargo",
                        node_id=label.node,
                        segment_id=seg.segment_id,
                        to_node=seg.to_node,
                        embargo_classes=tuple(sorted(seg.embargo_classes)),
                        matched_classes=tuple(sorted(matched)),
                    ),
                )
                continue

            dep, wait_min, forced = earliest_entry(
                label.t, seg.drive_minutes, seg.windows, allow_wait[label.node]
            )
            if dep is None:
                # 多标号按到达时刻出堆，首条即最早到达尝试，冻结为候选阻断原因；
                # 但若存在另一标号成功通过该边，则该边整体可行，最终会被剔除
                nowait_hits.setdefault(
                    key,
                    NoWaitingBlockage(
                        type="no_waiting",
                        node_id=label.node,
                        segment_id=seg.segment_id,
                        to_node=seg.to_node,
                        arrival=label.t,
                        drive_minutes=seg.drive_minutes,
                        windows=forced,
                    ),
                )
                continue
            passed_edges.add(key)

            step = Step(
                segment_id=seg.segment_id,
                from_node=label.node,
                to_node=seg.to_node,
                departure=dep,
                arrival=dep + seg.drive_minutes,
                drive_minutes=seg.drive_minutes,
                wait_before_minutes=wait_min,
            )
            waits = label.waits
            if wait_min > 0:
                waits = waits + (Wait(
                    node_id=label.node,
                    for_segment_id=seg.segment_id,
                    start=label.t,
                    end=dep,
                    wait_minutes=wait_min,
                    blocking_windows=tuple(sorted(forced, key=lambda w: (w.start, w.end))),
                ),)
            insert(_Label(
                node=seg.to_node,
                t=dep + seg.drive_minutes,
                drive=label.drive + seg.drive_minutes,
                total_wait=label.total_wait + wait_min,
                path=label.path + (seg.segment_id,),
                steps=label.steps + (step,),
                waits=waits,
                visited=label.visited | {seg.segment_id},
            ))

    if best_dest is not None:
        return RoutePlan(
            status="success",
            steps=list(best_dest.steps),
            waits=list(best_dest.waits),
            arrival=best_dest.t,
            total_drive=best_dest.drive,
            total_wait=best_dest.total_wait,
        )

    # 不可行：阻断原因稳定排序（类型 → 节点 → 路段），绝不返回部分行程。
    # 边被任一标号成功通过即整体可行（禁等节点上不同到达时刻的可行性可能不同），
    # 只保留从未通过的边。
    type_rank = {"embargo": 0, "no_waiting": 1}
    blockages: list = sorted(
        [*embargo_hits.values(),
         *(b for k, b in nowait_hits.items() if k not in passed_edges)],
        key=lambda b: (type_rank[b.type], b.node_id, b.segment_id),
    )
    if not blockages:
        blockages.append(UnreachableBlockage(
            type="unreachable", origin=origin, destination=destination
        ))
    return RoutePlan(status="infeasible", blockages=blockages)
