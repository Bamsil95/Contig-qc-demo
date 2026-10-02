"""AB1 기반 Contig 생성 예측 도구.

실행: python -m streamlit run "app_Streamlit(5).py"
설치(Python 3.10 이상): python -m pip install streamlit pandas biopython

네 조건에 동일한 정렬·품질 판정 규칙을 적용한다. 결과는 경험적 예측이며
실제 조립 결과나 검증된 성공 확률이 아니다. 샘플별 정답 보정은 사용하지 않는다.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import struct
from dataclasses import asdict, dataclass

import pandas as pd
import streamlit as st
from Bio import SeqIO
from Bio.Align import PairwiseAligner
from Bio.Seq import Seq

APP_VERSION = "2026.10.02-ab1-v6"
MAX_READ_BASES = 5000
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MIN_READ_BASES = 50
MIN_Q20_READ_BASES = 50
MIN_OVERLAP_BASES = 30
MIN_GAP_IDENTITY = 90.0
MIN_Q20_MATCHES = 20
MAX_Q20_CONFLICT_RATE = 0.02
MAX_Q20_JUNCTION = 10
MAX_GAP_RUN = 5
CONFLICT_CLUSTER_COLUMNS = 100
MIN_CLUSTER_Q20_CONFLICTS = 5
# 실제 조립 출력과 추가 대조가 필요한 검증용 보정 기준.
MIN_BIDIRECTIONAL_GAP_REGIONS = 2
TERMINAL_GAP_COLUMNS = 10
LOW_QUALITY_END_COLUMNS = 12
MAX_CORE_TRIM_COLUMNS = 5
CORE_CONFLICT_QUALITY_LIMIT = 30


@dataclass(frozen=True)
class Condition:
    label: str
    qt: int
    new_qt_enabled: bool
    window_size: int | None
    second_qt_enabled: bool = False

    def __post_init__(self):
        if not 1 <= self.qt <= 60:
            raise ValueError("QT는 1~60 사이여야 합니다.")
        if self.new_qt_enabled:
            if type(self.window_size) is not int or self.window_size < 1:
                raise ValueError("양의 정수 Window size가 필요합니다.")
        elif self.window_size is not None:
            raise ValueError("사용하지 않는 Window size가 지정되었습니다.")


CONDITIONS = (
    Condition("16", 16, False, None),
    Condition("20/10", 20, True, 10),
    Condition("30/20", 30, True, 20),
    Condition("10", 10, False, None),
)


@dataclass(frozen=True)
class Read:
    name: str
    sequence: str
    qualities: tuple[int, ...]
    file_name: str = ""
    source_sha256: str = ""

    def __post_init__(self):
        if not self.name or re.search(r"\s", self.name):
            raise ValueError("Read ID는 공백 없는 문자열이어야 합니다.")
        if len(self.sequence) > MAX_READ_BASES:
            raise ValueError(f"Read당 {MAX_READ_BASES:,} bp 이하의 파일을 지원합니다.")
        if len(self.qualities) != len(self.sequence):
            raise ValueError("서열과 Quality 길이가 다릅니다.")
        if any(type(q) is not int or not 0 <= q <= 99 for q in self.qualities):
            raise ValueError("Quality에는 0~99의 정수만 사용할 수 있습니다.")
        if set(self.sequence.upper()) - set("ACGTRYSWKMBDHVNX"):
            raise ValueError("DNA 서열에 지원하지 않는 문자가 있습니다.")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_ab1(data: bytes, filename: str, read_id: str) -> Read:
    if not filename.lower().endswith(".ab1") or not data.startswith(b"ABIF"):
        raise ValueError("올바른 AB1 파일을 업로드해 주세요.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("AB1 파일이 20 MiB를 초과합니다.")
    try:
        record = SeqIO.read(io.BytesIO(data), "abi")
    except (ValueError, IndexError, KeyError, OSError, EOFError, struct.error) as exc:
        raise ValueError(f"{filename}: AB1 파일을 읽을 수 없습니다.") from exc
    qualities = record.letter_annotations.get("phred_quality")
    if qualities is None:
        raise ValueError(f"{filename}: 염기별 Quality가 없습니다.")
    if not record.seq:
        raise ValueError(f"{filename}: 분석할 염기가 없습니다.")
    return Read(read_id, str(record.seq).upper(), tuple(qualities), filename, sha256(data))


def trim_read(read: Read, condition: Condition):
    """원본 AB1과 실제 출력의 대응에서 복원한 전처리 규칙.

    기본 QT: 양 끝에서 폭 10의 비중첩 구간을 검사한다. 뒤쪽 검사창은
    현재 염기를 포함하고, 확인된 오른쪽 좌표를 slice의 배타적 끝으로 쓴다.
    New QT: QT 이상 염기가 75% 이상인 첫 window에서 시작하고, 그 이후
    window 평균 Q가 QT 미만이 되는 첫 위치에서 끝낸다. 내부를 이어붙이지 않는다.

    제공된 16개 read 출력과 서열 전체가 일치한 복원 규칙이며, 미제공 조건까지
    원래 구현과 동일하다고 보증하지 않는다. 파일명·서열 정답 조회는 사용하지 않는다.
    """
    if condition.second_qt_enabled:
        raise ValueError("지원하지 않는 추가 trimming 설정입니다.")
    n = len(read.sequence)
    q = read.qualities
    if condition.new_qt_enabled:
        width = condition.window_size
        start = next((i for i in range(n - width + 1)
                      if 4 * sum(value >= condition.qt for value in q[i:i + width]) >= 3 * width), n)
        end = next((i for i in range(start, n - width + 1)
                    if sum(q[i:i + width]) < condition.qt * width), n)
    else:
        width = 10
        start = next((i for i in range(0, n - width + 1, width)
                      if sum(q[i:i + width]) >= condition.qt * width), n)
        end = next((i for i in range(n - 1, width - 2, -width)
                    if sum(q[i - width + 1:i + 1]) >= condition.qt * width), 0)
    if start >= end:
        start = end = 0
    trimmed = Read(read.name, read.sequence[start:end], read.qualities[start:end],
                   read.file_name, read.source_sha256)
    return trimmed, {
        "read": read.name,
        "start_1based": start + 1 if end > start else None,
        "end_1based": end if end > start else None,
        "original_bases": n,
        "retained_bases": end - start,
    }


def normalize_sequence(sequence):
    return "".join(base if base in "ACGT" else "N" for base in sequence.upper())


def usable_quality_bases(read: Read):
    """절단 후 Q20 이상인 A/C/G/T의 수. 연속 구간 길이와는 다르다."""
    return sum(base in "ACGT" and quality >= 20
               for base, quality in zip(read.sequence.upper(), read.qualities))


def supported_gap_regions(events):
    """같은 방향의 연속 gap을 한 구간으로 묶고 Q20 지지를 확인한다.

    정렬의 F/R gap 위치를 뜻하며 실제 변이나 오류가 난 read를 확정하지 않는다.
    구간 안의 저품질 염기가 하나의 gap을 여러 구간으로 나누지 않도록 한다.
    """
    counts = {"F": 0, "R": 0}
    previous_kind, previous_column, supported = None, None, False
    for event in events:
        kind, column = event["kind"], event["column"]
        if kind not in ("F Gap", "R Gap"):
            previous_kind, previous_column, supported = None, None, False
            continue
        if kind != previous_kind or column != previous_column + 1:
            supported = False
        if event["support_q"] >= 20 and not supported:
            counts[kind[0]] += 1
            supported = True
        previous_kind, previous_column = kind, column
    return counts


def conflict_cluster(events, alignment_columns):
    """겹침 내부에서 Q20 충돌이 가장 많이 모인 구간을 찾는다.

    폭은 gap을 포함한 정렬 열 수다. QT의 전처리 창과는 별개이며,
    짧은 정렬에서는 정렬 전체를 사용한다. 위치는 1부터 시작한다.
    """
    width = min(CONFLICT_CLUSTER_COLUMNS, alignment_columns)
    positions = [event["column"] for event in events if event["support_q"] >= 20]
    best_count, best_start, left = 0, None, 0
    for right, position in enumerate(positions):
        while position - positions[left] >= width:
            left += 1
        count = right - left + 1
        if count > best_count:
            best_count = count
            best_start = min(positions[left], alignment_columns - width + 1)
    return {"count": best_count, "window_columns": width,
            "start_column": best_start,
            "end_column": best_start + width - 1 if best_start is not None else None}


def _align_overlap(forward: Read, reverse: Read, scores):
    """지정된 점수로 양방향 국소 정렬을 평가한다."""
    if not forward.sequence or not reverse.sequence:
        return None
    match, mismatch, gap_open, gap_extend = scores
    aligner = PairwiseAligner(mode="local", match_score=match, mismatch_score=mismatch,
                              open_gap_score=gap_open, extend_gap_score=gap_extend)
    aligner.wildcard = "N"
    fseq = normalize_sequence(forward.sequence)
    candidates = []
    for complemented in (True, False):
        rseq = normalize_sequence(
            str(Seq(reverse.sequence).reverse_complement()) if complemented else reverse.sequence
        )
        rq = reverse.qualities[::-1] if complemented else reverse.qualities
        try:
            alignment = aligner.align(fseq, rseq)[0]
        except IndexError:
            continue
        matches = mismatches = gaps = ambiguous = q20_matches = q20_conflicts = 0
        q20_gap_conflicts = q20_base_conflicts = 0
        longest_gap = gap_run = 0
        markers, events = [], []
        f_cursor, r_cursor = int(alignment.coordinates[0, 0]), int(alignment.coordinates[1, 0])
        for i, j in zip(*alignment.indices):
            if i < 0 or j < 0:
                gaps += 1
                gap_run += 1
                longest_gap = max(longest_gap, gap_run)
                # Gap의 품질 충돌은 반대 read의 양쪽 인접 염기도 뒷받침해야 한다.
                # 한 read의 높은 Q만으로 저품질 상대 구간을 충돌로 세지 않는다.
                other_q, cursor = (rq, r_cursor) if i >= 0 else (forward.qualities, f_cursor)
                flank_q = min(other_q[cursor - 1:cursor + 1]) if 0 < cursor < len(other_q) else 0
                q = min(forward.qualities[i] if i >= 0 else rq[j], flank_q)
                q20_conflicts += q >= 20
                q20_gap_conflicts += q >= 20
                events.append({"column": len(markers) + 1, "kind": "R Gap" if i >= 0 else "F Gap",
                               "f_position": int(i) + 1 if i >= 0 else None,
                               "r_position": int(j) + 1 if j >= 0 else None,
                               "support_q": int(q)})
                if i >= 0:
                    f_cursor = i + 1
                if j >= 0:
                    r_cursor = j + 1
                markers.append(" ")
                continue
            gap_run = 0
            f_cursor, r_cursor = i + 1, j + 1
            q = min(forward.qualities[i], rq[j])
            if fseq[i] not in "ACGT" or rseq[j] not in "ACGT":
                ambiguous += 1
                q20_conflicts += q >= 20
                q20_base_conflicts += q >= 20
                events.append({"column": len(markers) + 1, "kind": "모호한 염기",
                               "f_position": int(i) + 1, "r_position": int(j) + 1, "support_q": int(q)})
                markers.append("?")
            elif fseq[i] == rseq[j]:
                matches += 1
                q20_matches += q >= 20
                markers.append("|")
            else:
                mismatches += 1
                q20_conflicts += q >= 20
                q20_base_conflicts += q >= 20
                events.append({"column": len(markers) + 1, "kind": "염기 불일치",
                               "f_position": int(i) + 1, "r_position": int(j) + 1, "support_q": int(q)})
                markers.append(".")
        paired = matches + mismatches + ambiguous
        coordinates = alignment.coordinates
        fs, fe, rs, re_ = (int(coordinates[0, 0]), int(coordinates[0, -1]),
                            int(coordinates[1, 0]), int(coordinates[1, -1]))
        # 연결되는 두 끝에 남은 Q20 이상 염기를 센다. 전체 read를 포함하는
        # 정렬도 유효한 배치로 고려하여 길이가 다른 완전 겹침을 배제하지 않는다.
        placements = [
            (sum(q >= 20 for q in forward.qualities[fe:]) + sum(q >= 20 for q in rq[:rs]),
             len(fseq) - fe + rs, "F → R"),
            (sum(q >= 20 for q in rq[re_:]) + sum(q >= 20 for q in forward.qualities[:fs]),
             len(rseq) - re_ + fs, "R → F"),
            (sum(q >= 20 for q in forward.qualities[:fs] + forward.qualities[fe:]),
             fs + len(fseq) - fe, "R 안에 F 포함"),
            (sum(q >= 20 for q in rq[:rs] + rq[re_:]),
             rs + len(rseq) - re_, "F 안에 R 포함"),
        ]
        junction_q20, junction_raw, connection = min(placements, key=lambda p: (p[0], p[1]))
        support = q20_matches + q20_conflicts
        candidates.append({
            "score": float(alignment.score), "paired_bases": paired,
            "gap_identity": 100 * matches / (paired + gaps) if paired + gaps else 0,
            "matches": matches, "mismatches": mismatches, "gaps": gaps,
            "ambiguous": ambiguous, "longest_gap": longest_gap,
            "q20_matches": int(q20_matches), "q20_conflicts": int(q20_conflicts),
            "q20_gap_conflicts": int(q20_gap_conflicts), "q20_base_conflicts": int(q20_base_conflicts),
            "q20_conflict_rate": q20_conflicts / support if support else 0,
            "quality_identity": 100 * q20_matches / support if support else 0,
            "junction_unaligned": int(junction_raw), "junction_q20": int(junction_q20),
            "connection": connection,
            "reverse_orientation": "Reverse complement" if complemented else "원본 방향",
            "forward_start": fs, "forward_end": fe, "reverse_start": rs, "reverse_end": re_,
            "aligned_f": str(alignment[0]), "aligned_r": str(alignment[1]),
            "markers": "".join(markers),
            "events": events,
            "conflict_cluster": conflict_cluster(events, len(markers)),
            "q20_gap_regions": supported_gap_regions(events),
        })
    return max(candidates, key=lambda x: (x["score"], x["paired_bases"], -x["junction_q20"])) if candidates else None


def inspect_overlap(forward: Read, reverse: Read):
    """긴 겹침과 말단 확장을 보수적으로 평가한 겹침을 함께 계산한다.

    보수적 비교의 점수는 제공된 두 실제 정렬의 끝 좌표와 대응했다.
    두 비교는 동일한 절단 서열을 사용하며 별도의 조립 프로그램을 실행하지 않는다.
    """
    overlap = _align_overlap(forward, reverse, (2, -3, -5, -1))
    if overlap is not None:
        overlap["core_alignment"] = _align_overlap(forward, reverse, (1, -2, -4, -3))
    return overlap


def terminal_gap_pair(overlap):
    """연결 끝까지 정렬된 단염기 gap 쌍에 대한 경험적 예외.

    추가 gap·불일치·모호한 염기·정렬 밖 연결 말단이 있으면 적용하지 않는다.
    같은 방향의 두 gap은 한 gap이 정렬 끝의 제한된 범위에 있을 때만 허용한다.
    관측 사례에서 얻은 보정이며 실제 조립 알고리즘을 복원한 규칙은 아니다.
    """
    eligible = (overlap["connection"] in ("F → R", "R → F")
            and overlap["junction_unaligned"] == 0
            and overlap["gaps"] == overlap["q20_gap_conflicts"] == 2
            and overlap["longest_gap"] == 1
            and overlap["mismatches"] == overlap["ambiguous"] == 0
            and overlap["q20_base_conflicts"] == 0)
    if not eligible:
        return False
    counts = overlap["q20_gap_regions"]
    if counts == {"F": 1, "R": 1}:
        return True
    if sorted(counts.values()) != [0, 2]:
        return False
    columns = len(overlap["markers"])
    return any(min(e["column"], columns - e["column"] + 1) <= TERMINAL_GAP_COLUMNS
               for e in overlap["events"])


def unresolved_low_quality_end(overlap):
    """말단의 저품질 gap·염기 불일치와 미정렬 연결 경계가 함께 남는지 확인."""
    if overlap["junction_unaligned"] == 0:
        return False
    columns = len(overlap["markers"])
    for start, end in ((1, min(columns, LOW_QUALITY_END_COLUMNS)),
                       (max(1, columns - LOW_QUALITY_END_COLUMNS + 1), columns)):
        events = [e for e in overlap["events"]
                  if start <= e["column"] <= end and e["support_q"] < 20]
        if (any(e["kind"] in ("F Gap", "R Gap") for e in events)
                and any(e["kind"] in ("염기 불일치", "모호한 염기") for e in events)):
            return True
    return False


def terminal_realign_supported(overlap):
    """실제 염기를 바꾸지 않고 짧은 말단만 제외한 대체 정렬의 제한적 인정.

    내부 gap 배치나 방향이 달라진 정렬, 고품질 충돌을 숨기는 절단은 제외한다.
    이 허용 범위는 검증 자료로 보정한 값이며 보편적인 조립 기준이 아니다.
    """
    core = overlap.get("core_alignment")
    if not core or overlap["q20_base_conflicts"] == 0:
        return False
    if (core["connection"] != overlap["connection"]
            or core["reverse_orientation"] != overlap["reverse_orientation"]
            or core["connection"] not in ("F → R", "R → F")
            or core["mismatches"] or core["ambiguous"]
            or core["paired_bases"] < MIN_OVERLAP_BASES
            or core["q20_matches"] < MIN_Q20_MATCHES
            or core["gap_identity"] < MIN_GAP_IDENTITY
            or core["junction_q20"] > MAX_Q20_JUNCTION
            or core["longest_gap"] > MAX_GAP_RUN):
        return False
    if any(e["support_q"] >= CORE_CONFLICT_QUALITY_LIMIT for e in core["events"]):
        return False
    removed = len(overlap["markers"]) - len(core["markers"])
    if not 1 <= removed <= MAX_CORE_TRIM_COLUMNS:
        return False
    # 두 정렬 문자열에서 같은 연속 구간만 허용한다. 내부 재배치는 제외한다.
    for left in range(removed + 1):
        right = left + len(core["markers"])
        if (overlap["aligned_f"][left:right] == core["aligned_f"]
                and overlap["aligned_r"][left:right] == core["aligned_r"]):
            excluded = [e for e in overlap["events"] if not left < e["column"] <= right]
            if (any(e["kind"] in ("염기 불일치", "모호한 염기") and e["support_q"] >= 20
                    for e in excluded)
                    and all(e["support_q"] < CORE_CONFLICT_QUALITY_LIMIT for e in excluded)):
                return True
    return False


def predict_contig(lengths, overlap, quality_bases=None):
    """품질이 뒷받침하는 충돌과 낮은 품질의 차이를 분리해 평가한다.

    한 개의 Q20 지지 단염기 gap은 짧은 겹침에서 비율만으로 기각하지 않는다.
    연결 끝까지 정렬된 단염기 gap 쌍과 짧은 말단 제외에 제한적 비율 예외를 적용한다.
    그 밖의 복수 Q20 충돌과 Q20 염기 불일치는 기존 충돌 비율 기준을 적용한다.
    긴 일치 구간이 국소 충돌을 희석하지 않도록 충돌 집중도도 검사한다.
    Q20 유효 염기 수, 양쪽 반복 gap 검사, 말단 gap 쌍 예외는 검증용 보정이다.
    모든 조건에 동일한 규칙을 사용하며, 실측 성공 확률로 환산하지 않는다.
    """
    if min(lengths) < MIN_READ_BASES:
        short = ", ".join(name for name, length in zip(("F", "R"), lengths) if length < MIN_READ_BASES)
        return {"label": "No contig", "state": "insufficient", "generated": False,
                "reasons": [f"{short}의 남은 서열이 {MIN_READ_BASES} bp 미만입니다."]}
    if quality_bases is not None and min(quality_bases) < MIN_Q20_READ_BASES:
        reasons = [f"{name}의 Q20 이상 유효 염기가 {count} bp로, "
                   f"기준 {MIN_Q20_READ_BASES} bp 미만입니다."
                   for name, count in zip(("F", "R"), quality_bases)
                   if count < MIN_Q20_READ_BASES]
        return {"label": "No contig", "state": "insufficient", "generated": False,
                "reasons": reasons}
    reasons = []
    terminal_gap_exception_applied = False
    terminal_realign_applied = False
    if overlap is None or overlap["paired_bases"] < MIN_OVERLAP_BASES:
        reasons.append(f"연결에 필요한 겹침이 {MIN_OVERLAP_BASES} bp 미만입니다.")
    if overlap is not None:
        if overlap["gap_identity"] < MIN_GAP_IDENTITY:
            reasons.append(f"Gap 포함 일치율이 {MIN_GAP_IDENTITY:g}% 미만입니다.")
        if overlap["q20_matches"] < MIN_Q20_MATCHES:
            reasons.append(f"양쪽 Q20 이상 일치가 {MIN_Q20_MATCHES} bp 미만입니다.")
        single_indel = (overlap["q20_gap_conflicts"] == 1
                        and overlap["q20_base_conflicts"] == 0 and overlap["longest_gap"] == 1)
        terminal_gap_exception_applied = (
            overlap["q20_conflict_rate"] > MAX_Q20_CONFLICT_RATE
            and terminal_gap_pair(overlap)
        )
        terminal_realign_applied = (overlap["q20_conflict_rate"] > MAX_Q20_CONFLICT_RATE
                                    and terminal_realign_supported(overlap))
        if (overlap["q20_conflict_rate"] > MAX_Q20_CONFLICT_RATE
                and not single_indel and not terminal_gap_exception_applied
                and not terminal_realign_applied):
            reasons.append(f"복수 Gap 또는 염기 불일치의 Q20 충돌 비율이 {MAX_Q20_CONFLICT_RATE:.0%}를 초과합니다.")
        cluster = overlap["conflict_cluster"]
        if cluster["count"] >= MIN_CLUSTER_Q20_CONFLICTS:
            reasons.append(
                f"정렬 {cluster['start_column']}–{cluster['end_column']} 위치의 "
                f"{cluster['window_columns']}개 정렬 열 안에 Q20 충돌 "
                f"{cluster['count']}개가 집중되어 있습니다 "
                f"(기준 {MIN_CLUSTER_Q20_CONFLICTS}개 이상)."
            )
        gap_regions = overlap["q20_gap_regions"]
        if min(gap_regions.values()) >= MIN_BIDIRECTIONAL_GAP_REGIONS:
            reasons.append(
                f"Q20으로 지지되는 Gap이 F 정렬 {gap_regions['F']}구간, "
                f"R 정렬 {gap_regions['R']}구간에 반복됩니다 "
                f"(양쪽 각각 {MIN_BIDIRECTIONAL_GAP_REGIONS}구간 이상: 검증용 보정)."
            )
        if overlap["junction_q20"] > MAX_Q20_JUNCTION:
            reasons.append(f"연결 경계에 정렬되지 않은 Q20 이상 염기가 {MAX_Q20_JUNCTION} bp를 초과합니다.")
        if overlap["longest_gap"] > MAX_GAP_RUN:
            reasons.append(f"연속 Gap이 {MAX_GAP_RUN} bp를 초과합니다.")
        if unresolved_low_quality_end(overlap):
            reasons.append(f"겹침 끝 {LOW_QUALITY_END_COLUMNS}개 정렬 열에 저품질 Gap과 "
                           "염기 불일치가 함께 있고, 연결 경계의 미정렬 염기도 남아 있습니다 "
                           "(검증용 보정).")
    if reasons:
        return {"label": "Contig2", "state": "separate", "generated": False, "reasons": reasons}
    passed = ["겹침 길이·일치율·품질·연결 경계 기준을 통과했습니다."]
    if terminal_gap_exception_applied:
        passed.append("연결 끝까지 정렬된 단염기 Gap 두 개의 방향과 말단 위치 조건으로 "
                      "충돌 비율 예외를 적용했습니다 (검증용 보정).")
    if terminal_realign_applied:
        passed.append(f"말단 {MAX_CORE_TRIM_COLUMNS}개 정렬 열 이내를 제외한 비교에서 "
                      f"염기 불일치가 해소되고 남은 Gap의 지지 Q가 {CORE_CONFLICT_QUALITY_LIMIT} 미만입니다 "
                      "(충돌 비율 예외: 검증용 보정).")
    return {"label": "Contig 생성", "state": "joined", "generated": True,
            "reasons": passed}


def analyze_reads(reads):
    if len(reads) != 2:
        raise ValueError("F/R 두 개의 AB1 파일이 필요합니다.")
    rows = []
    for condition in CONDITIONS:
        processed = [trim_read(read, condition) for read in reads]
        forward, reverse = (item[0] for item in processed)
        lengths = [len(forward.sequence), len(reverse.sequence)]
        quality_bases = [usable_quality_bases(forward), usable_quality_bases(reverse)]
        overlap = inspect_overlap(forward, reverse)
        rows.append({
            "condition": asdict(condition), "lengths": lengths, "quality_bases": quality_bases,
            "trim_metadata": [item[1] for item in processed],
            "overlap": overlap, "prediction": predict_contig(lengths, overlap, quality_bases),
        })
    return rows


def summary_rows(rows):
    records = []
    for row in rows:
        overlap = row["overlap"] or {}
        prediction = row["prediction"]
        records.append({
            "조건": row["condition"]["label"], "예측 결과": prediction["label"],
            "Contig 생성 예측": "생성" if prediction["generated"] else "생성 안 됨",
            "F 길이 (bp)": row["lengths"][0], "R 길이 (bp)": row["lengths"][1],
            "F Q20 유효 염기 (bp)": row["quality_bases"][0],
            "R Q20 유효 염기 (bp)": row["quality_bases"][1],
            "겹침 (bp)": overlap.get("paired_bases", 0),
            "Gap 포함 일치율 (%)": round(overlap.get("gap_identity", 0), 2),
            "Q20 지지 일치율 (%)": round(overlap.get("quality_identity", 0), 2),
            "Q20 염기 충돌": overlap.get("q20_base_conflicts", 0),
            "Q20 Gap 충돌": overlap.get("q20_gap_conflicts", 0),
            "F Q20 Gap 구간": overlap.get("q20_gap_regions", {}).get("F", 0),
            "R Q20 Gap 구간": overlap.get("q20_gap_regions", {}).get("R", 0),
            "연결 경계 잔여 (bp)": overlap.get("junction_unaligned", 0),
            "연결 경계 Q20 잔여 (bp)": overlap.get("junction_q20", 0),
            "말단 보수적 비교 겹침 (bp)": (overlap.get("core_alignment") or {}).get("paired_bases", 0),
            "말단 보수적 비교 Gap (bp)": (overlap.get("core_alignment") or {}).get("gaps", 0),
            "국소 Q20 충돌 최대 개수": overlap.get("conflict_cluster", {}).get("count", 0),
            "국소 충돌 구간 시작 (정렬 위치)": overlap.get("conflict_cluster", {}).get("start_column"),
            "국소 충돌 구간 끝 (정렬 위치)": overlap.get("conflict_cluster", {}).get("end_column"),
            "판정 근거": " / ".join(prediction["reasons"]), "분석 버전": APP_VERSION,
        })
    return records


def alignment_preview(overlap, width=90):
    lines = []
    for start in range(0, len(overlap["markers"]), width):
        end = start + width
        lines.extend(["F  " + overlap["aligned_f"][start:end],
                      "   " + overlap["markers"][start:end],
                      "R  " + overlap["aligned_r"][start:end], ""])
    return "\n".join(lines)


def quality_summary(reads):
    return pd.DataFrame([{
        "Read": r.name, "파일": r.file_name, "길이": len(r.sequence),
        "평균 Q": round(sum(r.qualities) / len(r.qualities), 2) if r.qualities else 0,
        "Q20 이상 bp": sum(q >= 20 for q in r.qualities),
        "Q30 이상 bp": sum(q >= 30 for q in r.qualities),
    } for r in reads])


def render_result(rows, reads):
    st.subheader("Contig 생성 예측")
    joined = [row["condition"]["label"] for row in rows if row["prediction"]["generated"]]
    if joined:
        st.success("생성 예상 조건: " + ", ".join(joined))
    else:
        st.info("네 조건에서 Contig 생성이 예상되는 조건은 없습니다.")

    for start in (0, 2):
        for column, row in zip(st.columns(2), rows[start:start + 2]):
            with column, st.container(border=True):
                prediction = row["prediction"]
                st.markdown(f"### {row['condition']['label']}")
                st.metric("예측 결과", prediction["label"])
                if prediction["state"] == "joined":
                    st.success("생성 예상")
                elif prediction["state"] == "separate":
                    st.warning("생성 안 됨 · F/R 분리 예상")
                else:
                    st.warning("생성 안 됨 · 유효 서열 부족")
                st.write(f"F **{row['lengths'][0]} bp** · R **{row['lengths'][1]} bp**")
                overlap = row["overlap"]
                if overlap:
                    st.write(f"겹침 **{overlap['paired_bases']} bp** · 일치율 **{overlap['gap_identity']:.1f}%**")
                else:
                    st.write("겹침 없음")

    with st.expander("판정 근거·정렬 상세", expanded=False):
        selected = st.selectbox("상세 조건", [c.label for c in CONDITIONS], key="details_condition")
        row = next(row for row in rows if row["condition"]["label"] == selected)
        st.write("**" + row["prediction"]["label"] + " 예측 근거**")
        for reason in row["prediction"]["reasons"]:
            st.write("• " + reason)
        st.dataframe(pd.DataFrame(row["trim_metadata"]).rename(columns={
            "read": "Read", "start_1based": "시작 위치", "end_1based": "끝 위치",
            "original_bases": "원본 길이", "retained_bases": "남은 길이",
        }), hide_index=True, use_container_width=True)
        st.write(f"Q20 이상 유효 염기: F {row['quality_bases'][0]} bp · "
                 f"R {row['quality_bases'][1]} bp")
        overlap = row["overlap"]
        if overlap:
            st.write(f"R 처리: {overlap['reverse_orientation']} · 배치: {overlap['connection']}")
            st.write(f"Gap {overlap['gaps']} bp · 최장 Gap {overlap['longest_gap']} bp · "
                     f"Q20 일치 {overlap['q20_matches']} bp · "
                     f"Q20 염기 충돌 {overlap['q20_base_conflicts']}개 / Gap 충돌 {overlap['q20_gap_conflicts']}개")
            st.write(f"연결 경계의 정렬 밖 염기 {overlap['junction_unaligned']} bp "
                     f"(Q20 이상 {overlap['junction_q20']} bp)")
            st.write(f"Q20 지지 Gap 구간: F 정렬 {overlap['q20_gap_regions']['F']}개 · "
                     f"R 정렬 {overlap['q20_gap_regions']['R']}개")
            cluster = overlap["conflict_cluster"]
            if cluster["count"]:
                st.write(f"국소 Q20 충돌 최대 {cluster['count']}개 · "
                         f"정렬 위치 {cluster['start_column']}–{cluster['end_column']} "
                         f"({cluster['window_columns']}개 정렬 열)")
            st.code(alignment_preview(overlap), language=None)
            core = overlap.get("core_alignment")
            if core:
                st.write(f"말단 확장을 보수적으로 평가한 비교: 겹침 {core['paired_bases']} bp · "
                         f"Gap {core['gaps']} bp · 연결 경계 잔여 {core['junction_unaligned']} bp")
                st.code(alignment_preview(core), language=None)
            if overlap["events"]:
                st.dataframe(pd.DataFrame(overlap["events"]).rename(columns={
                    "column": "정렬 위치", "kind": "차이 유형", "f_position": "F 위치",
                    "r_position": "정렬 방향 R 위치", "support_q": "충돌 지지 Q",
                }), hide_index=True, use_container_width=True)
        st.caption(
            f"공통 예측 기준: F/R 각각 {MIN_READ_BASES} bp 이상, 겹침 {MIN_OVERLAP_BASES} bp 이상, "
            f"Gap 포함 일치율 {MIN_GAP_IDENTITY:g}% 이상, 양쪽 Q20 일치 {MIN_Q20_MATCHES} bp 이상, "
            f"Q20 충돌 {MAX_Q20_CONFLICT_RATE:.0%} 이하, 연결 경계 Q20 잔여 {MAX_Q20_JUNCTION} bp 이하, "
            f"최장 Gap {MAX_GAP_RUN} bp 이하. Q20 염기 불일치 없이 단염기 Gap 하나만 "
            "Q20으로 뒷받침되는 경우는 충돌 비율만으로 기각하지 않습니다. "
            f"겹침 내 {CONFLICT_CLUSTER_COLUMNS}개 정렬 열 안에 Q20 충돌 "
            f"{MIN_CLUSTER_Q20_CONFLICTS}개 이상이 집중되면 통과시키지 않습니다. "
            f"검증용 보정: 한쪽의 Q20 이상 A/C/G/T가 {MIN_Q20_READ_BASES} bp 미만이면 "
            f"No contig, 양쪽 정렬에 Q20 지지 Gap이 각각 {MIN_BIDIRECTIONAL_GAP_REGIONS}구간 "
            "이상이면 Contig2로 예측합니다. "
            "연결되는 두 끝까지 정렬되고, 차이가 양쪽의 Q20 단염기 Gap 한 개씩뿐이면 "
            "충돌 비율 예외를 적용하되 나머지 기준은 그대로 검사합니다. "
            f"같은 방향의 단염기 Gap 두 개는 한 개가 정렬 끝 {TERMINAL_GAP_COLUMNS}개 열 안에 "
            "있는 경우에만 해당 예외를 적용합니다. "
            f"짧은 말단 제외({MAX_CORE_TRIM_COLUMNS}개 열 이내)의 대체 정렬은 Q30 이상 충돌을 "
            "숨기지 않는 경우에 한해 평가합니다. "
            f"말단 {LOW_QUALITY_END_COLUMNS}개 열에 저품질 Gap·염기 불일치와 미정렬 연결 경계가 "
            "함께 남는 경우는 Contig2로 예측합니다. "
            "제공된 사례로 보정한 예측 기준이며 새 샘플의 정확도는 별도 검증이 필요합니다."
        )

    with st.expander("AB1 원본 Quality", expanded=False):
        st.dataframe(quality_summary(reads), hide_index=True, use_container_width=True)
        for column, read in zip(st.columns(2), reads):
            with column:
                st.caption(read.file_name)
                frame = pd.DataFrame({"염기 위치": range(1, len(read.qualities) + 1), "Quality": read.qualities})
                st.line_chart(frame.set_index("염기 위치"))

    with st.expander("재반응 검토", expanded=False):
        if joined:
            st.write("생성 예상 조건의 겹침 구간과 원본 파형을 확인해 주세요.")
        else:
            st.write("각 조건에서 남은 F/R 길이와 겹침 구간을 먼저 확인해 주세요. "
                     "유효 서열 부족은 해당 방향의 Quality, 충돌이 많은 경우는 양쪽 파형을 함께 검토합니다.")
        st.caption("정렬의 Gap이나 불일치만으로 혼합 또는 InDel을 확정할 수는 없습니다.")

    st.download_button("분석 결과 다운로드 (CSV)",
        pd.DataFrame(summary_rows(rows)).to_csv(index=False).encode("utf-8-sig"),
        file_name="contig_prediction.csv", mime="text/csv", key="result_download")


def main():
    st.set_page_config(page_title="Contig Simulator", page_icon="🧬", layout="wide")
    st.title("🧬 Contig Simulator")
    st.write("F/R AB1 파일을 업로드하면 네 조건의 Contig 생성 여부를 예측합니다.")
    st.caption(f"{APP_VERSION} · 검증용 예측이며 실제 조립 결과와 다를 수 있습니다.")
    left, right = st.columns(2)
    with left:
        f_upload = st.file_uploader("Forward AB1", type=["ab1"], key="forward_file")
    with right:
        r_upload = st.file_uploader("Reverse AB1", type=["ab1"], key="reverse_file")
    ready = f_upload is not None and r_upload is not None
    upload_bytes = [f_upload.getvalue(), r_upload.getvalue()] if ready else []
    fingerprint = sha256(json.dumps({
        "version": APP_VERSION, "hashes": [sha256(data) for data in upload_bytes],
        "names": [f_upload.name, r_upload.name] if ready else [],
    }, sort_keys=True).encode())
    result_key = "ab1_comparison_v6"
    clicked = st.button("Contig 분석", type="primary", disabled=not ready, key="analyze_button")
    if clicked:
        st.session_state.pop(result_key, None)
        try:
            with st.spinner("네 조건의 Contig 생성 여부를 분석하고 있습니다…"):
                reads = (read_ab1(upload_bytes[0], f_upload.name, "F"),
                         read_ab1(upload_bytes[1], r_upload.name, "R"))
                rows = analyze_reads(reads)
                st.session_state[result_key] = (fingerprint, reads, rows)
        except (ValueError, OSError) as exc:
            st.error(str(exc))
    result = st.session_state.get(result_key)
    if result:
        saved_fingerprint, reads, rows = result
        if saved_fingerprint != fingerprint:
            st.info("파일이 변경되었습니다. Contig 분석을 다시 실행해 주세요.")
            return
        render_result(rows, reads)


if __name__ == "__main__":
    main()
