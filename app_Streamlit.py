"""AB1 기반 Contig 생성 예측 도구.

실행: python -m streamlit run "app_Streamlit(5).py"
설치(Python 3.10 이상): python -m pip install streamlit pandas biopython

여섯 조건에 동일한 정렬·품질 판정 규칙을 적용한다. 결과는 경험적 예측이며
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

APP_VERSION = "2026.10.28-ab1-v18"
RERUN_RULE_VERSION = "quality-review-v1"
# 재반응 전후 검증 전의 보수적 검토 기준. 성공 확률이나 사내 SOP가 아니다.
RERUN_REFERENCE_FRACTION = 0.80
RERUN_WEAK_FRACTION = 0.50
RERUN_MIN_DIFFERENCES = 3
RERUN_MIN_SUPPORTING_CONDITIONS = 2
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
MIN_GAP_PAIR_OUTER_Q20 = 10
MAX_SHORT_GAP_PAIR_BASES = 100
MIN_MIXED_GAP_Q20_RUN = 30
TERMINAL_GAP_COLUMNS = 10
LOW_QUALITY_END_COLUMNS = 12
MAX_CORE_TRIM_COLUMNS = 5
CORE_CONFLICT_QUALITY_LIMIT = 30
# 2026-10-20 실제 ACE 대조에서 확인한 짧은 겹침의 제한적 허용 범위.
# Gap 지지 Q<10인 불안정한 말단 및 Q30 이상 충돌은 이 보정에서 제외한다.
SHORT_GAP_MIN_SUPPORT_Q = 10
SHORT_GAP_MAX_SUPPORT_Q = 30  # exclusive
MAX_SHORT_LOW_Q_JUNCTION = 5
# 같은 염기 반복 안의 동점 Gap 배치와 짧은 말단 제외를 함께 확인한다.
MIN_CLIPPED_CORE_Q20_MATCHES = 30
# 짧은 연결에서 반복되는 같은 방향 단염기 Gap과 Q20 염기 차이의 동반 검사.
MIN_PERSISTENT_SHORT_GAP_REGIONS = 3
INTERNAL_REVIEW_MIN_JUNCTION_Q20 = 100
INTERNAL_REVIEW_MAX_Q20_FRACTION = 0.20
# 긴 겹침의 다수가 Q20 미만일 때만 사용하는 경험적 보정.
# 전체 판정 기준을 낮추지 않고, 강한 품질 충돌과 연결 위치를 별도로 검사한다.
LONG_OVERLAP_MIN_IDENTITY = 70.0
LONG_OVERLAP_MIN_SCORE = 100.0
LONG_OVERLAP_MIN_Q20_MATCHES = 10
LONG_OVERLAP_MAX_Q20_FRACTION = 0.5
LONG_OVERLAP_MAX_Q30_CONFLICTS = 2
LONG_OVERLAP_MAX_JUNCTION = 50
LONG_OVERLAP_MAX_Q20_JUNCTION = 20


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
    Condition("20", 20, False, None),
    Condition("30", 30, False, None),
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


def gap_outer_q20_anchors(events, q20_agreements):
    """첫 gap 바로 앞과 마지막 gap 바로 뒤의 연속 Q20 일치 길이.

    정렬 양끝 방향으로 센다. 내부 일치나 떨어진 일치를 합산하지 않으며,
    한쪽이라도 Q20 미만이거나 gap·불일치인 열에서 중단한다.
    """
    positions = [e["column"] - 1 for e in events if e["kind"] in ("F Gap", "R Gap")]
    anchors = {"left": 0, "right": 0}
    if not positions:
        return anchors
    for side, columns in (
        ("left", range(min(positions) - 1, -1, -1)),
        ("right", range(max(positions) + 1, len(q20_agreements))),
    ):
        for column in columns:
            if not q20_agreements[column]:
                break
            anchors[side] += 1
    return anchors


def opposing_gap_pair_evidence(events, q20_agreements):
    """반대 방향 단염기 gap 두 개 사이의 품질 지지를 요약한다.

    Q20 미만 gap도 구조상 차이로 유지한다. 연속 지지는 두 gap 사이만 세며,
    떨어진 일치 구간, gap 바깥쪽 일치, 한쪽만 높은 품질은 합산하지 않는다.
    """
    gaps = [e for e in events if e["kind"] in ("F Gap", "R Gap")]
    if len(gaps) != 2 or {e["kind"] for e in gaps} != {"F Gap", "R Gap"}:
        return None
    left, right = sorted(e["column"] - 1 for e in gaps)
    if right - left <= 1:
        return None
    longest = current = 0
    for agreement in q20_agreements[left + 1:right]:
        current = current + 1 if agreement else 0
        longest = max(longest, current)
    return {"q20_gap_count": sum(e["support_q"] >= 20 for e in gaps),
            "between_columns": right - left - 1, "between_q20_run": longest}


def persistent_weak_terminal_gap(overlap):
    """보수적 비교에도 남는 반대 방향 Gap 쌍의 약한 말단 지지를 확인한다.

    짧은 연결에 Q20 이상 Gap 하나와 Q20 미만 Gap 하나가 함께 있고,
    같은 read 좌표의 두 Gap이 보수적 정렬에도 유지되는 경우만 검사한다.
    저품질 Gap 바깥의 짧은 말단에서 연속 Q20 지지가 부족한 패턴이다.
    실제 출력 비교를 근거로 기존 길이·말단·연속 지지 범위를 재사용한
    경험적 보정이며 실제 조립기의 내부 판정 규칙을 재현한 것은 아니다.
    """
    if not overlap or overlap["paired_bases"] > MAX_SHORT_GAP_PAIR_BASES:
        return None
    pair, core = overlap.get("opposing_gap_pair"), overlap.get("core_alignment")
    if (not pair or pair["q20_gap_count"] != 1 or not core
            or overlap["connection"] not in ("F → R", "R → F")
            or core["connection"] != overlap["connection"]
            or core["reverse_orientation"] != overlap["reverse_orientation"]):
        return None
    core_pair = core.get("opposing_gap_pair")
    if not core_pair or core_pair["q20_gap_count"] != 1:
        return None
    gaps = [e for e in core["events"] if e["kind"] in ("F Gap", "R Gap")]
    original = [e for e in overlap["events"] if e["kind"] in ("F Gap", "R Gap")]
    def signature(event):
        return tuple(event[key] for key in ("kind", "f_position", "r_position", "support_q"))

    if sorted(map(signature, gaps)) != sorted(map(signature, original)):
        return None
    low = next(e for e in gaps if e["support_q"] < 20)
    left = low["column"] == min(e["column"] for e in gaps)
    outer_columns = low["column"] - 1 if left else len(core["markers"]) - low["column"]
    anchor = core["gap_outer_q20_anchors"]["left" if left else "right"]
    if outer_columns > LOW_QUALITY_END_COLUMNS or anchor >= MIN_GAP_PAIR_OUTER_Q20:
        return None
    return {"low_gap_support_q": low["support_q"], "low_gap_core_column": low["column"],
            "outer_columns": outer_columns, "outer_q20_run": anchor,
            "between_q20_run": core_pair["between_q20_run"],
            "core_paired_bases": core["paired_bases"], "core_gaps": core["gaps"]}


def short_gap_pair_reasons(overlap):
    """짧은 연결에서 저품질 gap을 포함한 반대 방향 쌍을 검토한다.

    제공된 실측 사례에 근거한 보정이며 일반적인 조립 기준은 아니다.
    단일 gap, 같은 방향 gap, 긴 겹침과 포함 정렬에는 이 보정을 적용하지 않는다.
    """
    pair = overlap.get("opposing_gap_pair")
    if (pair is None or overlap["paired_bases"] > MAX_SHORT_GAP_PAIR_BASES
            or overlap["connection"] not in ("F → R", "R → F")):
        return []
    if pair["q20_gap_count"] == 0 and overlap["junction_unaligned"] > 0:
        return ["짧은 겹침에 서로 반대 방향의 저품질 Gap 두 개가 있고, "
                f"연결 끝에 미정렬 염기 {overlap['junction_unaligned']} bp가 남아 있습니다 "
                "(검증용 보정)."]
    if pair["q20_gap_count"] == 1 and pair["between_q20_run"] < MIN_MIXED_GAP_Q20_RUN:
        return ["짧은 겹침에 Q20 이상 Gap과 저품질 Gap이 반대 방향으로 있으며, "
                f"두 Gap 사이의 연속 Q20 일치가 {pair['between_q20_run']} bp로 "
                f"기준 {MIN_MIXED_GAP_Q20_RUN} bp 미만입니다 (검증용 보정)."]
    terminal = persistent_weak_terminal_gap(overlap)
    if terminal:
        return ["보수적 비교에도 Q20 이상 Gap과 저품질 Gap이 반대 방향으로 남습니다. "
                f"저품질 Gap의 지지 Q는 {terminal['low_gap_support_q']}이고, "
                f"그 바깥 말단 {terminal['outer_columns']}개 정렬 열에서 연속 Q20 일치는 "
                f"{terminal['outer_q20_run']} bp로 기준 {MIN_GAP_PAIR_OUTER_Q20} bp 미만입니다 "
                "(검증용 보정)."]
    return []


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


def equivalent_single_gap_support(alignment, fseq, rseq, fq, rq):
    """일치하는 동일 염기 안에서 한 칸 Gap을 옮긴 동점 배치의 Q 범위.

    원래 정렬·서열·Q는 유지한다. 맞는 염기와 단염기 Gap의 위치만 교환하며,
    불일치·다른 염기·다른 Gap을 넘어 이동하지 않는다. 가능한 위치가 여럿인
    경우 처음 반환된 정렬의 높은 Q만으로 강한 충돌을 확정하지 않기 위한 근거다.
    """
    a,b = str(alignment[0]),str(alignment[1])
    fc,rc = map(int, alignment.coordinates[:,0])
    evidence = []
    for col,(i,j) in enumerate(zip(*alignment.indices)):
        if (i < 0) != (j < 0):
            missing,other = (a,b) if i < 0 else (b,a)
            gap_q,base_q = (fq,rq) if i < 0 else (rq,fq)
            cursor,base_pos = (fc,int(j)) if i < 0 else (rc,int(i))
            base = other[col]
            if ((col == 0 or missing[col-1] != '-')
                    and (col+1 == len(a) or missing[col+1] != '-') and base in 'ACGT'):
                left=right=col
                while left>0 and missing[left-1]==other[left-1]==base:
                    left-=1
                while right+1<len(a) and missing[right+1]==other[right+1]==base:
                    right+=1
                supports=[]
                for position in range(left,right+1):
                    shift=position-col
                    at=cursor+shift
                    flank=min(gap_q[at-1:at+1]) if 0<at<len(gap_q) else 0
                    supports.append({'column':position+1,'support_q':int(min(base_q[base_pos+shift],flank))})
                evidence.append({'column':col+1,'kind':'F Gap' if i<0 else 'R Gap',
                                 'min_support_q':min(x['support_q'] for x in supports),
                                 'max_support_q':max(x['support_q'] for x in supports),
                                 'placements':supports})
        if i>=0:fc=int(i)+1
        if j>=0:rc=int(j)+1
    return evidence


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
        markers, events, q20_agreements = [], [], []
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
                q20_agreements.append(False)
                continue
            gap_run = 0
            f_cursor, r_cursor = i + 1, j + 1
            q = min(forward.qualities[i], rq[j])
            q20_agreements.append(fseq[i] in "ACGT" and fseq[i] == rseq[j] and q >= 20)
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
            "equivalent_gap_support": equivalent_single_gap_support(alignment, fseq, rseq, forward.qualities, rq),
            "conflict_cluster": conflict_cluster(events, len(markers)),
            "q20_gap_regions": supported_gap_regions(events),
            "gap_outer_q20_anchors": gap_outer_q20_anchors(events, q20_agreements),
            "opposing_gap_pair": opposing_gap_pair_evidence(events, q20_agreements),
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
    서로 반대 방향의 gap은 바깥쪽 연속 Q20 일치가 한쪽 이상 충분해야 한다.
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
        return max(overlap["gap_outer_q20_anchors"].values()) >= MIN_GAP_PAIR_OUTER_Q20
    if sorted(counts.values()) != [0, 2]:
        return False
    columns = len(overlap["markers"])
    return any(min(e["column"], columns - e["column"] + 1) <= TERMINAL_GAP_COLUMNS
               for e in overlap["events"])


def terminal_low_quality_gap_regions(overlap):
    """미정렬 연결 끝과 함께 남는 복수 저품질 말단 gap 구간.

    같은 방향의 연속 gap은 길이와 관계없이 한 구간으로 센다. 구간 전체가
    Q20 미만이고 같은 말단 검사창 안에 들어가야 한다. 높은 품질의 일부를
    창 밖으로 잘라 저품질 gap으로 바꾸거나, 한 긴 gap을 여러 개로 세지 않는다.
    제공된 실제 출력으로 보완한 경험적 검사이며 조립 알고리즘의 복원은 아니다.
    """
    if overlap is None or overlap["junction_unaligned"] == 0:
        return []
    regions = []
    for event in overlap["events"]:
        if event["kind"] not in ("F Gap", "R Gap"):
            continue
        if (regions and regions[-1]["kind"] == event["kind"]
                and regions[-1]["end_column"] + 1 == event["column"]):
            region = regions[-1]
            region["end_column"] = event["column"]
            region["max_support_q"] = max(region["max_support_q"], event["support_q"])
        else:
            regions.append({"kind": event["kind"], "start_column": event["column"],
                            "end_column": event["column"], "max_support_q": event["support_q"]})
    columns = len(overlap["markers"])
    for start, end in ((1, min(columns, LOW_QUALITY_END_COLUMNS)),
                       (max(1, columns - LOW_QUALITY_END_COLUMNS + 1), columns)):
        low = [region for region in regions
               if start <= region["start_column"] <= region["end_column"] <= end
               and region["max_support_q"] < 20]
        if len(low) >= 2:
            return low
    return []


def unresolved_low_quality_end(overlap):
    """말단의 복합 저품질 차이와 미정렬 연결 경계가 함께 남는지 확인."""
    if overlap["junction_unaligned"] == 0:
        return False
    if terminal_low_quality_gap_regions(overlap):
        return True
    columns = len(overlap["markers"])
    for start, end in ((1, min(columns, LOW_QUALITY_END_COLUMNS)),
                       (max(1, columns - LOW_QUALITY_END_COLUMNS + 1), columns)):
        events = [e for e in overlap["events"]
                  if start <= e["column"] <= end and e["support_q"] < 20]
        if (any(e["kind"] in ("F Gap", "R Gap") for e in events)
                and any(e["kind"] in ("염기 불일치", "모호한 염기") for e in events)):
            return True
    return False


def unsupported_terminal_extension(overlap):
    """짧은 겹침에서 품질 지지 없이 Gap만 늘리는 말단 확장을 찾는다.

    보수적 정렬의 서열·좌표가 원래 정렬 안에 있고 내부 차이가 유지될 때만 비교한다.
    제외되는 말단 차이는 모두 Q20 미만이어야 하며, 확장으로 양쪽 Q20
    일치가 하나도 늘지 않고 연결 경계도 미정렬 상태일 때만 표시한다.
    기존의 짧은 겹침·말단 검사 범위를 재사용한 경험적 검사다.
    실제 조립기의 말단 처리 규칙을 재현하거나 염기를 다시 절단하지 않는다.
    """
    if not overlap:
        return None
    core = overlap.get("core_alignment")
    if (not core or overlap["paired_bases"] > MAX_SHORT_GAP_PAIR_BASES
            or overlap["connection"] not in ("F → R", "R → F")
            or overlap["junction_unaligned"] == 0
            or core["connection"] != overlap["connection"]
            or core["reverse_orientation"] != overlap["reverse_orientation"]
            or core["paired_bases"] < MIN_OVERLAP_BASES
            or core["q20_matches"] < MIN_Q20_MATCHES
            or core["q20_matches"] != overlap["q20_matches"]):
        return None
    removed = len(overlap["markers"]) - len(core["markers"])
    if not 1 <= removed <= LOW_QUALITY_END_COLUMNS:
        return None
    for role, key in (("forward", "aligned_f"), ("reverse", "aligned_r")):
        start, end = core[f"{role}_start"], core[f"{role}_end"]
        if not overlap[f"{role}_start"] <= start <= end <= overlap[f"{role}_end"]:
            return None
        offset = start - overlap[f"{role}_start"]
        if (overlap[key].replace("-", "")[offset:offset + end - start]
                != core[key].replace("-", "")):
            return None
    # 정렬 열은 말단 제외에 따라 이동하므로 실제 read 좌표와 Q로 대조한다.
    # 남은 Gap/불일치의 위치나 Q가 바뀌면 내부 재정렬로 보고 적용하지 않는다.
    def signature(event):
        return tuple(event[key] for key in ("kind", "f_position", "r_position", "support_q"))

    original_events = [signature(e) for e in overlap["events"]]
    core_events = [signature(e) for e in core["events"]]
    if any(event not in original_events for event in core_events):
        return None
    excluded = [e for e in overlap["events"] if signature(e) not in core_events]
    gaps = [e for e in excluded if e["kind"] in ("F Gap", "R Gap")]
    if not gaps or any(e["support_q"] >= 20 for e in excluded):
        return None
    left = max(core[f"{role}_start"] - overlap[f"{role}_start"]
               for role in ("forward", "reverse"))
    right = max(overlap[f"{role}_end"] - core[f"{role}_end"]
                for role in ("forward", "reverse"))
    # 말단 반복 염기에서 Gap이 경계의 같은 염기로 한 칸 이동하는 경우를 포함한다.
    # 그보다 내부의 차이가 사라지는 비교는 허용하지 않는다.
    if not all((left > 0 and e["column"] <= left + 1)
               or (right > 0 and e["column"] >= len(overlap["markers"]) - right)
               for e in excluded):
        return None
    return {"extension_columns": removed,
            "extension_paired_bases": overlap["paired_bases"] - core["paired_bases"],
            "additional_q20_matches": 0,
            "excluded_gap_columns": [e["column"] for e in gaps],
            "excluded_gap_max_q": max(e["support_q"] for e in gaps),
            "core_paired_bases": core["paired_bases"],
            "core_gaps": core["gaps"]}


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


def long_overlap_evidence(overlap):
    """긴 저품질 겹침을 위한 제한적 보정의 근거와 각 검사 결과.

    Q20 이상 충돌을 모두 같은 강도로 취급하는 기존 규칙은 낮은 품질이
    많은 긴 겹침을 과도하게 배제할 수 있다. 정렬 열의 과반수가 양쪽 Q20
    지지를 갖지 못하는 긴 연결만 별도 검토한다. Q30 충돌이 반복되거나
    연결 위치·정렬 점수·최소 일치 지지가 부족하면 허용하지 않는다.

    한쪽 read의 Q만으로 충돌을 세지 않는다. Gap도 기존의 양쪽 인접 품질
    규칙을 그대로 사용한다. 파일명, primer, QT 값, 실측 정답은 입력하지
    않는다. 보정 자료 재현을 위한 경험적 기준이며 독립 검증된 확률이 아니다.
    """
    if overlap is None:
        return {"accepted": False, "q20_supported_fraction": None,
                "q30_conflicts": 0, "checks": {}}
    q20_fraction = ((overlap["q20_matches"] + overlap["q20_conflicts"])
                    / max(1, len(overlap["markers"])))
    q30_conflicts = sum(e["support_q"] >= 30 for e in overlap["events"])
    checks = {
        "겹침 길이": overlap["paired_bases"] > MAX_SHORT_GAP_PAIR_BASES,
        "저품질 비중": q20_fraction < LONG_OVERLAP_MAX_Q20_FRACTION,
        "전체 일치율": overlap["gap_identity"] >= LONG_OVERLAP_MIN_IDENTITY,
        "정렬 점수": overlap["score"] >= LONG_OVERLAP_MIN_SCORE,
        "Q20 일치 지지": overlap["q20_matches"] >= LONG_OVERLAP_MIN_Q20_MATCHES,
        "Q30 충돌": q30_conflicts <= LONG_OVERLAP_MAX_Q30_CONFLICTS,
        "연결 배치": overlap["connection"] in ("F → R", "R → F"),
        "연결 경계 잔여": overlap["junction_unaligned"] <= LONG_OVERLAP_MAX_JUNCTION,
        "연결 경계 Q20 잔여": overlap["junction_q20"] <= LONG_OVERLAP_MAX_Q20_JUNCTION,
        "최장 Gap": overlap["longest_gap"] <= MAX_GAP_RUN,
    }
    return {"accepted": all(checks.values()), "q20_supported_fraction": q20_fraction,
            "q30_conflicts": int(q30_conflicts), "checks": checks}


def supported_short_gap_pair(overlap):
    """중간의 연속 품질 지지가 있는 중간 품질 단염기 Gap 쌍의 제한적 허용.

    두 정렬에서 같은 read 좌표의 반대 방향 Gap 두 개가 유지되고, 그 사이
    양쪽 Q20 연속 일치가 30 bp 이상일 때만 인정한다. Q10 미만의 불안정한
    Gap, Q30 이상 충돌, 염기 불일치, 고품질 미정렬 끝은 허용하지 않는다.
    실제 ACE 6개로 보정한 경험적 범위이며 독립 검증된 조립 규칙이 아니다.
    샘플명·프라이머·조건·정답 파일은 판정에 사용하지 않는다.
    """
    if not overlap:
        return None
    core = overlap.get("core_alignment")
    if (not core or core.get("connection") != overlap.get("connection")
            or core.get("reverse_orientation") != overlap.get("reverse_orientation")):
        return None
    gaps_by_alignment = []
    for item in (overlap, core):
        pair = item.get("opposing_gap_pair")
        if (not pair or item.get("connection") not in ("F → R", "R → F")
                or not MIN_OVERLAP_BASES <= item.get("paired_bases", 0) <= MAX_SHORT_GAP_PAIR_BASES
                or item.get("gap_identity", 0) < MIN_GAP_IDENTITY
                or item.get("q20_matches", 0) < MIN_Q20_MATCHES
                or item.get("gaps") != 2 or item.get("longest_gap") != 1
                or item.get("mismatches") != 0 or item.get("ambiguous") != 0
                or item.get("q20_base_conflicts") != 0
                or item.get("junction_q20") != 0
                or item.get("junction_unaligned", MAX_SHORT_LOW_Q_JUNCTION + 1) > MAX_SHORT_LOW_Q_JUNCTION
                or pair.get("between_q20_run", 0) < MIN_MIXED_GAP_Q20_RUN):
            return None
        gaps = item.get("events", [])
        if (len(gaps) != 2 or {e["kind"] for e in gaps} != {"F Gap", "R Gap"}
                or not all(SHORT_GAP_MIN_SUPPORT_Q <= e["support_q"] < SHORT_GAP_MAX_SUPPORT_Q
                           for e in gaps)):
            return None
        gaps_by_alignment.append(gaps)
    def signatures(events):
        return sorted(tuple(e[key] for key in ("kind", "f_position", "r_position", "support_q"))
                      for e in events)
    if signatures(gaps_by_alignment[0]) != signatures(gaps_by_alignment[1]):
        return None
    values = [e["support_q"] for e in gaps_by_alignment[0]]
    return {"between_q20_run": overlap["opposing_gap_pair"]["between_q20_run"],
            "core_between_q20_run": core["opposing_gap_pair"]["between_q20_run"],
            "min_gap_support_q": min(values), "max_gap_support_q": max(values),
            "junction_unaligned": overlap["junction_unaligned"],
            "core_junction_unaligned": core["junction_unaligned"]}


def supported_terminal_core(overlap):
    """말단 확장만 불안정하고 중심 정렬은 단순·고품질인 경우의 제한적 허용.

    기존 말단 확장 검사가 확인한 동일 좌표 중심부만 사용한다. 중심에
    Q20 충돌이나 복수 Gap, 염기 불일치가 있으면 허용하지 않는다.
    실제 성공 ACE 2개로 보정했으며 입력 서열이나 정렬을 고쳐 쓰지 않는다.
    """
    extension = unsupported_terminal_extension(overlap)
    if not extension:
        return None
    core = overlap["core_alignment"]
    for item in (overlap, core):
        if (item["gap_identity"] < MIN_GAP_IDENTITY
                or item["mismatches"] or item["ambiguous"] or item["q20_conflicts"]
                or item["junction_q20"] or item["junction_unaligned"] > MAX_SHORT_LOW_Q_JUNCTION
                or item["longest_gap"] > 1):
            return None
    if core["gaps"] > 1 or overlap["gaps"] > 2:
        return None
    return {"extension_columns": extension["extension_columns"],
            "core_paired_bases": core["paired_bases"], "core_q20_matches": core["q20_matches"],
            "core_gaps": core["gaps"], "core_q20_conflicts": core["q20_conflicts"],
            "core_junction_unaligned": core["junction_unaligned"]}


def persistent_same_direction_gaps(overlap):
    """짧은 연결 말단에 반복되는 같은 방향의 길이 차이와 약한 바깥 지지.

    두 정렬에 같은 좌표로 유지되는, 가까운 단염기 Gap 두 개를 확인한다.
    한 개는 Q20 이상이고 다른 하나는 Q20 미만이어야 한다. 동일 방향의
    반복 차이를 기존 '고품질 Gap 한 개' 예외가 무조건 허용하지 않도록 한다.
    """
    if not overlap or overlap['paired_bases']>MAX_SHORT_GAP_PAIR_BASES:
        return None
    core=overlap.get('core_alignment')
    if (not core or core['connection']!=overlap['connection']
            or core['reverse_orientation']!=overlap['reverse_orientation']
            or core['connection'] not in ('F → R','R → F')):
        return None
    gaps=[e for e in core['events'] if e['kind'] in ('F Gap','R Gap')]
    alternatives={e['column']:e for e in core.get('equivalent_gap_support',[])}
    lower_q=[alternatives.get(e['column'],{}).get('min_support_q',e['support_q']) for e in gaps]
    if (len(gaps)!=2 or gaps[0]['kind']!=gaps[1]['kind'] or core['longest_gap']!=1
            or not min(lower_q)<20<=max(lower_q)
            or gaps[1]['column']-gaps[0]['column']>LOW_QUALITY_END_COLUMNS):
        return None
    def signature(e):
        return tuple(e[k] for k in ('kind','f_position','r_position','support_q'))
    if not all(signature(e) in [signature(x) for x in overlap['events']] for e in gaps):
        return None
    left=gaps[0]['column']-1
    right=len(core['markers'])-gaps[1]['column']
    side='left' if left<right else 'right'
    if min(left,right)>LOW_QUALITY_END_COLUMNS or core['gap_outer_q20_anchors'][side]>=MIN_GAP_PAIR_OUTER_Q20:
        return None
    return {'gap_direction':gaps[0]['kind'], 'outer_columns':min(left,right),
            'outer_q20_run':core['gap_outer_q20_anchors'][side],
            'gap_spacing_columns':gaps[1]['column']-gaps[0]['column'],
            'gap_support_q':[e['support_q'] for e in gaps], 'equivalent_min_support_q':lower_q}


def supported_equivalent_gap_core(overlap):
    """짧은 말단 불일치와 동점 Gap 배치가 겹친 경우만 제한적으로 허용.

    연결 끝이 모두 맞는 짧은 정렬에서 Q20~29 불일치 하나만 말단 5열
    이내 제외로 없어져야 한다. 내부에는 반대 방향 단염기 Gap 두 개만
    남고, 같은 염기 안의 동점 배치로 강한 Gap 충돌은 한 개 이하가 되어야
    한다. Q30 염기 불일치, 복수 잔여 충돌, 긴 미정렬 끝은 허용하지 않는다.
    실제 ACE의 말단 제외·Gap 배치를 참고한 경험적 기준이다.
    """
    if not overlap or not MIN_OVERLAP_BASES<=overlap['paired_bases']<=MAX_SHORT_GAP_PAIR_BASES:
        return None
    core=overlap.get('core_alignment')
    if (not core or core['connection']!=overlap['connection']
            or core['reverse_orientation']!=overlap['reverse_orientation']
            or overlap['connection'] not in ('F → R','R → F')
            or overlap['junction_unaligned']!=0):
        return None
    if (overlap['gaps']!=2 or core['gaps']!=2
            or overlap['longest_gap']!=1 or core['longest_gap']!=1
            or overlap['mismatches']!=1 or overlap['ambiguous']
            or core['mismatches'] or core['ambiguous'] or not core.get('opposing_gap_pair')
            or core['paired_bases']<MIN_OVERLAP_BASES
            or min(overlap['q20_matches'],core['q20_matches'])<MIN_CLIPPED_CORE_Q20_MATCHES
            or min(overlap['gap_identity'],core['gap_identity'])<MIN_GAP_IDENTITY):
        return None
    removed=len(overlap['markers'])-len(core['markers'])
    if (not 1<=removed<=MAX_CORE_TRIM_COLUMNS
            or core['junction_unaligned']>MAX_CORE_TRIM_COLUMNS
            or core['junction_q20']>MAX_CORE_TRIM_COLUMNS):
        return None
    def signature(e):
        return tuple(e[k] for k in ('kind','f_position','r_position','support_q'))
    original=[e for e in overlap['events'] if e['kind'] in ('F Gap','R Gap')]
    if [signature(e) for e in original]!=[signature(e) for e in core['events']]:
        return None
    mismatch=next(e for e in overlap['events'] if e['kind']=='염기 불일치')
    if not 20<=mismatch['support_q']<CORE_CONFLICT_QUALITY_LIMIT:
        return None
    size=len(core['markers'])
    if not any(overlap['aligned_f'][left:left+size]==core['aligned_f']
               and overlap['aligned_r'][left:left+size]==core['aligned_r']
               and not left<mismatch['column']<=left+size
               for left in range(removed+1)):
        return None
    alternatives=core.get('equivalent_gap_support',[])
    if (len(alternatives)!=2
            or {e['column'] for e in alternatives}!={e['column'] for e in core['events']}
            or sum(e['min_support_q']>=20 for e in alternatives)>1
            or not any(e['min_support_q']<20<=e['max_support_q'] for e in alternatives)):
        return None
    ordered=sorted(alternatives,key=lambda e:e['column'])
    if max(p['column'] for p in ordered[0]['placements'])>=min(p['column'] for p in ordered[1]['placements']):
        return None
    return {'excluded_columns':removed,'excluded_mismatch_q':mismatch['support_q'],
            'core_q20_matches':core['q20_matches'],
            'gap_support_ranges':[[e['min_support_q'],e['max_support_q']] for e in alternatives]}


def supported_clipped_terminal_gap(overlap):
    """짧은 저품질 말단 Gap을 제외해도 단일 Gap 중심이 남을 때의 제한적 허용.

    실제 ACE에서 확인한 말단 제외와 같은 염기 안의 동점 Gap 이동만
    인정한다. 중간 정렬·염기·Q는 바꾸지 않는다. 반대 방향 Gap 두 개가
    모두 남는 조건, Q30 충돌, 5열을 넘는 제외에는 적용하지 않는다.
    제외되는 일치 염기는 모두 양쪽 Q20 지지가 있어야 하며, 중심에도
    양쪽 Q20 일치 30 bp 이상이 남아야 한다. 실제 성공·실패 사례로
    보정한 범위이며 실제 조립기의 내부 규칙을 복원한 것은 아니다.
    """
    if not overlap:
        return None
    core = overlap.get('core_alignment')
    if (not core or core['connection'] != overlap['connection']
            or core['reverse_orientation'] != overlap['reverse_orientation']
            or overlap['connection'] not in ('F → R', 'R → F')
            or overlap['junction_unaligned'] != 0 or overlap['junction_q20'] != 0
            or overlap['gaps'] != 2 or core['gaps'] != 1
            or not overlap.get('opposing_gap_pair')):
        return None
    for item in (overlap, core):
        if (not MIN_OVERLAP_BASES <= item['paired_bases'] <= MAX_SHORT_GAP_PAIR_BASES
                or item['gap_identity'] < MIN_GAP_IDENTITY
                or item['q20_matches'] < MIN_CLIPPED_CORE_Q20_MATCHES
                or item['mismatches'] or item['ambiguous'] or item['q20_base_conflicts']
                or item['q20_gap_conflicts'] != 1 or item['longest_gap'] != 1
                or item['junction_unaligned'] > MAX_CORE_TRIM_COLUMNS
                or item['junction_q20'] > MAX_CORE_TRIM_COLUMNS):
            return None
    removed = len(overlap['markers']) - len(core['markers'])
    if not 1 <= removed <= MAX_CORE_TRIM_COLUMNS:
        return None
    # 말단에서 Gap 이외에 제외되는 일치는 모두 양쪽 Q20 지지가 있어야 한다.
    # 같은 서열 모양이라도 그 바깥 일치가 저품질이면 이 예외를 적용하지 않는다.
    excluded_matches = overlap['paired_bases'] - core['paired_bases']
    if (excluded_matches < 1
            or overlap['q20_matches'] - core['q20_matches'] != excluded_matches
            or core['junction_q20'] != core['junction_unaligned']):
        return None
    original, retained = overlap['events'], core['events']
    if len(original) != 2 or len(retained) != 1:
        return None
    def signature(e):
        return tuple(e[k] for k in ('kind', 'f_position', 'r_position', 'support_q'))
    if (signature(retained[0]) not in [signature(e) for e in original]
            or not 20 <= retained[0]['support_q'] < CORE_CONFLICT_QUALITY_LIMIT):
        return None
    excluded = next(e for e in original if signature(e) != signature(retained[0]))
    if excluded['kind'] not in ('F Gap', 'R Gap') or excluded['support_q'] >= 20:
        return None
    for role in ('forward', 'reverse'):
        if not (overlap[role+'_start'] <= core[role+'_start']
                < core[role+'_end'] <= overlap[role+'_end']):
            return None
    alternative = next((e for e in overlap.get('equivalent_gap_support', [])
                        if e['column'] == excluded['column'] and e['kind'] == excluded['kind']), None)
    if not alternative or alternative['max_support_q'] >= CORE_CONFLICT_QUALITY_LIMIT:
        return None
    # 말단의 동일 염기 안에서만 Gap을 옮겨 중심 정렬 문자열을 대조한다.
    # 제외할 Gap이 중심에 남는 배치나 다른 내부 재배치는 허용하지 않는다.
    key = 'aligned_f' if excluded['kind'] == 'F Gap' else 'aligned_r'
    other = 'aligned_r' if key == 'aligned_f' else 'aligned_f'
    size = len(core['markers'])
    for placement in alternative['placements']:
        at = placement['column'] - 1
        candidate = list(overlap[key])
        candidate.pop(excluded['column'] - 1)
        candidate.insert(at, '-')
        candidate = ''.join(candidate)
        for left in range(removed + 1):
            right = left + size
            if (not left <= at < right
                    and candidate[left:right] == core[key]
                    and overlap[other][left:right] == core[other]):
                return {'excluded_columns': removed,
                        'excluded_gap_support_q': excluded['support_q'],
                        'equivalent_gap_max_q': alternative['max_support_q'],
                        'gap_shift_columns': abs(at - excluded['column'] + 1),
                        'core_paired_bases': core['paired_bases'],
                        'core_alignment_columns': size,
                        'core_q20_matches': core['q20_matches'],
                        'excluded_q20_matches': excluded_matches,
                        'remaining_gap_support_q': retained[0]['support_q'],
                        'core_junction_q20': core['junction_q20']}
    return None


def persistent_short_gap_mismatch(overlap):
    """두 정렬에 반복되는 같은 방향 Gap과 품질 지지 염기 불일치를 찾는다.

    짧은 연결의 단염기 Gap 세 구간 이상과 Q20 이상 염기 불일치가 모두
    같은 read 좌표에서 유지될 때만 적용한다. 반대 방향 Gap 조합, 긴
    겹침, 한두 Gap, 보수적 정렬에서 해소되는 차이는 이 검사로 제한하지
    않는다. Q가 낮은 Gap을 버려 전체 충돌 비율만 낮아지는 경우를 보완한다.
    실제 실패 ACE를 참고한 경험적 검사이며 조립기의 내부 규칙은 아니다.
    """
    if not overlap:
        return None
    core = overlap.get('core_alignment')
    if (not core or overlap.get('connection') not in ('F → R', 'R → F')
            or core.get('connection') != overlap.get('connection')
            or core.get('reverse_orientation') != overlap.get('reverse_orientation')):
        return None
    gap_sets, mismatch_sets = [], []
    def signature(event):
        return tuple(event[k] for k in ('kind', 'f_position', 'r_position', 'support_q'))
    for item in (overlap, core):
        if (not MIN_OVERLAP_BASES <= item.get('paired_bases', 0) <= MAX_SHORT_GAP_PAIR_BASES
                or item.get('longest_gap') != 1):
            return None
        gaps = [e for e in item.get('events', []) if e['kind'] in ('F Gap', 'R Gap')]
        if len(gaps) < MIN_PERSISTENT_SHORT_GAP_REGIONS or len({e['kind'] for e in gaps}) != 1:
            return None
        mismatches = [e for e in item.get('events', [])
                      if e['kind'] == '염기 불일치' and e['support_q'] >= 20]
        if not mismatches:
            return None
        gap_sets.append({signature(e) for e in gaps})
        mismatch_sets.append({signature(e) for e in mismatches})
    common_gaps = gap_sets[0] & gap_sets[1]
    common_mismatches = mismatch_sets[0] & mismatch_sets[1]
    if len(common_gaps) < MIN_PERSISTENT_SHORT_GAP_REGIONS or not common_mismatches:
        return None
    return {'gap_direction': next(iter(common_gaps))[0],
            'persistent_gap_regions': len(common_gaps),
            'persistent_q20_mismatches': len(common_mismatches),
            'mismatch_support_q': sorted(e[3] for e in common_mismatches),
            'main_gap_regions': len(gap_sets[0]), 'core_gap_regions': len(gap_sets[1]),
            'paired_bases': overlap['paired_bases'], 'core_paired_bases': core['paired_bases']}


def internal_low_quality_alignment(overlap):
    """낮은 품질의 내부 유사성에 비해 고품질 연결 끝이 크게 남는 경우 경고."""
    if not overlap or overlap['paired_bases']<=MAX_SHORT_GAP_PAIR_BASES:
        return None
    fraction=(overlap['q20_matches']+overlap['q20_conflicts'])/max(1,len(overlap['markers']))
    if (fraction>=INTERNAL_REVIEW_MAX_Q20_FRACTION
            or overlap['junction_q20']<=INTERNAL_REVIEW_MIN_JUNCTION_Q20):
        return None
    return {'q20_supported_fraction':fraction,'junction_q20':overlap['junction_q20'],
            'message':'낮은 품질의 내부 구간끼리 맞물린 정렬입니다. 파일 생성 여부는 단정하기 어렵고, '
                      'Contig 파일이 생성되어도 정상적인 F/R 연결로 인정하기 전에 정렬 확인이 필요합니다.'}


def predict_contig(lengths, overlap, quality_bases=None):
    """품질이 뒷받침하는 충돌과 낮은 품질의 차이를 분리해 평가한다.

    한 개의 Q20 지지 단염기 gap은 짧은 겹침에서 비율만으로 기각하지 않는다.
    연결 끝까지 정렬된 단염기 gap 쌍과 짧은 말단 제외에 제한적 비율 예외를 적용한다.
    그 밖의 복수 Q20 충돌과 Q20 염기 불일치는 기존 충돌 비율 기준을 적용한다.
    긴 일치 구간이 국소 충돌을 희석하지 않도록 충돌 집중도도 검사한다.
    Q20 유효 염기 수, 양쪽 반복 gap 검사, 말단 gap 쌍과 연속 지지는 검증용 보정이다.
    짧은 겹침에서는 저품질 gap도 포함한 반대 방향 쌍을 별도로 검토한다.
    긴 저품질 겹침은 별도의 품질·배치 검사를 모두 통과할 때만 보정한다.
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
    supported_pair = supported_short_gap_pair(overlap)
    supported_core = supported_terminal_core(overlap)
    equivalent_core = supported_equivalent_gap_core(overlap)
    clipped_gap_core = supported_clipped_terminal_gap(overlap)
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
                and not terminal_realign_applied and not supported_pair and not equivalent_core):
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
        if not supported_pair and not equivalent_core and not clipped_gap_core:
            reasons.extend(short_gap_pair_reasons(overlap))
        same_direction=persistent_same_direction_gaps(overlap)
        if same_direction:
            reasons.append('같은 방향의 단염기 Gap 두 개가 짧은 말단에 반복되고, '
                           f"그 바깥 연속 Q20 일치는 {same_direction['outer_q20_run']} bp로 "
                           f'기준 {MIN_GAP_PAIR_OUTER_Q20} bp 미만입니다. 보수적 정렬에도 같은 차이가 남습니다 (검증용 보정).')
        repeated_short = persistent_short_gap_mismatch(overlap)
        if repeated_short:
            reasons.append('짧은 겹침에서 같은 방향의 단염기 Gap '
                           f"{repeated_short['persistent_gap_regions']}구간과 Q20 이상 염기 불일치 "
                           f"{repeated_short['persistent_q20_mismatches']}개가 보수적 정렬에도 "
                           '같은 read 위치에 남습니다. 전체 일치율만으로 연결을 허용하지 않습니다 '
                           '(검증용 보정).')
        extension = unsupported_terminal_extension(overlap)
        if extension and not supported_core:
            reasons.append(
                f"보수적 비교보다 말단 {extension['extension_columns']}개 정렬 열을 확장했지만 "
                "양쪽 Q20 이상 일치는 늘지 않았습니다. "
                f"추가 Gap {len(extension['excluded_gap_columns'])}개 열의 지지 Q는 "
                f"최대 {extension['excluded_gap_max_q']}이고, 연결 경계에 미정렬 염기 "
                f"{overlap['junction_unaligned']} bp가 남아 있습니다 (검증용 보정)."
            )
        if unresolved_low_quality_end(overlap):
            terminal_gaps = terminal_low_quality_gap_regions(overlap)
            if terminal_gaps:
                positions = ", ".join(str(g["start_column"]) if g["start_column"] == g["end_column"]
                                      else f"{g['start_column']}–{g['end_column']}" for g in terminal_gaps)
                reasons.append(f"겹침 끝 {LOW_QUALITY_END_COLUMNS}개 정렬 열에 서로 떨어진 "
                               f"저품질 Gap {len(terminal_gaps)}구간(정렬 위치 {positions})이 있고, "
                               f"연결 경계에 미정렬 염기 {overlap['junction_unaligned']} bp가 남아 있습니다 "
                               "(검증용 보정).")
            else:
                reasons.append(f"겹침 끝 {LOW_QUALITY_END_COLUMNS}개 정렬 열에 저품질 Gap과 "
                               "염기 불일치가 함께 있고, 연결 경계의 미정렬 염기도 남아 있습니다 "
                               "(검증용 보정).")
    evidence = long_overlap_evidence(overlap)
    if reasons and evidence["accepted"]:
        return {"label": "Contig 생성", "state": "joined", "generated": True,
                "decision_rule": "long_low_quality", "standard_reasons": reasons,
                "reasons": [
                    "긴 저품질 겹침의 별도 품질·배치 기준을 모두 통과했습니다 (검증용 보정).",
                    f"겹침 {overlap['paired_bases']} bp · 전체 일치율 {overlap['gap_identity']:.1f}% · "
                    f"정렬 점수 {overlap['score']:g} · 양쪽 Q20 일치 {overlap['q20_matches']} bp.",
                    f"양쪽 Q20 지지 비중 {evidence['q20_supported_fraction']:.1%} · "
                    f"양쪽 Q30 충돌 {evidence['q30_conflicts']}개 "
                    f"(허용 {LONG_OVERLAP_MAX_Q30_CONFLICTS}개 이하).",
                    f"연결 끝 미정렬 {overlap['junction_unaligned']} bp "
                    f"(Q20 이상 {overlap['junction_q20']} bp)로 허용 범위 안에 있습니다.",
                ]}
    if reasons:
        result = {"label": "Contig2", "state": "separate", "generated": False, "reasons": reasons}
        if overlap is not None:
            repeated_short = persistent_short_gap_mismatch(overlap)
            if repeated_short:
                result['rejection_evidence'] = {'persistent_short_gap_mismatch': repeated_short}
        warning=internal_low_quality_alignment(overlap)
        if warning:
            result['alignment_review']=warning
        return result
    passed = ["겹침 길이·일치율·품질·연결 경계 기준을 통과했습니다."]
    if terminal_gap_exception_applied:
        passed.append("연결 끝까지 정렬된 단염기 Gap 두 개의 방향과 말단 위치 조건으로 "
                      "충돌 비율 예외를 적용했습니다 (검증용 보정).")
        if overlap["q20_gap_regions"] == {"F": 1, "R": 1}:
            anchors = overlap["gap_outer_q20_anchors"]
            passed.append(f"첫 Gap 앞 / 마지막 Gap 뒤의 연속 Q20 일치가 "
                          f"{anchors['left']} / {anchors['right']} bp이며, "
                          f"한쪽 이상 {MIN_GAP_PAIR_OUTER_Q20} bp 기준을 충족합니다.")
    if terminal_realign_applied:
        passed.append(f"말단 {MAX_CORE_TRIM_COLUMNS}개 정렬 열 이내를 제외한 비교에서 "
                      f"염기 불일치가 해소되고 남은 Gap의 지지 Q가 {CORE_CONFLICT_QUALITY_LIMIT} 미만입니다 "
                      "(충돌 비율 예외: 검증용 보정).")
    result = {"label": "Contig 생성", "state": "joined", "generated": True, "reasons": passed}
    corrections = {}
    if clipped_gap_core:
        corrections['supported_clipped_terminal_gap'] = clipped_gap_core
        passed.append(f"저품질 말단 Gap을 포함한 {clipped_gap_core['excluded_columns']}개 정렬 열을 "
                      f"제외해도 중심 겹침 {clipped_gap_core['core_paired_bases']} bp와 양쪽 Q20 일치 "
                      f"{clipped_gap_core['core_q20_matches']} bp가 남습니다. 동일 염기 안의 Gap 이동을 "
                      "확인했고 중심에는 중간 품질의 단염기 Gap 한 개만 남아, "
                      "짧은 Gap 쌍 제한에 예외를 적용했습니다 (검증용 보정).")
    if equivalent_core:
        corrections['supported_equivalent_gap_core']=equivalent_core
        passed.append(f"끝부분 {equivalent_core['excluded_columns']}개 정렬 열의 불일치를 제외하면 "
                      f"양쪽 Q20 일치 {equivalent_core['core_q20_matches']} bp가 남습니다. "
                      '같은 염기 안에서 빈칸을 옮겨도 동일하게 맞는 배치를 확인했으며, '
                      '그 배치에서는 Q20 Gap 충돌이 한 개 이하라 제한적으로 연결을 허용했습니다 (검증용 보정).')
    if supported_pair:
        corrections["supported_short_gap_pair"] = supported_pair
        passed.append(f"반대 방향 단염기 Gap 두 개의 지지 Q가 {supported_pair['min_gap_support_q']}–"
                      f"{supported_pair['max_gap_support_q']}이고, 그 사이 양쪽 Q20 연속 일치가 "
                      f"{supported_pair['between_q20_run']} bp입니다. 보수적 정렬에서도 같은 Gap과 "
                      "연속 일치가 유지되어 연결을 허용했습니다 (검증용 보정).")
    if supported_core:
        corrections["supported_terminal_core"] = supported_core
        passed.append(f"불안정한 말단 {supported_core['extension_columns']}개 정렬 열을 제외한 중심은 "
                      f"겹침 {supported_core['core_paired_bases']} bp, 양쪽 Q20 일치 "
                      f"{supported_core['core_q20_matches']} bp, Q20 충돌 0개, "
                      f"저품질 Gap {supported_core['core_gaps']}개로 연결 근거가 유지됩니다 "
                      "(검증용 보정).")
    if corrections:
        passed[0] = "겹침·품질·연결 경계를 확인하고, 중심 일치 근거에 따른 제한적 보완 기준을 통과했습니다."
        result["applied_corrections"] = corrections
    return result


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
        evidence = long_overlap_evidence(row["overlap"])
        extension = unsupported_terminal_extension(row["overlap"])
        terminal = persistent_weak_terminal_gap(row["overlap"])
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
            "Q30 충돌": evidence["q30_conflicts"],
            "양쪽 Q20 지지 비중 (%)": (round(100 * evidence["q20_supported_fraction"], 2)
                                    if evidence["q20_supported_fraction"] is not None else None),
            "긴 저품질 겹침 보정 적용": prediction.get("decision_rule") == "long_low_quality",
            "중심 Q20 지지 Gap 쌍 보정 적용": "supported_short_gap_pair" in prediction.get("applied_corrections", {}),
            "단순 중심 정렬 말단 보정 적용": "supported_terminal_core" in prediction.get("applied_corrections", {}),
            "동점 Gap 배치와 말단 보정 적용": "supported_equivalent_gap_core" in prediction.get("applied_corrections", {}),
            "저품질 말단 Gap 제외 보정 적용": "supported_clipped_terminal_gap" in prediction.get("applied_corrections", {}),
            "같은 방향 반복 말단 Gap 검사": persistent_same_direction_gaps(row['overlap']) is not None,
            "짧은 겹침 반복 Gap·염기 차이 검사": persistent_short_gap_mismatch(row['overlap']) is not None,
            "반복 Gap·염기 차이 상세": json.dumps(prediction.get('rejection_evidence', {}), ensure_ascii=False),
            "정렬 확인 경고": (prediction.get('alignment_review') or {}).get('message',''),
            "F Q20 Gap 구간": overlap.get("q20_gap_regions", {}).get("F", 0),
            "R Q20 Gap 구간": overlap.get("q20_gap_regions", {}).get("R", 0),
            "첫 Gap 앞 연속 Q20 일치 (bp)": overlap.get("gap_outer_q20_anchors", {}).get("left", 0),
            "마지막 Gap 뒤 연속 Q20 일치 (bp)": overlap.get("gap_outer_q20_anchors", {}).get("right", 0),
            "반대 방향 Gap 쌍 사이 최장 연속 Q20 일치 (bp)": (overlap.get("opposing_gap_pair") or {}).get("between_q20_run"),
            "반대 방향 Gap 쌍 중 Q20 이상 개수": (overlap.get("opposing_gap_pair") or {}).get("q20_gap_count"),
            "연결 경계 잔여 (bp)": overlap.get("junction_unaligned", 0),
            "연결 경계 Q20 잔여 (bp)": overlap.get("junction_q20", 0),
            "말단 보수적 비교 겹침 (bp)": (overlap.get("core_alignment") or {}).get("paired_bases", 0),
            "말단 보수적 비교 Gap (bp)": (overlap.get("core_alignment") or {}).get("gaps", 0),
            "품질 지지 없는 말단 확장 검사": extension is not None,
            "품질 지지 없는 확장 정렬 열": extension["extension_columns"] if extension else 0,
            "해당 확장의 추가 Q20 일치 (bp)": extension["additional_q20_matches"] if extension else None,
            "보수적 정렬의 약한 말단 Gap 검사": terminal is not None,
            "해당 말단 Gap 지지 Q": terminal["low_gap_support_q"] if terminal else None,
            "해당 Gap 바깥 연속 Q20 일치 (bp)": terminal["outer_q20_run"] if terminal else None,
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


def direction_quality_evidence(reads, row):
    """실제 정렬의 각 염기 Q를 방향별로 센다. Gap 방향은 오류 방향으로 쓰지 않는다.

    R의 Q 배열도 서열과 함께 뒤집는다. 겹침 안의 염기 불일치에서만
    저품질 쪽 / Q30 이상 반대쪽이라는 비대칭 근거를 센다.
    동일 read의 여러 조건은 독립된 표본이나 성공 확률 근거가 아니다.
    """
    overlap = row['overlap']
    if overlap is None:
        return None
    condition = Condition(**row['condition'])
    f, r = [trim_read(read, condition)[0] for read in reads]
    rq = r.qualities[::-1] if overlap['reverse_orientation'] == 'Reverse complement' else r.qualities
    cursors = [overlap['forward_start'], overlap['reverse_start']]
    qualities = [f.qualities, rq]
    evidence = {role: {'bases': 0, 'q20': 0, 'weak_differences': 0}
                for role in ('F', 'R')}
    for fb, rb in zip(overlap['aligned_f'], overlap['aligned_r']):
        bases = (fb, rb)
        qs = [qualities[i][cursors[i]] if base != '-' else None
              for i, base in enumerate(bases)]
        for i, role in enumerate(('F', 'R')):
            if bases[i] != '-':
                evidence[role]['bases'] += 1
                evidence[role]['q20'] += int(bases[i] in 'ACGT' and qs[i] >= 20)
                cursors[i] += 1
        # 빈칸의 위치만으로 어느 방향의 측정이 잘못됐는지 판단하지 않는다.
        if '-' in bases or (fb == rb and fb in 'ACGT'):
            continue
        for i, role in enumerate(('F', 'R')):
            j = 1 - i
            if ((qs[i] < 20 or bases[i] not in 'ACGT')
                    and qs[j] >= 30 and bases[j] in 'ACGT'):
                evidence[role]['weak_differences'] += 1
    for values in evidence.values():
        values['q20_fraction'] = values['q20'] / values['bases'] if values['bases'] else 0.0
    return {'condition': row['condition']['label'], 'directions': evidence,
            'paired_bases': overlap['paired_bases'], 'connection': overlap['connection'],
            'high_quality_conflicts': overlap['q20_base_conflicts'],
            'q20_gap_regions': overlap['q20_gap_regions']}


def rerun_profiles(reads, rows):
    profiles = []
    for i, role in enumerate(('F', 'R')):
        read = reads[i]
        valid = [base in 'ACGT' and q >= 20 for base, q in zip(read.sequence, read.qualities)]
        longest = current = 0
        for accepted in valid:
            current = current + 1 if accepted else 0
            longest = max(longest, current)
        fractions = [row['quality_bases'][i] / row['lengths'][i] for row in rows
                     if row['lengths'][i] >= MIN_READ_BASES
                     and row['quality_bases'][i] >= MIN_Q20_READ_BASES]
        profiles.append({'direction': role, 'file_name': read.file_name,
                         'source_sha256': read.source_sha256,
                         'original_bases': len(read.sequence), 'original_q20': sum(valid),
                         'longest_continuous_q20': longest,
                         'max_retained_q20': max(row['quality_bases'][i] for row in rows),
                         'best_retained_q20_fraction': max(fractions, default=0.0)})
    return profiles


def recommend_rerun(reads, rows):
    """재반응 전의 관측만으로 검토안을 만든다. 입력과 기존 Contig 판정은 변경하지 않는다.

    실제 재반응 효과는 미검증이다. Q를 가상으로 올리거나 서열을 수정하지 않는다.
    실제 결과 입력 없이 AB1만으로 검토하며, 재반응 불필요를 확정하지 않는다.
    """
    if len(reads) != 2 or len(rows) != len(CONDITIONS):
        raise ValueError(f'F/R 두 파일과 {len(CONDITIONS)}개 조건의 분석 결과가 필요합니다.')
    profiles = rerun_profiles(reads, rows)
    local = [direction_quality_evidence(reads, row) for row in rows]
    local = [item for item in local if item is not None]
    joined = [row['condition']['label'] for row in rows if row['prediction']['generated']]
    result = {
        'rule_version': RERUN_RULE_VERSION,
        'profiles': profiles, 'local_evidence': local, 'predicted_joined_conditions': joined,
        'validation': '재반응 효과 미검증 · 서열과 Quality에 근거한 검토안',
        'thresholds': {'reference_q20_fraction': RERUN_REFERENCE_FRACTION,
                       'weak_q20_fraction': RERUN_WEAK_FRACTION,
                       'min_asymmetric_differences': RERUN_MIN_DIFFERENCES,
                       'min_supporting_conditions': RERUN_MIN_SUPPORTING_CONDITIONS},
    }

    def finish(code, label, reasons, next_step, evidence_type):
        result.update(code=code, label=label, reasons=reasons, next_step=next_step,
                      evidence_type=evidence_type)
        return result

    if joined:
        return finish('recheck_existing', '기존 데이터 재검토 우선',
                      [', '.join(joined) + ' 조건에서 Contig 생성이 예상됩니다.',
                       '생성 예측만으로 재반응 불필요를 확정하지 않습니다.'],
                      '해당 조건으로 실제 조립한 뒤 최종 품질 기준을 확인해 주세요.', '생성 예상 조건 있음')

    insufficient = [p['max_retained_q20'] < MIN_Q20_READ_BASES for p in profiles]
    if all(insufficient):
        return finish('both', 'F/R 양방향 재반응 검토',
                      [f"검토한 조건 중 가장 많이 남긴 경우에도 Q20 이상 염기가 F {profiles[0]['max_retained_q20']} bp, "
                       f"R {profiles[1]['max_retained_q20']} bp로, 두 방향 모두 최소 {MIN_Q20_READ_BASES} bp에 못 미칩니다."],
                      '양쪽 파형과 시료 상태를 먼저 확인해 주세요. 시료·정제 문제가 있으면 그 원인부터 조치해야 합니다.',
                      '양쪽 유효 서열 부족')
    # 품질이 높은 양쪽 자료가 반복해서 다르면 낮은 평균 Q만으로 방향을 고르지 않는다.
    conflicting = [x['condition'] for x in local
                   if x['paired_bases'] >= MIN_OVERLAP_BASES
                   and (x['high_quality_conflicts'] >= RERUN_MIN_DIFFERENCES
                        or min(x['q20_gap_regions'].values()) >= MIN_BIDIRECTIONAL_GAP_REGIONS)]
    if len(conflicting) >= RERUN_MIN_SUPPORTING_CONDITIONS:
        return finish('trace_review', '파형·시료 확인 우선',
                      [', '.join(conflicting) + ' 조건에서 품질이 뒷받침되는 서열 차이가 반복됩니다.',
                       '어느 한쪽이 잘못 읽혔다고 정렬만으로 정할 수 없습니다.'],
                      '양쪽 원본 파형, 샘플 짝, 프라이머와 시료 상태를 확인해 주세요. '
                      '정렬 차이만으로 Mix 또는 InDel을 확정하지 않습니다.', '양쪽 자료의 충돌')
    if sum(insufficient) == 1:
        weak = insufficient.index(True)
        good = 1 - weak
        if profiles[good]['best_retained_q20_fraction'] >= RERUN_REFERENCE_FRACTION:
            role, other = profiles[weak]['direction'], profiles[good]['direction']
            return finish(role.lower() + '_only', role + ' 우선 재반응 검토 · 단측 후보',
                          [f"{role}은 검토한 모든 조건에서 Q20 이상 염기가 {MIN_Q20_READ_BASES} bp 미만입니다.",
                           f"{other}은 이 최소 기준을 넘는 조건이 있고, 그중 품질이 가장 좋은 구간의 Q20 비중은 "
                           f"{profiles[good]['best_retained_q20_fraction']:.0%}입니다."],
                          f'기존 {other}이 필요한 구간을 충분히 읽었는지 확인한 후 {role} 단측을 검토해 주세요. '
                          '재반응 후 길이 회복과 연결 성공은 아직 예측할 수 없습니다.', '한쪽 유효 서열 부족')

    votes = {'F': [], 'R': []}
    both_weak = []
    for x in local:
        if x['paired_bases'] < MIN_OVERLAP_BASES or x['connection'] not in ('F → R', 'R → F'):
            continue
        sides = x['directions']
        for role, other in (('F', 'R'), ('R', 'F')):
            if (sides[role]['q20_fraction'] < RERUN_WEAK_FRACTION
                    and sides[other]['q20_fraction'] >= RERUN_REFERENCE_FRACTION
                    and sides[role]['weak_differences'] >= RERUN_MIN_DIFFERENCES):
                votes[role].append(x['condition'])
        if all(sides[role]['q20_fraction'] < RERUN_WEAK_FRACTION for role in ('F', 'R')):
            both_weak.append(x['condition'])
    if votes['F'] and votes['R']:
        return finish('hold', '판단 보류 · 조건별 방향 근거가 다름',
                      ['조건에 따라 품질이 약한 방향이 달라 단측을 고르기 어렵습니다.'],
                      '겹침 구간의 양쪽 파형을 함께 확인해 주세요.', '방향 근거 불일치')
    for role, other in (('F', 'R'), ('R', 'F')):
        if len(votes[role]) >= RERUN_MIN_SUPPORTING_CONDITIONS and not both_weak:
            return finish(role.lower() + '_only', role + ' 우선 재반응 검토 · 단측 후보',
                          [', '.join(votes[role]) + f' 조건에서 겹침 구간의 {role} 품질이 상대적으로 낮습니다.',
                           f'각 조건에서 {role}의 Q20 비중은 {RERUN_WEAK_FRACTION:.0%} 미만, '
                           f'{other}은 {RERUN_REFERENCE_FRACTION:.0%} 이상이며, '
                           f'{role}이 낮은 품질이고 {other}이 Q30 이상인 서로 다른 염기 위치가 '
                           f'{RERUN_MIN_DIFFERENCES}개 이상입니다.'],
                          f'해당 위치의 파형과 기존 {other}의 필요 구간을 확인한 뒤 {role} 단측을 검토해 주세요. '
                          '현재 품질 차이는 관찰되지만 재반응으로 개선될지는 아직 검증되지 않았습니다.',
                          '겹침 구간의 방향별 품질 차이')
    if len(both_weak) >= RERUN_MIN_SUPPORTING_CONDITIONS and not any(votes.values()):
        return finish('both', 'F/R 양방향 재반응 검토',
                      [', '.join(both_weak) + ' 조건의 겹침 구간에서 두 방향 모두 Q20 비중이 '
                       f'{RERUN_WEAK_FRACTION:.0%} 미만입니다.'],
                      '양쪽 파형과 시료 상태를 확인한 뒤 양방향을 검토해 주세요. '
                      '연결에 필요한 길이가 확보될지, 다른 조치가 필요한지도 함께 확인해야 합니다.',
                      '겹침 구간의 양쪽 품질 부족')
    if not local or max(x['paired_bases'] for x in local) < MIN_OVERLAP_BASES:
        return finish('hold', '판단 보류 · 연결 구간 근거 부족',
                      ['방향별 품질을 비교할 만큼 충분한 겹침을 찾지 못했습니다.'],
                      '프라이머·읽힌 구간·샘플 짝을 확인해 주세요. '
                      '겹침 부족만으로 어느 방향을 반복해야 할지 정할 수 없습니다.', '겹침 부족')
    return finish('hold', '판단 보류 · 단측 선택 근거 부족',
                  ['한 방향만 바꾸면 해결될 것이라고 제안할 만큼 품질 차이가 명확하지 않습니다.'],
                  '겹침 끝과 서열 차이가 있는 위치의 양쪽 파형을 확인해 주세요. '
                  '현재 결과만으로 재반응 생략이나 양방향 재반응을 확정하지 않습니다.', '방향 특정 어려움')


def plain_reason(reason, row):
    """기존 판정 규칙을 바꾸지 않고 그 의미를 쉬운 문장으로 설명한다."""
    o = row['overlap'] or {}
    if '남은 서열이' in reason:
        return '품질이 낮은 부분을 제외한 뒤 남은 서열이 너무 짧아 연결을 평가하기 어렵습니다. ' + reason
    if 'Q20 이상 유효 염기가' in reason:
        return reason.replace('Q20 이상 유효 염기', '품질 기준(Q20)을 넘는 A/C/G/T 염기')
    if reason.startswith('짧은 겹침에서 같은 방향의 단염기 Gap'):
        e = row['prediction']['rejection_evidence']['persistent_short_gap_mismatch']
        return (f"짧은 연결 구간에서 한쪽의 글자 수 차이가 {e['persistent_gap_regions']}곳 반복되고, "
                f"양쪽 품질이 뒷받침하는 염기 차이도 {e['persistent_q20_mismatches']}곳 남습니다. "
                '더 엄격하게 정렬해도 같은 위치의 차이가 유지되어 Contig2로 예상합니다. '
                '이 근거만으로 재반응할 방향을 확정하지는 않습니다.')
    if reason.startswith('저품질 말단 Gap을 포함한'):
        e = row['prediction']['applied_corrections']['supported_clipped_terminal_gap']
        return (f"끝의 불확실한 부분 {e['excluded_columns']}칸을 제외해도 양쪽이 품질 좋게 일치하는 "
                f"위치가 {e['core_q20_matches']}개 남습니다. 중심에는 작은 길이 차이 한 개만 남고 "
                "강한 품질 충돌이 없어, 실제 사례에서 보완한 기준으로 연결을 허용했습니다.")
    if reason.startswith('반대 방향 단염기 Gap 두 개의'):
        e = row['prediction']['applied_corrections']['supported_short_gap_pair']
        return (f"작은 길이 차이 두 곳 사이에서 양쪽이 품질 좋게 연속 일치한 구간이 {e['between_q20_run']} bp입니다. "
                "다른 정렬 방식에서도 이 근거가 유지되고 강한 품질 충돌이 없어, 보완 기준으로 연결을 허용했습니다.")
    if reason.startswith('같은 방향의 단염기 Gap 두 개가'):
        e=persistent_same_direction_gaps(o)
        return (f"한쪽 서열에서 글자 수 차이가 두 번 반복됩니다. 그 바깥에서 양쪽이 품질 좋게 연속 일치한 부분은 "
                f"{e['outer_q20_run']} bp뿐입니다. 전체 일치율이 높아도 이 끝부분의 연결 근거가 약해 Contig2로 예상합니다.")
    if reason.startswith('불안정한 말단 '):
        e = row['prediction']['applied_corrections']['supported_terminal_core']
        return (f"불확실한 끝부분을 제외해도 중심 {e['core_paired_bases']} bp가 겹치고, "
                f"양쪽 모두 품질 좋은 일치가 {e['core_q20_matches']}개 남습니다. "
                "중심에 품질 좋은 충돌이 없고 길이 차이도 적어, 보완 기준으로 연결을 허용했습니다.")
    if reason.startswith('연결에 필요한 겹침'):
        return f"F와 R이 같은 구간을 읽은 부분이 {o.get('paired_bases', 0)} bp로 짧습니다. 현재 기준은 {MIN_OVERLAP_BASES} bp 이상입니다."
    if reason.startswith('Gap 포함 일치율'):
        return f"두 서열을 맞춰 보았을 때 같은 염기의 비율은 {o['gap_identity']:.1f}%로, 현재 기준 {MIN_GAP_IDENTITY:g}%에 못 미칩니다. 빈칸도 차이로 셉니다."
    if reason.startswith('양쪽 Q20 이상 일치가'):
        return f"두 방향 모두 품질이 좋고 같은 염기를 읽은 위치가 {o['q20_matches']}개로, 현재 기준 {MIN_Q20_MATCHES}개에 못 미칩니다."
    if '충돌 비율이' in reason:
        return '두 방향 모두 품질이 좋은데 서로 다른 결과를 보이는 비율이 허용 범위를 넘었습니다. 어느 쪽이 맞는지는 원본 파형 확인이 필요합니다.'
    if '집중되어 있습니다' in reason:
        c = o['conflict_cluster']
        return f"짧은 구간에 품질이 뒷받침되는 차이 {c['count']}개가 몰려 있습니다. 전체 일치율이 높아도 이 구간 때문에 연결이 어려울 수 있습니다."
    if reason.startswith('Q20으로 지지되는 Gap'):
        return '두 서열의 글자 수가 맞지 않는 구간이 양쪽에 반복됩니다. 단순한 끝부분 품질 저하만으로 설명하기 어려워 연결을 보류했습니다.'
    if reason.startswith('연결 경계에 정렬되지 않은'):
        return f"서로 연결되어야 할 끝부분에 품질이 좋은 염기 {o['junction_q20']} bp가 대응되지 않고 남았습니다. 중간 일부만 비슷해도 전체 연결이 된 것은 아닙니다."
    if reason.startswith('연속 Gap이'):
        return f"두 서열의 길이를 맞추기 위해 넣은 빈칸이 연속 {o['longest_gap']}개로 깁니다. 이 차이를 확인해야 합니다."
    if reason.startswith('보수적 비교에도 Q20 이상 Gap'):
        t = persistent_weak_terminal_gap(o)
        return f"짧은 연결 부위의 길이 차이가 다른 정렬 방식에서도 남습니다. 품질이 낮은 차이 바깥쪽에서 양쪽이 연속으로 잘 읽힌 부분은 {t['outer_q20_run']} bp뿐이라, 연결 끝을 충분히 뒷받침하지 못합니다."
    if reason.startswith('보수적 비교보다 말단'):
        return '겹치는 끝을 더 길게 잡아도 양쪽이 확실히 같은 염기를 읽은 구간은 늘지 않았습니다. 불확실한 끝을 연결 근거로 쓰기 어려워 보류했습니다.'
    if reason.startswith('짧은 겹침에'):
        return '짧은 연결 구간에 두 군데 길이 차이가 있고, 그 사이 또는 바깥 끝부분을 뒷받침하는 품질이 부족합니다.'
    if reason.startswith('겹침 끝'):
        return '겹치는 구간의 끝부분에 품질이 낮은 차이가 모여 있고, 연결되지 않은 끝도 남아 있습니다. 연결 경계가 불확실합니다.'
    if reason.startswith('겹침 길이·일치율'):
        return '같은 구간을 읽은 길이, 염기 일치, 품질, 두 끝의 연결 상태가 현재 예측 기준을 통과했습니다.'
    if reason.startswith('긴 저품질 겹침의'):
        return '긴 겹침 안에 낮은 품질이 많지만, 연결 위치와 높은 품질의 차이를 따로 확인하는 보완 기준을 통과했습니다. 실제 조립 확인이 필요합니다.'
    if reason.startswith('연결 끝까지 정렬된') or reason.startswith('첫 Gap 앞 /'):
        return '작은 길이 차이는 있지만 연결 끝까지 맞춰지고 주변의 품질 근거가 있어, 기존 사례에서 보완한 기준으로 생성 가능성을 인정했습니다.'
    if reason.startswith('말단 ') and '비교에서' in reason:
        return '끝의 아주 짧은 불확실한 부분을 제외해 비교했을 때 연결 근거가 확보되었습니다. 기존 사례에서 보완한 기준이 적용되었습니다.'
    if row['prediction'].get('decision_rule') == 'long_low_quality':
        return (f"같은 구간 {o['paired_bases']} bp를 비교했고, 양쪽 모두 품질이 좋으면서 같은 염기를 읽은 곳은 "
                f"{o['q20_matches']}개입니다. 자세한 보완 기준은 기술 지표에서 볼 수 있습니다.")
    return reason


def plain_prediction_reasons(row):
    return list(dict.fromkeys(plain_reason(reason, row) for reason in row['prediction']['reasons']))


def review_export(rows, review):
    records = summary_rows(rows)
    display = rerun_display(review)
    for record, row in zip(records, rows):
        record.update({'쉬운 판정 설명': ' / '.join(plain_prediction_reasons(row)),
                       '재반응 검토': display['headline'], '재반응 판정 코드': review['code'],
                       'F 방향 안내': display['directions'][0]['status'],
                       'R 방향 안내': display['directions'][1]['status'],
                       '검토할 조합': display['combination'],
                       '재반응 근거': ' / '.join(review['reasons']),
                       '다음 확인 사항': review['next_step'], '검증 상태': review['validation'],
                       '재반응 기준 버전': review['rule_version']})
        for p in review['profiles']:
            record[p['direction'] + ' 파일 SHA256'] = p['source_sha256']
    return records


def rerun_display(review):
    """판정 코드를 F/R별 조치와 다음 조합으로 풀어 쓴다. 판단 기준은 바꾸지 않는다."""
    code = review['code']
    if code in ('f_only', 'r_only'):
        target, other = ('F', 'R') if code == 'f_only' else ('R', 'F')
        weak_reason = ('검토한 모든 조건에서 연결에 사용할 염기가 부족합니다.'
                       if review['evidence_type'] == '한쪽 유효 서열 부족'
                       else '겹침 구간에서 이 방향의 품질이 낮은 차이가 반복됩니다.')
        return {
            'headline': target + ' 우선 재반응 검토',
            'summary': f'{target}의 품질을 보완하는 방향부터 검토해 보세요.',
            'combination': f'새 {target} + 기존 {other}',
            'next_action': f'{target}만 재반응한다면, 새 {target}을 기존 {other}과 조립해 확인합니다.',
            'check': f'기존 {other}이 필요한 서열 구간을 충분히 읽었는지 먼저 확인해 주세요.',
            'directions': [
                {'role': role, 'status': '우선 재반응 후보' if role == target else '기존 파일 활용 검토',
                 'emphasis': role == target,
                 'reason': weak_reason if role == target else
                           '추천 근거가 되는 구간의 품질이 상대적으로 높습니다. 필요한 구간의 확보 여부는 확인이 필요합니다.'}
                for role in ('F', 'R')],
        }
    if code == 'both':
        reason = ('검토한 모든 조건에서 연결에 사용할 염기가 부족합니다.'
                  if review['evidence_type'] == '양쪽 유효 서열 부족'
                  else '연결 구간에서 품질이 낮은 부분이 많습니다.')
        return {
            'headline': 'F + R 양방향 재반응 검토',
            'summary': '현재는 한쪽만 보완하는 방향을 제안하기 어렵습니다.',
            'combination': '새 F + 새 R',
            'next_action': '양쪽 파형과 시료 상태를 확인한 뒤, 두 방향의 재반응을 검토합니다.',
            'check': '시료·정제 문제가 있다면 같은 반응을 반복하기 전에 그 원인부터 조치해 주세요.',
            'directions': [{'role': role, 'status': '재반응 검토 대상', 'emphasis': True,
                            'reason': reason} for role in ('F', 'R')],
        }
    if code == 'recheck_existing':
        conditions = ', '.join(review['predicted_joined_conditions'])
        return {
            'headline': '재반응 전 기존 데이터로 조립 확인',
            'summary': f'{conditions} 조건에서 연결이 예상됩니다. 먼저 현재 파일로 실제 조립 결과를 확인해 보세요.',
            'combination': '기존 F + 기존 R',
            'next_action': f'{conditions} 조건으로 조립하고, 생성된 결과의 최종 품질을 확인합니다.',
            'check': '실제로 모든 조건에서 Contig2가 나왔다면 예측과 다른 사례입니다. 이 안내만으로 재반응을 생략하지 마세요.',
            'directions': [{'role': role, 'status': '현재 파일로 조립 확인', 'emphasis': False,
                            'reason': '기존 F/R 조합에서 연결 가능성이 예측되었습니다.'} for role in ('F', 'R')],
        }
    if code == 'trace_review':
        return {
            'headline': '재반응 전 양쪽 파형 확인',
            'summary': '품질이 높은 구간에서도 F와 R의 결과가 달라, 어느 쪽을 다시 해야 할지 고르기 어렵습니다.',
            'combination': '재반응 방향 선택 보류',
            'next_action': '정렬 상세의 ‘원본 파형에서 확인할 위치’를 열어 F와 R을 함께 확인합니다.',
            'check': '샘플 짝·프라이머·시료 상태도 확인해 주세요. 정렬 차이만으로 혼합이나 삽입·결실을 확정하지 않습니다.',
            'directions': [{'role': role, 'status': '차이 난 위치 확인', 'emphasis': False,
                            'reason': '반대 방향과 다르게 읽힌 위치의 원본 파형을 확인해야 합니다.'} for role in ('F', 'R')],
        }
    if review['evidence_type'] == '겹침 부족':
        summary = '연결 구간이 부족해 어느 방향을 보완해야 할지 비교하기 어렵습니다.'
        check = '프라이머가 읽는 구간과 샘플 짝을 먼저 확인해 주세요.'
    elif review['evidence_type'] == '방향 근거 불일치':
        summary = '조건에 따라 품질이 약한 방향이 달라, F 또는 R을 우선 선택하기 어렵습니다.'
        check = '조건별 겹침 위치와 양쪽 원본 파형을 함께 확인해 주세요.'
    else:
        summary = '한쪽만 다시 하면 해결될 것으로 제안할 만큼 방향별 품질 차이가 뚜렷하지 않습니다.'
        check = '겹침 끝과 차이가 난 위치의 파형을 확인해 주세요. 양방향 재반응이 꼭 필요하다는 뜻도 아닙니다.'
    return {
        'headline': 'F/R 중 재반응 방향 선택 어려움', 'summary': summary,
        'combination': '재반응 방향 선택 보류',
        'next_action': '판정 근거와 원본 파형을 확인한 뒤 재반응 방향을 결정합니다.',
        'check': check,
        'directions': [{'role': role, 'status': '우선순위 판단 보류', 'emphasis': False,
                        'reason': '현재 자료로는 이 방향을 다시 했을 때의 이점을 구분하기 어렵습니다.'}
                       for role in ('F', 'R')],
    }


def render_rerun_review(rows, reads):
    st.subheader("어느 방향을 다시 반응할까요?")
    review = recommend_rerun(reads, rows)
    display = rerun_display(review)
    st.markdown('### ' + display['headline'])
    st.write(display['summary'])
    for column, direction in zip(st.columns(2), display['directions']):
        role = direction['role']
        profile = next(p for p in review['profiles'] if p['direction'] == role)
        with column, st.container(border=True):
            st.markdown('### ' + role + ' 방향')
            if direction['emphasis']:
                st.warning('● ' + direction['status'])
            else:
                st.info(direction['status'])
            st.write(direction['reason'])
            st.caption(profile['file_name'])
    st.write('**검토할 조합: ' + display['combination'] + '**')
    st.write(display['next_action'])
    st.write('**반응 전에 확인:** ' + display['check'])
    st.caption('현재 서열·품질로 제안한 검토안입니다. 재반응 후 실제로 개선되는지는 아직 검증 중입니다.')

    with st.expander("왜 이렇게 안내했나요? · 근거와 F/R 품질 비교", expanded=False):
        for reason in review['reasons']:
            st.write('• ' + reason)
        for column, profile in zip(st.columns(2), review['profiles']):
            with column:
                st.write("**" + profile['direction'] + " 방향**")
                st.caption(profile['file_name'])
                st.write(f"원본 길이: {profile['original_bases']} bp")
                st.write(f"원본에서 Q20 이상인 A/C/G/T: {profile['original_q20']} bp")
                st.write(f"검토한 조건 중 가장 많이 남은 Q20 염기: {profile['max_retained_q20']} bp")
                st.write(f"원본에서 Q20 이상이 끊기지 않고 이어진 최장 구간: {profile['longest_continuous_q20']} bp")
        st.caption("Q20 염기 합계와 연속 구간 길이는 다릅니다. 이 숫자만으로 필요한 구간 전체를 읽었다고 판단하지 않습니다.")
        if review['local_evidence']:
            table = []
            for item in review['local_evidence']:
                f, r = item['directions']['F'], item['directions']['R']
                table.append({"조건": item['condition'], "같은 구간을 비교한 길이(bp)": item['paired_bases'],
                              "겹침의 F Q20 비중": f"{f['q20_fraction']:.0%}",
                              "겹침의 R Q20 비중": f"{r['q20_fraction']:.0%}",
                              "F가 저품질·R이 Q30 이상인 차이(개)": f['weak_differences'],
                              "R이 저품질·F가 Q30 이상인 차이(개)": r['weak_differences']})
            st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
        st.caption("‘차이’는 서로 다른 염기 또는 확정하지 못한 염기 위치입니다. Gap이 표시된 방향만으로 재반응 방향을 정하지 않습니다.")
        st.write("현재 단측 검토 기준: 한쪽이 검토한 모든 조건에서 최소 유효 염기 수에 못 미치거나, "
                 "겹침에서 한쪽 Q20 비중이 50% 미만·반대쪽 80% 이상이고 방향별 품질 차이가 있는 "
                 "염기 위치가 3개 이상인 패턴이 두 조건 이상에서 확인되어야 합니다. "
                 "양쪽의 높은 품질이 서로 충돌하거나 조건별 방향이 다르면 먼저 보류합니다.")
        st.caption("이 수치는 초기 검토용 기준입니다. 같은 파일의 여러 조건은 독립된 검증 표본이 아니며, 재반응 전후 자료로 기준을 개선할 예정입니다.")
    return review


def explain_metrics(row):
    o = row['overlap']
    if not o:
        return []
    return [
        {"살펴본 내용": "같은 구간을 읽은 길이", "이번 결과": f"{o['paired_bases']} bp",
         "뜻": "두 방향에서 공통으로 비교할 수 있는 염기 수입니다. 빈칸은 길이에 포함하지 않습니다."},
        {"살펴본 내용": "전체 일치율", "이번 결과": f"{o['gap_identity']:.1f}%",
         "뜻": "두 서열을 나란히 놓았을 때 같은 염기의 비율입니다. 빈칸도 차이로 계산합니다."},
        {"살펴본 내용": "품질이 좋은 공통 염기", "이번 결과": f"{o['q20_matches']}개",
         "뜻": "F와 R이 같은 염기를 읽었고, 두 방향 모두 Q20 이상인 위치입니다."},
        {"살펴본 내용": "품질이 좋은데 서로 다른 결과", "이번 결과": f"{o['q20_conflicts']}개",
         "뜻": "높은 품질이 차이를 뒷받침하는 위치입니다. 어느 방향이 틀렸다는 뜻은 아닙니다."},
        {"살펴본 내용": "길이를 맞추려고 넣은 빈칸(Gap)", "이번 결과": f"{o['gaps']}개",
         "뜻": "한쪽에 대응되는 염기가 없어 빈칸을 넣었습니다. 실제 삽입·결실로 확정한 것은 아닙니다."},
        {"살펴본 내용": "연결 끝에서 대응하지 않는 염기", "이번 결과": f"{o['junction_unaligned']} bp · 이 중 Q20 이상 {o['junction_q20']} bp",
         "뜻": "서로 이어져야 할 끝부분에 남은 염기입니다. 최종 Contig에서 제거할 길이라는 뜻은 아닙니다."},
    ]


def original_event_positions(row):
    o = row['overlap']
    names = {'F Gap': 'F 쪽 빈칸(길이 차이)', 'R Gap': 'R 쪽 빈칸(길이 차이)',
             '모호한 염기': 'A/C/G/T로 확정하지 못한 염기', '염기 불일치': '서로 다른 염기'}
    records = []
    for e in o['events']:
        f, r = e['f_position'], e['r_position']
        f_original = row['trim_metadata'][0]['start_1based'] + f - 1 if f is not None else None
        if r is not None:
            offset = row['lengths'][1] - r if o['reverse_orientation'] == 'Reverse complement' else r - 1
            r_original = row['trim_metadata'][1]['start_1based'] + offset
        else:
            r_original = None
        records.append({'정렬에서의 위치': e['column'], '관찰한 차이': names.get(e['kind'], e['kind']),
                        '원본 F 위치': str(f_original) if f_original is not None else '빈칸',
                        '원본 R 위치': str(r_original) if r_original is not None else '빈칸',
                        '차이를 뒷받침하는 Q': e['support_q']})
    return records


def render_result(rows, reads):
    review = render_rerun_review(rows, reads)
    st.divider()
    st.subheader("Contig 생성 예측")
    st.caption("F/R 조립 가능성에 대한 예측입니다. BLAST 동정 결과는 별도로 확인해 주세요.")
    joined = [row["condition"]["label"] for row in rows if row["prediction"]["generated"]]
    if joined:
        st.success("생성 예상 조건: " + ", ".join(joined))
    else:
        st.info("검토한 조건에서 Contig 생성이 예상되는 조건은 없습니다.")

    for start in range(0, len(rows), 2):
        for column, row in zip(st.columns(2), rows[start:start + 2]):
            with column, st.container(border=True):
                prediction = row["prediction"]
                st.markdown(f"### {row['condition']['label']}")
                st.metric("예측 결과", prediction["label"])
                if prediction["state"] == "joined":
                    st.success("두 방향을 하나로 연결할 수 있을 것으로 예상")
                elif prediction["state"] == "separate":
                    st.warning("Contig2 · 두 방향이 하나로 연결되지 않을 것으로 예상")
                else:
                    st.warning("No contig · 연결에 쓸 서열이나 품질이 부족")
                if prediction.get('alignment_review'):
                    st.warning('정렬 확인 필요 · 파일이 생성되어도 정상 결합으로 해석하면 안 됩니다.')
                    st.caption(prediction['alignment_review']['message'])
                st.write(f"품질 정리 후 길이: F **{row['lengths'][0]} bp** · R **{row['lengths'][1]} bp**")
                overlap = row["overlap"]
                if overlap:
                    st.write(f"모델 겹침 **{overlap['paired_bases']} bp** · 일치율 **{overlap['gap_identity']:.1f}%**")
                else:
                    st.write("비교할 겹침을 찾지 못했습니다.")

    with st.expander("판정 근거 · 쉽게 보기", expanded=False):
        selected = st.selectbox("설명할 조건", [c.label for c in CONDITIONS], key="details_condition")
        row = next(row for row in rows if row["condition"]["label"] == selected)
        st.write("**이 조건을 이렇게 예상한 이유**")
        for reason in plain_prediction_reasons(row):
            st.write("• " + reason)
        if row['prediction'].get('standard_reasons'):
            st.info("일반 기준에서 부족한 항목이 있지만, 기존 사례에 근거한 보완 기준을 통과한 예측입니다. 실제 조립 확인이 필요합니다.")
        elif row['prediction'].get('applied_corrections'):
            st.info("실제 Contig 정렬 사례를 반영한 제한적 보완 기준을 적용했습니다. 실제 조립 결과로 확인해 주세요.")
        st.write("**숫자를 읽는 방법**")
        st.write("bp는 서열 길이를 세는 단위이고, Q는 염기를 얼마나 확실하게 읽었는지를 나타내는 품질 값입니다. "
                 "여기서는 Q20 이상을 품질 비교 기준으로 사용합니다. Q20 염기 비중은 재반응 성공 확률이 아닙니다.")
        metrics = explain_metrics(row)
        if metrics:
            st.dataframe(pd.DataFrame(metrics), hide_index=True, use_container_width=True)
        st.write("**품질이 낮은 부분을 제외한 결과**")
        for role, meta, count in zip(('F', 'R'), row['trim_metadata'], row['quality_bases']):
            if meta['retained_bases']:
                st.write(f"{role}: 원본 {meta['original_bases']} bp 중 {meta['start_1based']}–{meta['end_1based']}번을 사용했습니다. "
                         f"남은 {meta['retained_bases']} bp 중 Q20 이상 A/C/G/T는 {count} bp입니다.")
            else:
                st.write(f"{role}: 이 조건에서는 품질 정리 후 사용할 서열이 남지 않았습니다.")
        st.caption("조건 16·20·30·10은 QT만 사용하며 New QT는 사용하지 않습니다. 20/10과 30/20의 앞 숫자는 QT, 뒤 숫자는 New QT의 검사 창 크기입니다. 두 번째 품질 기준값이 아닙니다.")

    with st.expander("정렬 상세 · 읽는 법과 기술 지표", expanded=False):
        st.write(f"**현재 보는 조건: {selected}** · 조건은 ‘판정 근거 · 쉽게 보기’에서 바꿀 수 있습니다.")
        st.caption("아래는 프로그램이 비교한 정렬입니다. 실제 조립 출력에서 사용하는 위치와 다를 수 있습니다.")
        o = row['overlap']
        if not o:
            st.write("표시할 정렬이 없습니다. 남은 서열의 길이와 품질을 먼저 확인해 주세요.")
        else:
            if o['reverse_orientation'] == 'Reverse complement':
                st.write("R은 F와 같은 방향에서 비교할 수 있도록 순서를 뒤집고 상보 염기로 바꿨습니다. 원본 파일은 바꾸지 않습니다.")
            else:
                st.write("R을 원본 방향으로 놓았을 때 더 잘 맞아 그 비교를 표시합니다.")
            if o['connection'] in ('F → R', 'R → F'):
                st.write("연결 배치: **" + o['connection'] + "**. 화살표 순서로 두 읽기 구간이 이어지는 배치입니다.")
            else:
                st.write("연결 배치: **" + o['connection'] + "**. 한 읽기 구간이 다른 구간 안에 들어가는 배치입니다.")
            st.write("**정렬 읽는 법:** 위 줄은 F, 아래 줄은 방향을 맞춘 R입니다. "
                     "가운데 `|`는 같은 염기, `.`은 다른 염기, `?`는 확정하기 어려운 염기입니다. "
                     "서열 안의 `-`는 길이를 맞추기 위해 넣은 빈칸(Gap)입니다.")
            st.caption("빈칸이 생긴 방향이 재반응해야 할 방향이라는 뜻은 아닙니다. 이 정렬만으로 Mix·InDel을 확정할 수 없습니다.")
            st.code(alignment_preview(o, width=70), language=None)
            core = o.get('core_alignment')
            if core:
                st.write("**끝부분을 더 엄격하게 비교한 정렬**")
                st.write("같은 입력을 다른 정렬 점수로 다시 비교했습니다. 어느 한쪽을 고쳐서 만든 결과가 아니며, "
                         "끝을 조금 다르게 잡아도 연결 근거가 유지되는지 보는 참고 자료입니다.")
                st.write(f"겹침 {core['paired_bases']} bp · 빈칸 {core['gaps']}개 · 연결 끝에서 대응하지 않는 염기 {core['junction_unaligned']} bp")
                st.code(alignment_preview(core, width=70), language=None)
            if o['events']:
                st.write("**원본 파형에서 확인할 위치**")
                st.caption("위치는 업로드한 AB1의 원본 염기 순서로, 1부터 셉니다. R의 방향 변환과 품질 절단을 되돌려 표시했습니다. 빈칸에는 대응 염기 위치가 없습니다.")
                st.dataframe(pd.DataFrame(original_event_positions(row)), hide_index=True, use_container_width=True)
                st.caption("‘차이를 뒷받침하는 Q’는 양쪽 품질 중 보수적인 값을 사용합니다. 빈칸은 반대편 인접 염기의 품질도 함께 봅니다.")
            anchors, pair = o['gap_outer_q20_anchors'], o.get('opposing_gap_pair')
            st.write(f"빈칸 바깥에서 두 방향 모두 Q20 이상으로 연속 일치한 길이: 앞 {anchors['left']} bp · 뒤 {anchors['right']} bp")
            if pair:
                st.write(f"서로 반대 방향의 빈칸 두 개 사이에서 두 방향 모두 Q20 이상으로 연속 일치한 최장 길이: {pair['between_q20_run']} bp")
            evidence = long_overlap_evidence(o)
            st.write(f"양쪽 Q30 이상이 뒷받침하는 차이: {evidence['q30_conflicts']}개")
            if o['paired_bases'] > MAX_SHORT_GAP_PAIR_BASES:
                st.write("긴 저품질 겹침에 적용하는 보완 기준:")
                st.dataframe(pd.DataFrame([
                    {'기술 항목': name, '현재 기준 충족': '예' if passed else '아니요'}
                    for name, passed in evidence['checks'].items()
                ]), hide_index=True, use_container_width=True)
        st.write("**기존 판정 기준의 상세 기록**")
        for reason in row['prediction']['reasons']:
            st.write("• " + reason)
        if row['prediction'].get('standard_reasons'):
            st.write("보완 기준 적용 전에 부족했던 항목:")
            for reason in row['prediction']['standard_reasons']:
                st.write("• " + reason)
        st.caption("예측 기준은 제공된 실제 사례로 보완한 규칙입니다. 사내 조립기의 내부 규칙과 완전히 같거나 새 샘플에서 검증된 정확도라는 뜻은 아닙니다.")

    with st.expander("AB1 원본 품질 · 위치별 그래프", expanded=False):
        st.write("가로축은 원본 염기 위치, 세로축은 Quality입니다. 값이 낮아지는 위치를 살펴보세요. 이 그래프는 형광 피크 파형 자체가 아닙니다.")
        st.dataframe(quality_summary(reads).rename(columns={"Read": "방향", "길이": "원본 길이(bp)"}),
                     hide_index=True, use_container_width=True)
        for column, read in zip(st.columns(2), reads):
            with column:
                st.caption(read.file_name)
                frame = pd.DataFrame({"원본 염기 위치": range(1, len(read.qualities) + 1), "Quality": read.qualities})
                st.line_chart(frame.set_index("원본 염기 위치"))

    st.download_button("분석·재반응 검토 결과 다운로드 (CSV)",
                       pd.DataFrame(review_export(rows, review)).to_csv(index=False).encode("utf-8-sig"),
                       file_name="contig_prediction.csv", mime="text/csv", key="result_download")
    snapshot = {'schema_version': 2, 'app_version': APP_VERSION,
                'recommendation': review, 'display': rerun_display(review), 'conditions': rows,
                'note': 'AB1의 서열·품질에 근거한 검토 기록입니다. 실제 재반응 효과를 검증한 기록은 아닙니다.'}
    st.download_button("후속 비교용 분석 기록 다운로드 (JSON)",
                       json.dumps(snapshot, ensure_ascii=False, indent=2).encode('utf-8'),
                       file_name="contig_review_record.json", mime="application/json", key="review_download")
    st.caption("재반응 전 AB1, 재반응 후 AB1, 실제 조립 결과와 이 기록을 함께 남기면 이후 판정 근거를 비교하기 좋습니다.")


def main():
    st.set_page_config(page_title="Contig Simulator", page_icon="🧬", layout="wide")
    st.title("🧬 Contig Simulator")
    st.write("F/R AB1 파일을 업로드하면 여섯 조건의 Contig 생성 여부와 재반응 검토 방향을 확인할 수 있습니다.")
    st.caption("작업 시작 기준: 16S는 16, 18S·26S·ITS는 20. 조건 20·30은 New QT를 사용하지 않습니다.")
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
    result_key = "ab1_comparison_v10"
    clicked = st.button("Contig 분석", type="primary", disabled=not ready, key="analyze_button")
    if clicked:
        st.session_state.pop(result_key, None)
        try:
            with st.spinner("여섯 조건의 Contig 생성 여부를 분석하고 있습니다…"):
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
