import io

import pandas as pd
import streamlit as st
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Align import PairwiseAligner


# --------------------------------------------------
# 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="Contig QC Demo",
    page_icon="🧬",
    layout="wide",
)


# --------------------------------------------------
# AB1 파일 읽기
# --------------------------------------------------
def read_ab1(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    file_handle = io.BytesIO(file_bytes)

    record = SeqIO.read(file_handle, "abi")

    sequence = str(record.seq).upper()
    quality_scores = list(
        record.letter_annotations.get("phred_quality", [])
    )

    return {
        "file_name": uploaded_file.name,
        "record_id": record.id,
        "sequence": sequence,
        "quality_scores": quality_scores,
    }


# --------------------------------------------------
# AB1 요약
# --------------------------------------------------
def make_summary(read_data, direction):
    sequence = read_data["sequence"]
    qualities = read_data["quality_scores"]

    if qualities:
        average_quality = sum(qualities) / len(qualities)
        q20_count = sum(q >= 20 for q in qualities)
        q30_count = sum(q >= 30 for q in qualities)
    else:
        average_quality = None
        q20_count = 0
        q30_count = 0

    return {
        "방향": direction,
        "파일명": read_data["file_name"],
        "Read 길이": len(sequence),
        "평균 Quality": (
            round(average_quality, 2)
            if average_quality is not None
            else "없음"
        ),
        "Q20 이상 염기 수": q20_count,
        "Q30 이상 염기 수": q30_count,
    }


# --------------------------------------------------
# Reverse-complement 처리
# --------------------------------------------------
def make_reverse_complement(sequence, qualities):
    reverse_complement_sequence = str(
        Seq(sequence).reverse_complement()
    )

    # 서열의 방향이 뒤집히므로 Quality 순서도 뒤집음
    reverse_complement_qualities = qualities[::-1]

    return (
        reverse_complement_sequence,
        reverse_complement_qualities,
    )


# --------------------------------------------------
# Quality 그래프
# --------------------------------------------------
def show_quality_chart(read_data, direction):
    quality_scores = read_data["quality_scores"]

    if not quality_scores:
        st.warning(
            f"{direction} 파일에서 Quality 정보를 찾지 못했습니다."
        )
        return

    chart_data = pd.DataFrame(
        {
            "Base position": range(
                1,
                len(quality_scores) + 1,
            ),
            "Quality": quality_scores,
        }
    )

    chart_data = chart_data.set_index("Base position")

    st.subheader(f"{direction} Quality 분포")
    st.line_chart(chart_data)

# --------------------------------------------------
# Alignment용 서열 정리
# --------------------------------------------------
def normalize_alignment_sequence(sequence):
    """
    A, C, G, T 이외의 염기는 N으로 변환합니다.
    서열 길이와 위치는 유지됩니다.
    """

    return "".join(
        base if base in "ACGT" else "N"
        for base in sequence.upper()
    )


# --------------------------------------------------
# Quality-weighted terminal overlap 분석
# --------------------------------------------------
TERMINAL_SEARCH_BASES = 700
GAP_QUALITY_WEIGHT = 0.5
JUNCTION_ANCHOR_BASES = 60


def quality_weight(quality):
    """Phred quality를 0~1 범위의 경험적 가중치로 변환합니다."""

    bounded_quality = min(max(float(quality), 0.0), 40.0)
    return bounded_quality / 40.0


def normalize_quality_scores(sequence, qualities):
    """Quality 길이를 서열 길이에 맞추고 누락값은 Q0으로 채웁니다."""

    normalized = list(qualities or [])[: len(sequence)]

    if len(normalized) < len(sequence):
        normalized.extend([0] * (len(sequence) - len(normalized)))

    return normalized


def build_terminal_candidate(
    left_sequence,
    left_qualities,
    right_sequence,
    right_qualities,
    left_label,
    right_label,
    qt_threshold,
    new_qt_threshold,
):
    """
    왼쪽 read의 suffix와 오른쪽 read의 prefix만 검색합니다.
    정렬 밖의 junction 인접 염기는 삭제하지 않고 soft-clip 후보로
    유지한 뒤 Quality를 이용해 부담을 계산합니다.
    """

    left_window_start = max(
        0,
        len(left_sequence) - TERMINAL_SEARCH_BASES,
    )

    left_window = left_sequence[left_window_start:]
    right_window = right_sequence[:TERMINAL_SEARCH_BASES]

    if not left_window or not right_window:
        return None

    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2
    aligner.mismatch_score = -3
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -1
    aligner.wildcard = "N"

    alignments = aligner.align(
        normalize_alignment_sequence(left_window),
        normalize_alignment_sequence(right_window),
    )

    try:
        alignment = alignments[0]
    except IndexError:
        return None

    coordinates = alignment.coordinates

    left_start = (
        left_window_start + int(coordinates[0, 0])
    )
    left_end = (
        left_window_start + int(coordinates[0, -1])
    )
    right_start = int(coordinates[1, 0])
    right_end = int(coordinates[1, -1])

    aligned_left = str(alignment[0])
    aligned_right = str(alignment[1])
    indices = alignment.indices

    markers = []
    matches = 0
    mismatches = 0
    gaps = 0
    ambiguous = 0

    match_weight_total = 0.0
    mismatch_weight_total = 0.0
    gap_weight_total = 0.0
    ambiguous_weight_total = 0.0

    qt_supported_matches = 0
    qt_supported_mismatches = 0
    qt_supported_gaps = 0
    qt_supported_ambiguous = 0
    qt_supported_longest_match_run = 0
    qt_supported_current_match_run = 0

    for left_index, right_index in zip(
        indices[0],
        indices[1],
    ):
        if left_index < 0 or right_index < 0:
            gaps += 1
            markers.append(" ")
            qt_supported_current_match_run = 0

            if left_index >= 0:
                full_left_index = (
                    left_window_start + int(left_index)
                )
                gap_quality = left_qualities[full_left_index]
            else:
                gap_quality = right_qualities[int(right_index)]

            gap_weight_total += (
                quality_weight(gap_quality)
                * GAP_QUALITY_WEIGHT
            )

            if gap_quality >= qt_threshold:
                qt_supported_gaps += 1

            continue

        full_left_index = (
            left_window_start + int(left_index)
        )
        full_right_index = int(right_index)

        left_base = left_sequence[full_left_index]
        right_base = right_sequence[full_right_index]
        paired_quality = min(
            left_qualities[full_left_index],
            right_qualities[full_right_index],
        )
        paired_weight = quality_weight(paired_quality)

        if left_base not in "ACGT" or right_base not in "ACGT":
            ambiguous += 1
            ambiguous_weight_total += paired_weight
            markers.append("?")
            qt_supported_current_match_run = 0

            if paired_quality >= qt_threshold:
                qt_supported_ambiguous += 1

        elif left_base == right_base:
            matches += 1
            match_weight_total += paired_weight
            markers.append("|")

            if paired_quality >= qt_threshold:
                qt_supported_matches += 1
                qt_supported_current_match_run += 1
                qt_supported_longest_match_run = max(
                    qt_supported_longest_match_run,
                    qt_supported_current_match_run,
                )
            else:
                qt_supported_current_match_run = 0

        else:
            mismatches += 1
            mismatch_weight_total += paired_weight
            markers.append(".")
            qt_supported_current_match_run = 0

            if paired_quality >= qt_threshold:
                qt_supported_mismatches += 1

    paired_bases = matches + mismatches + ambiguous
    alignment_columns = len(aligned_left)

    base_identity = (
        matches / paired_bases * 100
        if paired_bases > 0
        else 0.0
    )
    gap_included_identity = (
        matches / alignment_columns * 100
        if alignment_columns > 0
        else 0.0
    )
    gap_rate = (
        gaps / alignment_columns * 100
        if alignment_columns > 0
        else 0.0
    )

    weighted_denominator = (
        match_weight_total
        + mismatch_weight_total
        + gap_weight_total
        + ambiguous_weight_total
    )

    quality_weighted_identity = (
        match_weight_total / weighted_denominator * 100
        if weighted_denominator > 0
        else 0.0
    )

    left_tail_qualities = left_qualities[left_end:]
    right_head_qualities = right_qualities[:right_start]
    terminal_qualities = (
        left_tail_qualities + right_head_qualities
    )
    terminal_slack_total = len(terminal_qualities)

    # 정렬 경계 안쪽의 품질을 read별로 따로 평가합니다. 이 값은
    # contig 성공 여부가 아니라 어느 방향의 재반응이 더 유리한지
    # 판단하는 보조 근거로 사용합니다.
    left_anchor_qualities = left_qualities[
        max(left_start, left_end - JUNCTION_ANCHOR_BASES):left_end
    ]
    right_anchor_qualities = right_qualities[
        right_start:min(
            right_end,
            right_start + JUNCTION_ANCHOR_BASES,
        )
    ]

    def mean_quality(values):
        return sum(values) / len(values) if values else 0.0

    def threshold_fraction(values, threshold):
        if not values:
            return 0.0

        return sum(value >= threshold for value in values) / len(values)

    new_qt_enabled = new_qt_threshold is not None
    terminal_quality_threshold = (
        new_qt_threshold
        if new_qt_enabled
        else qt_threshold
    )

    if terminal_quality_threshold > 0:
        terminal_high_quality_bases = sum(
            quality >= terminal_quality_threshold
            for quality in terminal_qualities
        )
        terminal_quality_burden = sum(
            max(quality - terminal_quality_threshold + 1, 0)
            / max(41 - terminal_quality_threshold, 1)
            for quality in terminal_qualities
        )
    else:
        # 품질 경계가 0인 예외 조건은 모든 말단 염기를 유지합니다.
        terminal_high_quality_bases = terminal_slack_total
        terminal_quality_burden = float(terminal_slack_total)

    if terminal_quality_threshold > 0:
        left_terminal_high_quality = sum(
            quality >= terminal_quality_threshold
            for quality in left_tail_qualities
        )
        right_terminal_high_quality = sum(
            quality >= terminal_quality_threshold
            for quality in right_head_qualities
        )
    else:
        left_terminal_high_quality = len(left_tail_qualities)
        right_terminal_high_quality = len(right_head_qualities)

    left_terminal_low_quality = (
        len(left_tail_qualities) - left_terminal_high_quality
    )
    right_terminal_low_quality = (
        len(right_head_qualities) - right_terminal_high_quality
    )

    left_source = "Forward" if left_label == "Forward" else "Reverse"
    right_source = "Forward" if right_label == "Forward" else "Reverse"

    junction_metrics_by_source = {
        left_source: {
            "softclip": len(left_tail_qualities),
            "high_quality_softclip": left_terminal_high_quality,
            "low_quality_softclip": left_terminal_low_quality,
            "anchor_mean_quality": mean_quality(left_anchor_qualities),
            "anchor_qt_fraction": threshold_fraction(
                left_anchor_qualities,
                qt_threshold,
            ),
        },
        right_source: {
            "softclip": len(right_head_qualities),
            "high_quality_softclip": right_terminal_high_quality,
            "low_quality_softclip": right_terminal_low_quality,
            "anchor_mean_quality": mean_quality(right_anchor_qualities),
            "anchor_qt_fraction": threshold_fraction(
                right_anchor_qualities,
                qt_threshold,
            ),
        },
    }

    forward_junction = junction_metrics_by_source.get(
        "Forward",
        {},
    )
    reverse_junction = junction_metrics_by_source.get(
        "Reverse",
        {},
    )

    terminal_low_quality_fraction = (
        (
            terminal_slack_total
            - terminal_high_quality_bases
        )
        / terminal_slack_total
        if terminal_slack_total > 0
        else 1.0
    )

    qt_supported_conflicts = (
        qt_supported_mismatches
        + qt_supported_gaps
        + qt_supported_ambiguous
    )

    if left_label == "Forward":
        forward_start = left_start
        forward_end = left_end
        reverse_start = right_start
        reverse_end = right_end
    else:
        reverse_start = left_start
        reverse_end = left_end
        forward_start = right_start
        forward_end = right_end

    candidate_is_usable = (
        paired_bases >= 20
        and quality_weighted_identity >= 80
        and qt_supported_matches >= 10
    )

    return {
        "score": round(float(alignment.score), 2),
        "paired_bases": paired_bases,
        "alignment_columns": alignment_columns,
        "matches": matches,
        "mismatches": mismatches,
        "gaps": gaps,
        "ambiguous": ambiguous,
        "identity": round(base_identity, 2),
        "base_identity": round(base_identity, 2),
        "gap_included_identity": round(
            gap_included_identity,
            2,
        ),
        "quality_weighted_identity": round(
            quality_weighted_identity,
            2,
        ),
        "gap_rate": round(gap_rate, 2),
        "qt_supported_matches": qt_supported_matches,
        "qt_supported_longest_match_run": (
            qt_supported_longest_match_run
        ),
        "qt_supported_mismatches": qt_supported_mismatches,
        "qt_supported_gaps": qt_supported_gaps,
        "qt_supported_ambiguous": qt_supported_ambiguous,
        "qt_supported_conflicts": qt_supported_conflicts,
        "forward_start": forward_start,
        "forward_end": forward_end,
        "reverse_start": reverse_start,
        "reverse_end": reverse_end,
        "forward_head_unaligned": forward_start,
        "forward_tail_unaligned": (
            len(right_sequence if right_label == "Forward" else left_sequence)
            - forward_end
        ),
        "reverse_head_unaligned": reverse_start,
        "reverse_tail_unaligned": (
            len(right_sequence if right_label == "Reverse-complement" else left_sequence)
            - reverse_end
        ),
        "connection_direction": (
            f"{left_label} → {right_label}"
        ),
        "left_sequence": left_label,
        "right_sequence": right_label,
        "left_tail_unaligned": len(left_tail_qualities),
        "right_head_unaligned": len(right_head_qualities),
        "terminal_slack_total": terminal_slack_total,
        "terminal_high_quality_bases": (
            terminal_high_quality_bases
        ),
        "terminal_quality_burden": round(
            terminal_quality_burden,
            2,
        ),
        "terminal_low_quality_fraction": round(
            terminal_low_quality_fraction,
            4,
        ),
        "new_qt_enabled": new_qt_enabled,
        "terminal_quality_threshold": terminal_quality_threshold,
        "forward_junction_softclip": forward_junction.get(
            "softclip",
            0,
        ),
        "reverse_junction_softclip": reverse_junction.get(
            "softclip",
            0,
        ),
        "forward_junction_high_quality_softclip": (
            forward_junction.get("high_quality_softclip", 0)
        ),
        "reverse_junction_high_quality_softclip": (
            reverse_junction.get("high_quality_softclip", 0)
        ),
        "forward_junction_low_quality_softclip": (
            forward_junction.get("low_quality_softclip", 0)
        ),
        "reverse_junction_low_quality_softclip": (
            reverse_junction.get("low_quality_softclip", 0)
        ),
        "forward_junction_anchor_mean_quality": round(
            forward_junction.get("anchor_mean_quality", 0.0),
            2,
        ),
        "reverse_junction_anchor_mean_quality": round(
            reverse_junction.get("anchor_mean_quality", 0.0),
            2,
        ),
        "forward_junction_anchor_qt_fraction": round(
            forward_junction.get("anchor_qt_fraction", 0.0),
            4,
        ),
        "reverse_junction_anchor_qt_fraction": round(
            reverse_junction.get("anchor_qt_fraction", 0.0),
            4,
        ),
        "candidate_is_usable": candidate_is_usable,
        "aligned_left": aligned_left,
        "aligned_right": aligned_right,
        "aligned_forward": aligned_left,
        "aligned_reverse": aligned_right,
        "markers": "".join(markers),
    }


def analyze_overlap(
    forward_sequence,
    forward_qualities,
    reverse_complement_sequence,
    reverse_complement_qualities,
    qt_threshold=30,
    new_qt_threshold=20,
    reverse_label="Reverse-complement",
):
    """
    원본 read를 hard trimming하지 않고 두 연결 방향의 terminal
    overlap을 각각 평가한 뒤 Quality 부담이 낮은 후보를 선택합니다.
    """

    if not forward_sequence or not reverse_complement_sequence:
        return None

    forward_sequence = normalize_alignment_sequence(
        forward_sequence
    )
    reverse_complement_sequence = normalize_alignment_sequence(
        reverse_complement_sequence
    )

    forward_qualities = normalize_quality_scores(
        forward_sequence,
        forward_qualities,
    )
    reverse_complement_qualities = normalize_quality_scores(
        reverse_complement_sequence,
        reverse_complement_qualities,
    )

    candidates = [
        build_terminal_candidate(
            forward_sequence,
            forward_qualities,
            reverse_complement_sequence,
            reverse_complement_qualities,
            "Forward",
            reverse_label,
            qt_threshold,
            new_qt_threshold,
        ),
        build_terminal_candidate(
            reverse_complement_sequence,
            reverse_complement_qualities,
            forward_sequence,
            forward_qualities,
            reverse_label,
            "Forward",
            qt_threshold,
            new_qt_threshold,
        ),
    ]

    candidates = [
        candidate
        for candidate in candidates
        if candidate is not None
    ]

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda candidate: (
            0 if candidate["candidate_is_usable"] else 1,
            candidate["terminal_high_quality_bases"],
            candidate["terminal_quality_burden"],
            -candidate["quality_weighted_identity"],
            -candidate["qt_supported_matches"],
            -candidate["paired_bases"],
        ),
    )


def analyze_best_reverse_orientation(
    forward_sequence,
    forward_qualities,
    reverse_sequence,
    reverse_qualities,
    qt_threshold=30,
    new_qt_threshold=20,
):
    """
    Reverse 파일의 원본 방향과 reverse-complement 방향을 모두
    분석합니다. 파일명이나 primer 표기 대신 실제 서열 정렬
    근거가 더 좋은 방향을 자동 선택합니다.
    """

    reverse_complement, reverse_complement_qualities = (
        make_reverse_complement(
            reverse_sequence,
            reverse_qualities,
        )
    )

    raw_candidate = analyze_overlap(
        forward_sequence,
        forward_qualities,
        reverse_sequence,
        reverse_qualities,
        qt_threshold,
        new_qt_threshold,
        reverse_label="Reverse read (원본)",
    )

    reverse_complement_candidate = analyze_overlap(
        forward_sequence,
        forward_qualities,
        reverse_complement,
        reverse_complement_qualities,
        qt_threshold,
        new_qt_threshold,
        reverse_label="Reverse-complement",
    )

    candidates = []

    if raw_candidate is not None:
        raw_candidate["reverse_orientation"] = (
            "원본 Reverse 방향 사용"
        )
        candidates.append(raw_candidate)

    if reverse_complement_candidate is not None:
        reverse_complement_candidate["reverse_orientation"] = (
            "Reverse-complement 적용"
        )
        candidates.append(reverse_complement_candidate)

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda candidate: (
            0 if candidate["candidate_is_usable"] else 1,
            candidate["terminal_high_quality_bases"],
            candidate["terminal_quality_burden"],
            candidate["terminal_slack_total"],
            -candidate["base_identity"],
            -candidate["quality_weighted_identity"],
            -candidate["paired_bases"],
        ),
    )


# --------------------------------------------------
# 회사 프로그램 조건 프리셋
# --------------------------------------------------
COMPANY_CONDITIONS = [
    "16",
    "20",
    "20/10",
    "20/20",
    "30/10",
    "30/20",
    "30/40",
    "10",
]

CONDITION_THRESHOLDS = {
    # None은 New QT 체크 해제(미사용)를 의미합니다.
    "16": (16, None),
    "10": (10, None),
    "20": (20, None),
    "20/10": (20, 10),
    "20/20": (20, 20),
    "30/10": (30, 10),
    "30/20": (30, 20),
    "30/40": (30, 40),
}

# 30/20에서 저품질 말단이 제거 가능하더라도, 짧은 overlap만으로
# F/R 결합 성공을 과대 판정하지 않도록 하는 경험적 하한입니다.
# 현재 성공 사례(KHS2: 75 bp)와 Contig2 사례
# (WT: 60 bp, KNIBR033: 69 bp)를 기준으로 보정했습니다.
MIN_DIRECT_30_20_OVERLAP = 70
MIN_CONTIG2_OVERLAP = 40

# New QT를 사용하지 않는 QT16 조건은 낮은 품질 말단을 별도로
# 확장/제외하는 보조 단계가 없습니다. 70 bp 미만의 overlap은 더
# 엄격한 gap 기준을 만족하는 경우에만 허용하고, 70 bp 이상도 gap
# 포함 Identity 하한을 적용합니다. MMC(68 bp/gap 4, 실제 미결합)와
# KCKM1002-4(gap 포함 Identity 87.39%, 실제 미결합) 사례로
# 보정했습니다.
MIN_DIRECT_QT16_OVERLAP = 70
MIN_QT16_GAP_INCLUDED_IDENTITY = 90
MIN_SHORT_QT16_OVERLAP = 40
MIN_SHORT_QT16_GAP_INCLUDED_IDENTITY = 94
MAX_SHORT_QT16_GAPS = 3

# 현재까지 확인된 열두 AB1 쌍의 실제 결과로 보정한 조건별
# 출력 유형입니다.
# 20계열은 terminal overlap이 좋아 보여도 성공으로 자동 승격하지
# 않고 F/R 개별 출력(Contig2)을 우선합니다.
CONTIG2_CALIBRATED_CONDITIONS = {
    "10",
    "20",
    "20/10",
    "20/20",
}

NO_CONTIG_CALIBRATED_CONDITIONS = {
    "30/10",
    "30/40",
}

STANDARD_CONDITION_BY_VALUES = {
    thresholds: label
    for label, thresholds in CONDITION_THRESHOLDS.items()
}


def parse_company_condition(condition_label):
    if condition_label not in CONDITION_THRESHOLDS:
        raise ValueError(
            f"지원하지 않는 조건입니다: {condition_label}"
        )

    return CONDITION_THRESHOLDS[condition_label]


def format_condition_label(qt_value, new_qt_value):
    """내부 New QT 미사용값(None)은 화면에서 QT만 표시합니다."""

    if new_qt_value is None:
        return str(qt_value)

    return f"{qt_value}/{new_qt_value}"


def standard_condition_label(qt_value, new_qt_value):
    """현재 실제 결과로 보정된 표준 조건명을 반환합니다."""

    return STANDARD_CONDITION_BY_VALUES.get(
        (qt_value, new_qt_value)
    )


def has_strong_deep_internal_structure(overlap_result):
    """긴 양쪽 overhang 사이에 신뢰 가능한 내부 overlap이 있는지 확인합니다."""

    if overlap_result is None:
        return False

    qt_matches = overlap_result["qt_supported_matches"]
    qt_conflicts = overlap_result["qt_supported_conflicts"]
    forward_softclip = overlap_result[
        "forward_junction_softclip"
    ]
    reverse_softclip = overlap_result[
        "reverse_junction_softclip"
    ]

    return (
        overlap_result["paired_bases"] >= 250
        and overlap_result["base_identity"] >= 98
        and overlap_result["gap_included_identity"] >= 94
        and overlap_result["quality_weighted_identity"] >= 96
        and overlap_result["gap_rate"] <= 5
        and qt_matches >= 180
        and qt_conflicts <= max(15, int(qt_matches * 0.08))
        and 200 <= forward_softclip <= 500
        and 200 <= reverse_softclip <= 500
        and overlap_result["terminal_slack_total"] <= 900
        and overlap_result.get("reverse_orientation")
        == "Reverse-complement 적용"
    )


def deep_internal_overlap_mode(
    overlap_result,
    qt_threshold,
    new_qt_threshold,
):
    """
    두 read 말단에 긴 read-through가 남아 있지만 내부에 매우 강한
    overlap이 있는 유형을 판별합니다. Primer명이 아니라 정렬 구조와
    junction anchor 품질만 사용합니다.
    """

    if overlap_result is None or new_qt_threshold is None:
        return None

    longest_run = overlap_result[
        "qt_supported_longest_match_run"
    ]
    minimum_anchor_quality = min(
        overlap_result["forward_junction_anchor_mean_quality"],
        overlap_result["reverse_junction_anchor_mean_quality"],
    )

    if not has_strong_deep_internal_structure(overlap_result):
        return None

    # WT-M13처럼 junction anchor가 매우 깨끗하면 QT20/New QT10의
    # 긴 연속 match로 내부 overlap을 사용할 수 있습니다.
    if (
        20 <= qt_threshold < 25
        and 10 <= new_qt_threshold <= 12
        and minimum_anchor_quality >= 38
        and longest_run >= 40
    ):
        return "고품질 장거리 내부 overlap"

    # A-B21-M13처럼 anchor 품질이 상대적으로 낮으면 QT를 높여
    # 노이즈를 억제한 조건에서 충분한 연속 match가 남아야 합니다.
    if (
        25 <= qt_threshold <= 30
        and 15 <= new_qt_threshold <= 25
        and minimum_anchor_quality >= 30
        and longest_run >= 20
    ):
        return "고QT 장거리 내부 overlap"

    return None


def deep_internal_contig2_candidate(
    overlap_result,
    qt_threshold,
    new_qt_threshold,
):
    """20/10 부근에서 내부 overlap은 강하지만 anchor가 경계인 유형입니다."""

    if overlap_result is None or new_qt_threshold is None:
        return False

    longest_run = overlap_result[
        "qt_supported_longest_match_run"
    ]
    minimum_anchor_quality = min(
        overlap_result["forward_junction_anchor_mean_quality"],
        overlap_result["reverse_junction_anchor_mean_quality"],
    )

    return (
        has_strong_deep_internal_structure(overlap_result)
        and 20 <= qt_threshold < 25
        and 10 <= new_qt_threshold <= 12
        and minimum_anchor_quality >= 38
        and 30 <= longest_run < 40
    )


# --------------------------------------------------
# Contig 생성 결과 판정
# --------------------------------------------------
def classify_contig_prediction(
    overlap_result,
    qt_threshold,
    new_qt_threshold,
    condition_label,
):
    """
    서열 기반 overlap 근거와 실제 회사 프로그램 조건별 결과를
    함께 사용하는 경험적 분류입니다.
    """

    if overlap_result is None:
        return {
            "status": "No contig 예상",
            "rank": 0,
            "reason": "terminal overlap 후보를 찾지 못함",
        }

    overlap_length = overlap_result["paired_bases"]
    gap_included_identity = overlap_result[
        "gap_included_identity"
    ]
    weighted_identity = overlap_result[
        "quality_weighted_identity"
    ]
    qt_matches = overlap_result["qt_supported_matches"]
    qt_longest_match_run = overlap_result[
        "qt_supported_longest_match_run"
    ]
    qt_conflicts = overlap_result["qt_supported_conflicts"]
    total_slack = overlap_result["terminal_slack_total"]
    terminal_high_quality = overlap_result[
        "terminal_high_quality_bases"
    ]
    low_quality_fraction = overlap_result[
        "terminal_low_quality_fraction"
    ]
    internal_overlap_mode = deep_internal_overlap_mode(
        overlap_result,
        qt_threshold,
        new_qt_threshold,
    )
    internal_contig2_candidate = deep_internal_contig2_candidate(
        overlap_result,
        qt_threshold,
        new_qt_threshold,
    )
    new_qt_enabled = new_qt_threshold is not None
    boundary_label = (
        f"New QT {new_qt_threshold}"
        if new_qt_enabled
        else f"QT {qt_threshold}"
    )

    structural_fail_reasons = []

    if overlap_length < 20:
        structural_fail_reasons.append("overlap 20 bp 미만")

    if weighted_identity < 82:
        structural_fail_reasons.append(
            "Quality 가중 Identity 82% 미만"
        )

    if qt_matches < 10:
        structural_fail_reasons.append(
            f"Q{qt_threshold} 지지 match 10 bp 미만"
        )

    conflict_limit = max(25, int(qt_matches * 0.45))

    if qt_conflicts > conflict_limit:
        structural_fail_reasons.append(
            f"고품질 mismatch/gap {qt_conflicts}개"
        )

    if (
        total_slack > 350
        and internal_overlap_mode is None
        and not internal_contig2_candidate
    ):
        structural_fail_reasons.append(
            f"junction soft-clip 후보 {total_slack} bp 초과"
        )

    if (
        new_qt_enabled
        and internal_overlap_mode is None
        and not internal_contig2_candidate
        and terminal_high_quality
        > max(100, int(overlap_length * 0.75))
    ):
        structural_fail_reasons.append(
            f"{boundary_label} 이상 말단 염기 "
            f"{terminal_high_quality}개"
        )

    if structural_fail_reasons:
        return {
            "status": "No contig 예상",
            "rank": 0,
            "reason": "; ".join(structural_fail_reasons),
        }

    # 실제 보정 결과상 아래 조건은 contig 결과가 생성되지
    # 않았습니다. 후속 사례가 쌓이면 이 prior를 재보정합니다.
    if condition_label in NO_CONTIG_CALIBRATED_CONDITIONS:
        return {
            "status": "No contig 예상",
            "rank": 0,
            "reason": (
                f"보정 사례에서 조건 {condition_label}은 No contig"
            ),
        }

    tight_conflict_limit = max(
        20,
        int(qt_matches * 0.35),
    )

    # 양 read가 junction에서 거의 바로 만나고 염기 일치도가
    # 높은 구조입니다. 이 값은 구조적 지지 근거일 뿐, 모든 QT
    # 조건을 성공으로 승격시키는 독립 기준으로 사용하지 않습니다.
    tight_terminal_overlap = (
        overlap_length >= 25
        and overlap_result["base_identity"] >= 97
        and weighted_identity >= 85
        and qt_matches >= 20
        and overlap_result["mismatches"] <= 2
        and qt_conflicts <= tight_conflict_limit
        and total_slack <= 20
    )

    pass_conflict_limit = max(8, int(qt_matches * 0.15))

    terminal_pass = (
        terminal_high_quality <= 30
        and (
            total_slack <= 20
            or low_quality_fraction >= 0.65
        )
    )

    pass_conditions = (
        overlap_length >= 25
        and weighted_identity >= 92
        and qt_matches >= 20
        and qt_conflicts <= pass_conflict_limit
        and terminal_pass
    )

    # S-A 및 PW-41의 30/20 성공 사례처럼 junction에 비교적 긴
    # 말단이 남아도 대부분이 New QT 미만이면 회사 프로그램이
    # 저품질 말단을 제외한 뒤 조립하는 유형으로 해석합니다.
    low_quality_terminal_rescue = (
        overlap_length >= 200
        and overlap_result["base_identity"] >= 98
        and weighted_identity >= 92
        and qt_matches >= 50
        and qt_conflicts <= max(8, int(qt_matches * 0.15))
        and 60 <= total_slack <= 180
        and terminal_high_quality <= 30
        and low_quality_fraction >= 0.70
    )

    reverse_orientation = overlap_result.get(
        "reverse_orientation",
        "",
    )

    # 현재 확인된 NS1/NS24 및 785F/907R 성공 패턴입니다.
    # 짧은 overlap은 말단 품질이 좋아도 회사 프로그램에서
    # Contig2로 남을 수 있으므로, 직접 결합형에는 최소 70 bp의
    # overlap을 요구합니다. 긴 저품질 말단 rescue 유형은 별도로
    # 평가합니다.
    direct_30_20_success = (
        overlap_length >= MIN_DIRECT_30_20_OVERLAP
        and qt_longest_match_run >= 8
        and (pass_conditions or tight_terminal_overlap)
    )

    if (
        condition_label == "30/20"
        and reverse_orientation == "Reverse-complement 적용"
        and (
            direct_30_20_success
            or low_quality_terminal_rescue
            or internal_overlap_mode is not None
        )
    ):
        if internal_overlap_mode is not None:
            success_basis = (
                f"30/20 {internal_overlap_mode} 성공 패턴 충족; "
                f"Q{qt_threshold} 최장 연속 match "
                f"{qt_longest_match_run} bp"
            )
        elif low_quality_terminal_rescue:
            success_basis = (
                "30/20 저품질 말단 soft-clip 성공 패턴 충족; "
                f"말단 저품질 비율 {low_quality_fraction * 100:.1f}%"
            )
        else:
            success_basis = (
                "30/20 연속 Quality anchor 성공 패턴 충족; "
                f"Q{qt_threshold} 최장 연속 match "
                f"{qt_longest_match_run} bp"
            )

        return {
            "status": "F+R 결합 성공 예상",
            "rank": 2,
            "reason": success_basis,
        }

    # WT-260831-28 및 KNIBR033의 실제 30/20 Contig2 패턴입니다.
    # Quality와 junction은 양호하지만 overlap 자체가 70 bp보다
    # 짧아 하나의 안정적인 consensus로 승격되기에는 근거가
    # 부족한 경우 F/R 개별 출력으로 분류합니다.
    short_overlap_contig2 = (
        condition_label == "30/20"
        and reverse_orientation == "Reverse-complement 적용"
        and MIN_CONTIG2_OVERLAP
        <= overlap_length
        < MIN_DIRECT_30_20_OVERLAP
        and qt_longest_match_run >= 8
        and (pass_conditions or tight_terminal_overlap)
    )

    if short_overlap_contig2:
        return {
            "status": "Contig2 예상",
            "rank": 1,
            "reason": (
                "30/20 정렬 근거는 있으나 terminal overlap "
                f"{overlap_length} bp로 {MIN_DIRECT_30_20_OVERLAP} "
                "bp 미만; F/R 개별 출력 보정 패턴"
            ),
        }

    # 긴 read-through가 양쪽에 남은 read 쌍에서 내부 overlap이
    # 매우 강하고 junction anchor가 깨끗한 경우입니다. WT-M13의
    # 실제 20/10 성공 사례로 보정했지만 primer명은 사용하지 않아
    # 같은 구조의 다른 서비스에도 적용됩니다.
    if (
        condition_label == "20/10"
        and internal_overlap_mode is not None
    ):
        return {
            "status": "F+R 결합 성공 예상",
            "rank": 2,
            "reason": (
                f"20/10 {internal_overlap_mode} 성공 패턴 충족; "
                f"Q{qt_threshold} 최장 연속 match "
                f"{qt_longest_match_run} bp"
            ),
        }

    if (
        condition_label == "20/10"
        and internal_contig2_candidate
    ):
        return {
            "status": "Contig2 예상",
            "rank": 1,
            "reason": (
                "장거리 내부 overlap은 강하지만 Q20 최장 연속 "
                f"match {qt_longest_match_run} bp로 성공 기준 "
                "40 bp 미만; F/R 개별 출력 예상"
            ),
        }

    # 16S 기본값인 QT16 단독 조건입니다. New QT를 사용하지 않고
    # QT16을 말단 품질 경계로 삼아 구조적 overlap을 평가합니다.
    # New QT의 말단 보정이 없으므로 70 bp 이상의 직접 overlap과
    # gap 포함 Identity 90% 이상을 함께 요구합니다. 다만 기존에
    # 일치했던 짧고 깨끗한 overlap 사례는 gap 포함 Identity 94%
    # 이상, gap 3개 이하일 때만 별도로 허용합니다.
    qt16_direct_overlap = (
        overlap_length >= MIN_DIRECT_QT16_OVERLAP
        and gap_included_identity
        >= MIN_QT16_GAP_INCLUDED_IDENTITY
    )

    qt16_clean_short_overlap = (
        MIN_SHORT_QT16_OVERLAP
        <= overlap_length
        < MIN_DIRECT_QT16_OVERLAP
        and gap_included_identity
        >= MIN_SHORT_QT16_GAP_INCLUDED_IDENTITY
        and overlap_result["gaps"] <= MAX_SHORT_QT16_GAPS
    )

    qt16_only_success = (
        condition_label == "16"
        and (
            qt16_direct_overlap
            or qt16_clean_short_overlap
        )
        and overlap_result["base_identity"] >= 97
        and weighted_identity >= 90
        and qt_matches >= 25
        and qt_longest_match_run >= 8
        and qt_conflicts <= max(8, int(qt_matches * 0.20))
        and (terminal_pass or tight_terminal_overlap)
    )

    if qt16_only_success:
        return {
            "status": "F+R 결합 성공 예상",
            "rank": 2,
            "reason": (
                "QT16 단독 조건의 terminal overlap 기준 충족; "
                f"Q16 최장 연속 match {qt_longest_match_run} bp"
            ),
        }

    # HCU-A의 실제 QT10 성공 패턴입니다. Reverse 원본 방향에서
    # 매우 직접적인 terminal overlap이 확인되는 경우에 한해
    # QT10 성공 후보로 판정합니다.
    if (
        condition_label == "10"
        and reverse_orientation == "원본 Reverse 방향 사용"
        and tight_terminal_overlap
    ):
        return {
            "status": "F+R 결합 성공 예상",
            "rank": 2,
            "reason": (
                "QT10 성공 보정 패턴과 직접 terminal overlap 충족; "
                f"junction 잔여 {total_slack} bp"
            ),
        }

    # U126-A02-01 QT10 성공 사례처럼 reverse-complement 방향에서
    # 긴 overlap과 직접 junction을 가지며 Q10 연속 anchor가 긴
    # 패턴입니다. Q30에서는 anchor가 짧아져 30/20로 승격하지
    # 않습니다.
    qt10_long_anchor_profile = (
        condition_label == "10"
        and reverse_orientation == "Reverse-complement 적용"
        and overlap_length >= 250
        and overlap_result["base_identity"] >= 97
        and weighted_identity >= 90
        and total_slack <= 30
        and qt_longest_match_run >= 30
        and qt_conflicts <= max(45, int(qt_matches * 0.20))
    )

    if qt10_long_anchor_profile:
        return {
            "status": "F+R 결합 성공 예상",
            "rank": 2,
            "reason": (
                "QT10 장거리 terminal overlap과 연속 Quality "
                f"anchor 충족; Q10 최장 연속 match "
                f"{qt_longest_match_run} bp"
            ),
        }

    if condition_label == "16":
        qt16_reasons = []

        if overlap_length < MIN_SHORT_QT16_OVERLAP:
            qt16_reasons.append(
                f"terminal overlap {overlap_length} bp로 "
                f"{MIN_SHORT_QT16_OVERLAP} bp 미만"
            )

        elif overlap_length < MIN_DIRECT_QT16_OVERLAP:
            short_failures = []

            if (
                gap_included_identity
                < MIN_SHORT_QT16_GAP_INCLUDED_IDENTITY
            ):
                short_failures.append(
                    "gap 포함 Identity "
                    f"{gap_included_identity:.2f}%"
                )

            if overlap_result["gaps"] > MAX_SHORT_QT16_GAPS:
                short_failures.append(
                    f"gap {overlap_result['gaps']}개"
                )

            if short_failures:
                qt16_reasons.append(
                    "70 bp 미만 overlap의 엄격 기준 미충족("
                    + ", ".join(short_failures)
                    + ")"
                )

        if (
            overlap_length >= MIN_DIRECT_QT16_OVERLAP
            and
            gap_included_identity
            < MIN_QT16_GAP_INCLUDED_IDENTITY
        ):
            qt16_reasons.append(
                "gap 포함 Identity "
                f"{gap_included_identity:.2f}%로 "
                f"{MIN_QT16_GAP_INCLUDED_IDENTITY}% 미만"
            )

        if not qt16_reasons:
            qt16_reasons.append(
                "QT16 직접 결합용 Quality/junction 기준 미충족"
            )

        return {
            "status": "Contig2 예상",
            "rank": 1,
            "reason": (
                "QT16 단독 조건에서 overlap 후보는 있으나 "
                + "; ".join(qt16_reasons)
                + "; F/R 개별 출력 예상"
            ),
        }

    if condition_label in CONTIG2_CALIBRATED_CONDITIONS:
        return {
            "status": "Contig2 예상",
            "rank": 1,
            "reason": (
                f"조건 {condition_label}은 현재 보정 사례에서 "
                "F/R 개별 출력 우선; junction 지표는 보조 근거"
            ),
        }

    no_contig_reasons = []

    if overlap_length < 25:
        no_contig_reasons.append("overlap 25 bp 미만")

    if weighted_identity < 92:
        no_contig_reasons.append(
            "Quality 가중 Identity 92% 미만"
        )

    if qt_matches < 20:
        no_contig_reasons.append(
            f"Q{qt_threshold} 지지 match 20 bp 미만"
        )

    if (
        condition_label == "30/20"
        and qt_longest_match_run < 8
        and not low_quality_terminal_rescue
    ):
        no_contig_reasons.append(
            f"Q{qt_threshold} 최장 연속 match 8 bp 미만"
        )

    if qt_conflicts > pass_conflict_limit:
        no_contig_reasons.append(
            f"고품질 mismatch/gap {qt_conflicts}개"
        )

    if not terminal_pass:
        if new_qt_enabled:
            no_contig_reasons.append(
                f"{boundary_label} 이상 soft-clip 염기 "
                f"{terminal_high_quality}개"
            )
        else:
            no_contig_reasons.append(
                f"{boundary_label} 단독 조건의 junction 경계 미충족"
            )

    return {
        "status": "No contig 예상",
        "rank": 0,
        "reason": "; ".join(no_contig_reasons),
    }


def classify_exploratory_prediction(
    overlap_result,
    qt_threshold,
    new_qt_threshold,
):
    """
    실제 결과로 직접 보정되지 않은 확장 QT 조합을 구조적으로
    평가합니다. 단일 지표가 아니라 overlap, gap 포함 Identity,
    Quality anchor, junction을 모두 통과해야 성공 후보가 됩니다.
    """

    if overlap_result is None:
        return {
            "status": "No contig 예상",
            "rank": 0,
            "reason": "terminal overlap 후보를 찾지 못함",
        }

    overlap_length = overlap_result["paired_bases"]
    base_identity = overlap_result["base_identity"]
    gap_identity = overlap_result["gap_included_identity"]
    weighted_identity = overlap_result[
        "quality_weighted_identity"
    ]
    qt_matches = overlap_result["qt_supported_matches"]
    longest_run = overlap_result[
        "qt_supported_longest_match_run"
    ]
    conflicts = overlap_result["qt_supported_conflicts"]
    gaps = overlap_result["gaps"]
    terminal_slack = overlap_result["terminal_slack_total"]
    terminal_high_quality = overlap_result[
        "terminal_high_quality_bases"
    ]
    low_quality_fraction = overlap_result[
        "terminal_low_quality_fraction"
    ]
    internal_overlap_mode = deep_internal_overlap_mode(
        overlap_result,
        qt_threshold,
        new_qt_threshold,
    )
    internal_contig2_candidate = deep_internal_contig2_candidate(
        overlap_result,
        qt_threshold,
        new_qt_threshold,
    )

    hard_failures = []

    if overlap_length < 20:
        hard_failures.append("overlap 20 bp 미만")

    if weighted_identity < 82:
        hard_failures.append("Quality 가중 Identity 82% 미만")

    if qt_matches < 10:
        hard_failures.append(
            f"Q{qt_threshold} 지지 match 10 bp 미만"
        )

    if conflicts > max(25, int(qt_matches * 0.45)):
        hard_failures.append(
            f"고품질 mismatch/gap {conflicts}개"
        )

    if (
        terminal_slack > 350
        and internal_overlap_mode is None
        and not internal_contig2_candidate
    ):
        hard_failures.append(
            f"junction soft-clip {terminal_slack} bp 초과"
        )

    if hard_failures:
        return {
            "status": "No contig 예상",
            "rank": 0,
            "reason": "; ".join(hard_failures),
        }

    terminal_pass = (
        terminal_high_quality <= 30
        and (
            terminal_slack <= 20
            or low_quality_fraction >= 0.65
        )
    )

    direct_success = (
        overlap_length >= 70
        and base_identity >= 97
        and gap_identity >= 90
        and weighted_identity >= 92
        and qt_matches >= 25
        and longest_run >= 8
        and conflicts <= max(8, int(qt_matches * 0.15))
        and terminal_pass
    )

    clean_short_success = (
        40 <= overlap_length < 70
        and base_identity >= 98
        and gap_identity >= 94
        and weighted_identity >= 95
        and gaps <= 3
        and qt_matches >= 25
        and longest_run >= 10
        and terminal_high_quality <= 10
        and terminal_slack <= 30
    )

    low_quality_terminal_rescue = (
        new_qt_threshold is not None
        and overlap_length >= 200
        and base_identity >= 98
        and gap_identity >= 88
        and weighted_identity >= 92
        and qt_matches >= 50
        and longest_run >= 8
        and conflicts <= max(8, int(qt_matches * 0.15))
        and 60 <= terminal_slack <= 180
        and terminal_high_quality <= 30
        and low_quality_fraction >= 0.70
    )

    if (
        direct_success
        or clean_short_success
        or low_quality_terminal_rescue
        or internal_overlap_mode is not None
    ):
        if internal_overlap_mode is not None:
            success_reason = (
                f"확장 조건의 {internal_overlap_mode} 충족; "
                f"Q{qt_threshold} 최장 연속 match "
                f"{longest_run} bp"
            )
        elif low_quality_terminal_rescue:
            success_reason = (
                "확장 조건의 저품질 말단 제외 패턴 충족; "
                f"soft-clip 저품질 비율 "
                f"{low_quality_fraction * 100:.1f}%"
            )
        elif clean_short_success:
            success_reason = (
                "짧지만 gap이 적은 terminal overlap 충족; "
                f"gap 포함 Identity {gap_identity:.2f}%"
            )
        else:
            success_reason = (
                "확장 조건의 직접 terminal overlap 충족; "
                f"Q{qt_threshold} 최장 연속 match "
                f"{longest_run} bp"
            )

        return {
            "status": "F+R 결합 성공 예상",
            "rank": 2,
            "reason": success_reason,
        }

    if internal_contig2_candidate:
        return {
            "status": "Contig2 예상",
            "rank": 1,
            "reason": (
                "장거리 내부 overlap은 강하지만 Q"
                f"{qt_threshold} 최장 연속 match {longest_run} bp로 "
                "성공 anchor 기준 미충족"
            ),
        }

    contig2_candidate = (
        overlap_length >= 40
        and base_identity >= 95
        and weighted_identity >= 85
        and qt_matches >= 15
    )

    if contig2_candidate:
        return {
            "status": "Contig2 예상",
            "rank": 1,
            "reason": (
                "overlap 후보는 있으나 확장 조건의 직접 결합 기준 "
                "또는 인접 조건 안정성 확인 필요"
            ),
        }

    return {
        "status": "No contig 예상",
        "rank": 0,
        "reason": (
            "확장 조건에서 안정적인 terminal overlap 근거 부족"
        ),
    }


# --------------------------------------------------
# 재반응 방향 및 기대 효과 평가
# --------------------------------------------------
def estimate_quality_readthrough(
    qualities,
    threshold=20,
    window_size=20,
):
    """마지막으로 안정적인 Quality window가 끝나는 위치를 구합니다."""

    if not qualities:
        return 0

    if len(qualities) < window_size:
        average_quality = sum(qualities) / len(qualities)
        supported_fraction = sum(
            quality >= threshold
            for quality in qualities
        ) / len(qualities)

        if average_quality >= threshold and supported_fraction >= 0.70:
            return len(qualities)

        return 0

    last_supported_end = 0

    for start in range(0, len(qualities) - window_size + 1):
        quality_window = qualities[start:start + window_size]
        average_quality = sum(quality_window) / window_size
        supported_fraction = sum(
            quality >= threshold
            for quality in quality_window
        ) / window_size

        if average_quality >= threshold and supported_fraction >= 0.70:
            last_supported_end = start + window_size

    return last_supported_end


def build_reaction_quality_profile(
    read_data,
    source_name,
    overlap_result,
    qt_threshold,
):
    """재반응 필요도를 read별로 계산하는 경험적 품질 프로필입니다."""

    sequence = normalize_alignment_sequence(read_data["sequence"])
    qualities = normalize_quality_scores(
        sequence,
        read_data["quality_scores"],
    )
    read_length = len(sequence)

    if read_length > 0:
        average_quality = sum(qualities) / read_length
        q20_fraction = sum(
            quality >= 20
            for quality in qualities
        ) / read_length
        q30_fraction = sum(
            quality >= 30
            for quality in qualities
        ) / read_length
        ambiguous_fraction = sequence.count("N") / read_length
    else:
        average_quality = 0.0
        q20_fraction = 0.0
        q30_fraction = 0.0
        ambiguous_fraction = 1.0

    readthrough_length = estimate_quality_readthrough(
        qualities,
        threshold=20,
    )
    readthrough_fraction = (
        readthrough_length / read_length
        if read_length > 0
        else 0.0
    )

    source_key = source_name.lower()
    junction_softclip = 0
    junction_high_quality_softclip = 0
    junction_low_quality_softclip = 0
    junction_anchor_mean_quality = 0.0
    junction_anchor_qt_fraction = 0.0

    if overlap_result is not None:
        junction_softclip = overlap_result.get(
            f"{source_key}_junction_softclip",
            0,
        )
        junction_high_quality_softclip = overlap_result.get(
            f"{source_key}_junction_high_quality_softclip",
            0,
        )
        junction_low_quality_softclip = overlap_result.get(
            f"{source_key}_junction_low_quality_softclip",
            0,
        )
        junction_anchor_mean_quality = overlap_result.get(
            f"{source_key}_junction_anchor_mean_quality",
            0.0,
        )
        junction_anchor_qt_fraction = overlap_result.get(
            f"{source_key}_junction_anchor_qt_fraction",
            0.0,
        )

    need_score = 0
    need_reasons = []

    if average_quality < 20:
        need_score += 3
        need_reasons.append("평균 Quality 낮음")
    elif average_quality < 28:
        need_score += 1
        need_reasons.append("평균 Quality 경계")

    if q20_fraction < 0.45:
        need_score += 3
        need_reasons.append("Q20 염기 비율 낮음")
    elif q20_fraction < 0.65:
        need_score += 1
        need_reasons.append("Q20 염기 비율 경계")

    if q30_fraction < 0.20:
        need_score += 2
        need_reasons.append("Q30 염기 비율 낮음")
    elif q30_fraction < 0.40:
        need_score += 1
        need_reasons.append("Q30 염기 비율 경계")

    if readthrough_fraction < 0.45:
        need_score += 3
        need_reasons.append("Q20 read-through 짧음")
    elif readthrough_fraction < 0.70:
        need_score += 1
        need_reasons.append("Q20 read-through 경계")

    if ambiguous_fraction > 0.05:
        need_score += 3
        need_reasons.append("N 염기 비율 높음")
    elif ambiguous_fraction > 0.02:
        need_score += 1
        need_reasons.append("N 염기 존재")

    if overlap_result is not None:
        if junction_anchor_mean_quality < max(10, qt_threshold - 5):
            need_score += 3
            need_reasons.append("junction anchor Quality 낮음")
        elif junction_anchor_mean_quality < qt_threshold:
            need_score += 2
            need_reasons.append("junction anchor Quality 부족")
        elif junction_anchor_mean_quality < qt_threshold + 5:
            need_score += 1
            need_reasons.append("junction anchor Quality 경계")

        if junction_anchor_qt_fraction < 0.35:
            need_score += 3
            need_reasons.append("junction QT 지지율 낮음")
        elif junction_anchor_qt_fraction < 0.65:
            need_score += 2
            need_reasons.append("junction QT 지지율 부족")
        elif junction_anchor_qt_fraction < 0.80:
            need_score += 1
            need_reasons.append("junction QT 지지율 경계")

        if junction_low_quality_softclip >= 50:
            need_score += 2
            need_reasons.append("저품질 junction 말단 김")
        elif junction_low_quality_softclip >= 15:
            need_score += 1
            need_reasons.append("저품질 junction 말단 존재")

    if need_score >= 7:
        need_level = "높음"
    elif need_score >= 2:
        need_level = "중간"
    else:
        need_level = "낮음"

    return {
        "source": source_name,
        "need_score": need_score,
        "need_level": need_level,
        "need_reasons": need_reasons,
        "read_length": read_length,
        "average_quality": round(average_quality, 2),
        "q20_fraction": round(q20_fraction, 4),
        "q30_fraction": round(q30_fraction, 4),
        "readthrough_length": readthrough_length,
        "readthrough_fraction": round(readthrough_fraction, 4),
        "ambiguous_fraction": round(ambiguous_fraction, 4),
        "junction_softclip": junction_softclip,
        "junction_high_quality_softclip": (
            junction_high_quality_softclip
        ),
        "junction_low_quality_softclip": (
            junction_low_quality_softclip
        ),
        "junction_anchor_mean_quality": round(
            junction_anchor_mean_quality,
            2,
        ),
        "junction_anchor_qt_fraction": round(
            junction_anchor_qt_fraction,
            4,
        ),
    }


def evaluate_structural_support(overlap_result, qt_threshold):
    """재반응으로 회복 가능한 overlap 구조인지 3단계로 평가합니다."""

    if overlap_result is None:
        return {
            "level": "낮음",
            "score": 0,
            "reason": "terminal overlap 후보가 없습니다.",
        }

    overlap_length = overlap_result["paired_bases"]
    base_identity = overlap_result["base_identity"]
    weighted_identity = overlap_result["quality_weighted_identity"]
    qt_matches = overlap_result["qt_supported_matches"]
    qt_conflicts = overlap_result["qt_supported_conflicts"]
    terminal_high_quality = overlap_result[
        "terminal_high_quality_bases"
    ]

    support_score = 0

    if overlap_length >= 70:
        support_score += 2
    elif overlap_length >= 35:
        support_score += 1

    if base_identity >= 97:
        support_score += 2
    elif base_identity >= 90:
        support_score += 1

    if weighted_identity >= 92:
        support_score += 2
    elif weighted_identity >= 85:
        support_score += 1

    if qt_matches >= 30:
        support_score += 2
    elif qt_matches >= 15:
        support_score += 1

    if qt_conflicts <= max(5, int(qt_matches * 0.15)):
        support_score += 1

    if terminal_high_quality <= 30:
        support_score += 1

    if support_score >= 8:
        support_level = "높음"
    elif support_score >= 5:
        support_level = "중간"
    else:
        support_level = "낮음"

    return {
        "level": support_level,
        "score": support_score,
        "reason": (
            f"terminal overlap {overlap_length} bp, "
            f"Quality 가중 Identity {weighted_identity}%, "
            f"Q{qt_threshold} 지지 match {qt_matches} bp"
        ),
    }


def expectation_level(score):
    if score >= 2:
        return "높음"
    if score >= 1:
        return "중간"
    return "낮음"


def evaluate_rerun_scenarios(
    forward_data,
    reverse_data,
    overlap_result,
    current_prediction,
    simulation_rows,
    selected_condition,
    qt_threshold,
):
    """
    현재 AB1에서 품질 제한 방향과 구조적 overlap 가능성을 분리해
    F/R 재반응의 상대적 기대 효과를 제시합니다.
    """

    forward_profile = build_reaction_quality_profile(
        forward_data,
        "Forward",
        overlap_result,
        qt_threshold,
    )
    reverse_profile = build_reaction_quality_profile(
        reverse_data,
        "Reverse",
        overlap_result,
        qt_threshold,
    )
    structural_support = evaluate_structural_support(
        overlap_result,
        qt_threshold,
    )

    successful_conditions = [
        row["조건"]
        for row in simulation_rows
        if row["Contig 예측"] == "F+R 결합 성공 예상"
    ]
    current_success = (
        current_prediction["status"] == "F+R 결합 성공 예상"
    )
    alternative_successes = [
        condition
        for condition in successful_conditions
        if condition != selected_condition
    ]

    if current_success:
        recommendation = "재반응 불필요"
        recommendation_reason = (
            f"현재 조건 {selected_condition}에서 F+R 결합 성공이 "
            "예상됩니다."
        )
        forward_effect = 0
        reverse_effect = 0
        both_effect = 0
        failure_risk = 0
        confidence = "높음"

    elif alternative_successes:
        preferred_condition = alternative_successes[0]
        recommendation = f"조건 {preferred_condition} 적용 우선"
        recommendation_reason = (
            "재반응 전에 동일 AB1의 성공 예상 조건을 먼저 적용하는 "
            "편이 효율적입니다."
        )
        forward_effect = 0
        reverse_effect = 0
        both_effect = 0
        failure_risk = 0
        confidence = "중간"

    else:
        forward_need = forward_profile["need_score"]
        reverse_need = reverse_profile["need_score"]
        structural_score = structural_support["score"]

        def one_side_effect(side_need, other_need):
            if structural_score < 5:
                return 1 if side_need >= 7 and other_need < 7 else 0

            if side_need >= 7 and other_need < 7:
                return 2

            if (
                side_need >= 4
                and side_need - other_need >= 2
            ):
                return 2 if structural_score >= 8 else 1

            if (
                side_need >= 2
                and side_need - other_need >= 2
            ):
                return 1

            if side_need >= 3:
                return 1

            return 0

        forward_effect = one_side_effect(
            forward_need,
            reverse_need,
        )
        reverse_effect = one_side_effect(
            reverse_need,
            forward_need,
        )

        if structural_score >= 8 and min(
            forward_need,
            reverse_need,
        ) >= 3:
            both_effect = 2
        elif structural_score >= 5 and max(
            forward_need,
            reverse_need,
        ) >= 3:
            both_effect = 1
        elif structural_score < 5 and min(
            forward_need,
            reverse_need,
        ) >= 7:
            both_effect = 1
        else:
            both_effect = 0

        if structural_score < 5:
            failure_risk = 2
        elif (
            forward_need < 2
            and reverse_need < 2
            and not current_success
        ):
            failure_risk = 2
        elif structural_score < 8:
            failure_risk = 1
        elif max(forward_need, reverse_need) >= 7:
            failure_risk = 0
        else:
            failure_risk = 1

        if failure_risk >= 2 and max(
            forward_effect,
            reverse_effect,
            both_effect,
        ) == 0:
            recommendation = "단순 재반응 효과 낮음"
            recommendation_reason = (
                "품질보다 pairing, primer, 혼합 template 또는 "
                "구조적 overlap 문제를 먼저 확인해야 합니다."
            )
        elif (
            both_effect > max(forward_effect, reverse_effect)
            or (
                both_effect >= 1
                and abs(forward_need - reverse_need) <= 1
                and min(forward_need, reverse_need) >= 3
            )
        ):
            recommendation = "양방향 재반응 권장"
            recommendation_reason = (
                "F와 R 모두 품질 보완 필요성이 확인됩니다."
            )
        elif forward_effect > reverse_effect:
            recommendation = "Forward만 재반응 우선"
            recommendation_reason = (
                "Forward가 Reverse보다 결합 제한 요인으로 평가됩니다."
            )
        elif reverse_effect > forward_effect:
            recommendation = "Reverse만 재반응 우선"
            recommendation_reason = (
                "Reverse가 Forward보다 결합 제한 요인으로 평가됩니다."
            )
        elif forward_profile["need_score"] > reverse_profile["need_score"]:
            recommendation = "Forward만 재반응 검토"
            recommendation_reason = (
                "Forward의 상대적 품질 보완 필요성이 더 큽니다."
            )
        elif reverse_profile["need_score"] > forward_profile["need_score"]:
            recommendation = "Reverse만 재반응 검토"
            recommendation_reason = (
                "Reverse의 상대적 품질 보완 필요성이 더 큽니다."
            )
        else:
            recommendation = "양방향 재반응 검토"
            recommendation_reason = (
                "한 방향만을 우선할 근거가 충분하지 않습니다."
            )

        if structural_score >= 8 and abs(
            forward_need - reverse_need
        ) >= 5:
            confidence = "높음"
        elif structural_score >= 5:
            confidence = "중간"
        else:
            confidence = "낮음"

    forward_reason = (
        f"F 재반응 필요도 {forward_profile['need_level']} · "
        f"Q20 {forward_profile['q20_fraction'] * 100:.1f}% · "
        f"Q20 read-through {forward_profile['readthrough_length']} bp"
    )
    reverse_reason = (
        f"R 재반응 필요도 {reverse_profile['need_level']} · "
        f"Q20 {reverse_profile['q20_fraction'] * 100:.1f}% · "
        f"Q20 read-through {reverse_profile['readthrough_length']} bp"
    )

    return {
        "recommendation": recommendation,
        "recommendation_reason": recommendation_reason,
        "confidence": confidence,
        "structural_support": structural_support,
        "forward_profile": forward_profile,
        "reverse_profile": reverse_profile,
        "scenarios": [
            {
                "name": "F만 재반응",
                "level": expectation_level(forward_effect),
                "kind": "benefit",
                "reason": forward_reason,
            },
            {
                "name": "R만 재반응",
                "level": expectation_level(reverse_effect),
                "kind": "benefit",
                "reason": reverse_reason,
            },
            {
                "name": "양방향 재반응",
                "level": expectation_level(both_effect),
                "kind": "benefit",
                "reason": (
                    "양쪽 read의 품질 제한을 동시에 보완했을 때의 "
                    "상대적 기대 효과"
                ),
            },
            {
                "name": "재반응 후 결합 실패",
                "level": expectation_level(failure_risk),
                "kind": "risk",
                "reason": structural_support["reason"],
            },
        ],
    }


# --------------------------------------------------
# 조건 시뮬레이션 및 범위 탐색
# --------------------------------------------------
def empty_overlap_result():
    return {
        "paired_bases": 0,
        "base_identity": 0,
        "gap_included_identity": 0,
        "quality_weighted_identity": 0,
        "gaps": 0,
        "qt_supported_matches": 0,
        "qt_supported_longest_match_run": 0,
        "qt_supported_conflicts": 0,
        "connection_direction": "-",
        "reverse_orientation": "-",
        "left_tail_unaligned": None,
        "right_head_unaligned": None,
        "terminal_high_quality_bases": None,
        "terminal_low_quality_fraction": 0,
    }


def evaluate_condition_values(
    forward_data,
    reverse_data,
    qt_value,
    current_new_qt,
):
    calibrated_label = standard_condition_label(
        qt_value,
        current_new_qt,
    )
    condition_label = (
        calibrated_label
        if calibrated_label is not None
        else format_condition_label(qt_value, current_new_qt)
    )

    overlap_result = analyze_best_reverse_orientation(
        forward_data["sequence"],
        forward_data["quality_scores"],
        reverse_data["sequence"],
        reverse_data["quality_scores"],
        qt_value,
        current_new_qt,
    )

    if calibrated_label is not None:
        prediction = classify_contig_prediction(
            overlap_result,
            qt_value,
            current_new_qt,
            calibrated_label,
        )
        condition_type = "표준"
        condition_order = COMPANY_CONDITIONS.index(
            calibrated_label
        )
    else:
        prediction = classify_exploratory_prediction(
            overlap_result,
            qt_value,
            current_new_qt,
        )
        condition_type = "확장"
        condition_order = 1000 + qt_value * 100 + (
            -1 if current_new_qt is None else current_new_qt
        )

    safe_overlap = (
        overlap_result
        if overlap_result is not None
        else empty_overlap_result()
    )

    return {
        "조건": condition_label,
        "조건 유형": condition_type,
        "QT": qt_value,
        "New QT": (
            "미사용"
            if current_new_qt is None
            else str(current_new_qt)
        ),
        "Overlap": safe_overlap["paired_bases"],
        "Gap 제외 Identity (%)": safe_overlap["base_identity"],
        "Gap 포함 Identity (%)": safe_overlap[
            "gap_included_identity"
        ],
        "Quality 가중 Identity (%)": safe_overlap[
            "quality_weighted_identity"
        ],
        "QT 지지 Match": safe_overlap[
            "qt_supported_matches"
        ],
        "QT 최장 연속 Match": safe_overlap[
            "qt_supported_longest_match_run"
        ],
        "고품질 Mismatch/Gap": safe_overlap[
            "qt_supported_conflicts"
        ],
        "Gap": safe_overlap["gaps"],
        "연결 방향": safe_overlap["connection_direction"],
        "Reverse 처리": safe_overlap["reverse_orientation"],
        "왼쪽 junction soft-clip": safe_overlap[
            "left_tail_unaligned"
        ],
        "오른쪽 junction soft-clip": safe_overlap[
            "right_head_unaligned"
        ],
        "경계 기준 이상 soft-clip 염기": safe_overlap[
            "terminal_high_quality_bases"
        ],
        "soft-clip 저품질 비율 (%)": round(
            safe_overlap["terminal_low_quality_fraction"] * 100,
            2,
        ),
        "Contig 예측": prediction["status"],
        "인접 성공": "-",
        "추천 안정성": "-",
        "판정 근거": prediction["reason"],
        "_rank": prediction["rank"],
        "_condition_order": condition_order,
        "_qt_value": qt_value,
        "_new_qt_value": current_new_qt,
        "_standard": calibrated_label is not None,
        "_neighbor_success": 0,
        "_neighbor_total": 0,
        "_stability_rank": 0,
    }


def sort_simulation_rows(rows):
    return sorted(
        rows,
        key=lambda row: (
            -row["_rank"],
            -int(row.get("_standard", False)),
            -row.get("_stability_rank", 0),
            -row["Quality 가중 Identity (%)"],
            -row["Gap 포함 Identity (%)"],
            -row["QT 지지 Match"],
            row["경계 기준 이상 soft-clip 염기"]
            if row["경계 기준 이상 soft-clip 염기"] is not None
            else float("inf"),
            -row["Overlap"],
            row["_condition_order"],
        ),
    )


def simulate_company_conditions(
    forward_data,
    reverse_data,
    condition_labels,
):
    rows = []
    requested = set(condition_labels)

    for condition_label in COMPANY_CONDITIONS:
        if condition_label not in requested:
            continue

        qt_value, current_new_qt = parse_company_condition(
            condition_label
        )
        rows.append(
            evaluate_condition_values(
                forward_data,
                reverse_data,
                qt_value,
                current_new_qt,
            )
        )

    return sort_simulation_rows(rows)


def inclusive_range(start, end, step):
    values = list(range(start, end + 1, step))

    if values[-1] != end:
        values.append(end)

    return values


def annotate_neighbor_stability(rows):
    for row in rows:
        if row["_rank"] != 2:
            continue

        neighbors = []
        row_new_qt = row["_new_qt_value"]

        for candidate in rows:
            if candidate is row:
                continue

            if abs(candidate["_qt_value"] - row["_qt_value"]) > 1:
                continue

            candidate_new_qt = candidate["_new_qt_value"]

            if row_new_qt is None or candidate_new_qt is None:
                if row_new_qt is not None or candidate_new_qt is not None:
                    continue
            elif abs(candidate_new_qt - row_new_qt) > 2:
                continue

            neighbors.append(candidate)

        neighbor_success = sum(
            candidate["_rank"] == 2
            for candidate in neighbors
        )
        neighbor_total = len(neighbors)
        stability_ratio = (
            neighbor_success / neighbor_total
            if neighbor_total
            else 0
        )

        if neighbor_success >= 3 and stability_ratio >= 0.60:
            stability_label = "높음"
            stability_rank = 2
        elif neighbor_success >= 1 and stability_ratio >= 0.35:
            stability_label = "중간"
            stability_rank = 1
        else:
            stability_label = "낮음"
            stability_rank = 0

        row["_neighbor_success"] = neighbor_success
        row["_neighbor_total"] = neighbor_total
        row["_stability_rank"] = stability_rank
        row["인접 성공"] = (
            f"{neighbor_success}/{neighbor_total}"
            if neighbor_total
            else "0/0"
        )
        row["추천 안정성"] = stability_label

    return rows


def simulate_condition_range(
    forward_data,
    reverse_data,
    qt_min,
    qt_max,
    new_qt_min,
    new_qt_max,
    include_qt_only=True,
    selected_condition=None,
):
    """
    넓은 범위는 QT 2/New QT 5 단위로 먼저 검색하고, 상위 세
    후보 주변을 QT 1/New QT 1 단위로 다시 계산합니다.
    """

    coarse_qt_values = set(inclusive_range(qt_min, qt_max, 2))
    coarse_new_qt_values = set(
        inclusive_range(new_qt_min, new_qt_max, 5)
    )

    for standard_qt, standard_new_qt in CONDITION_THRESHOLDS.values():
        if qt_min <= standard_qt <= qt_max:
            coarse_qt_values.add(standard_qt)

        if (
            standard_new_qt is not None
            and new_qt_min <= standard_new_qt <= new_qt_max
        ):
            coarse_new_qt_values.add(standard_new_qt)

    condition_specs = set()

    for qt_value in sorted(coarse_qt_values):
        if include_qt_only:
            condition_specs.add((qt_value, None))

        for current_new_qt in sorted(coarse_new_qt_values):
            condition_specs.add((qt_value, current_new_qt))

    if selected_condition in CONDITION_THRESHOLDS:
        selected_values = CONDITION_THRESHOLDS[selected_condition]
        condition_specs.add(selected_values)

    rows_by_values = {}

    def evaluate_specs(specs):
        for qt_value, current_new_qt in sorted(
            specs,
            key=lambda spec: (
                spec[0],
                -1 if spec[1] is None else spec[1],
            ),
        ):
            key = (qt_value, current_new_qt)

            if key in rows_by_values:
                continue

            rows_by_values[key] = evaluate_condition_values(
                forward_data,
                reverse_data,
                qt_value,
                current_new_qt,
            )

    evaluate_specs(condition_specs)

    coarse_rows = sort_simulation_rows(
        list(rows_by_values.values())
    )
    seed_rows = coarse_rows[:3]
    fine_specs = set()

    for seed in seed_rows:
        seed_qt = seed["_qt_value"]
        seed_new_qt = seed["_new_qt_value"]

        for qt_value in range(
            max(qt_min, seed_qt - 1),
            min(qt_max, seed_qt + 1) + 1,
        ):
            if seed_new_qt is None:
                if include_qt_only:
                    fine_specs.add((qt_value, None))
                continue

            for current_new_qt in range(
                max(new_qt_min, seed_new_qt - 2),
                min(new_qt_max, seed_new_qt + 2) + 1,
            ):
                fine_specs.add((qt_value, current_new_qt))

    evaluate_specs(fine_specs)

    rows = list(rows_by_values.values())
    annotate_neighbor_stability(rows)
    return sort_simulation_rows(rows)


def choose_best_simulation(simulation_rows):
    if not simulation_rows:
        return None

    return sort_simulation_rows(simulation_rows)[0]


# --------------------------------------------------
# Alignment 내용을 여러 줄로 출력
# --------------------------------------------------
def format_alignment_preview(
    aligned_left,
    markers,
    aligned_right,
    left_label="Left",
    right_label="Right",
    line_width=100,
):
    output_lines = []

    total_length = len(aligned_left)

    for start in range(
        0,
        total_length,
        line_width,
    ):
        end = min(
            start + line_width,
            total_length,
        )

        output_lines.append(
            f"{left_label:<20} " + aligned_left[start:end]
        )

        output_lines.append(
            " " * 21 + markers[start:end]
        )

        output_lines.append(
            f"{right_label:<20} " + aligned_right[start:end]
        )

        output_lines.append("")

    return "\n".join(output_lines)

# --------------------------------------------------
# 메인 화면
# --------------------------------------------------
st.title("🧬 Contig QC Demo")

st.write(
    """
    Forward와 Reverse AB1 파일의 염기서열과 Quality를 읽고,
    원본 read를 유지한 채 Quality 가중 terminal overlap과
    junction soft-clipping 가능성을 평가합니다. Primer명과 무관하게
    16S, ITS, M13F/M13R처럼 서로 마주 보는 read 쌍을 사용할 수
    있습니다.
    """
)

st.divider()


# --------------------------------------------------
# 조건 설정
# --------------------------------------------------
st.subheader("조건 설정")

st.caption(
    "실제 분석에 사용하는 조건을 개별 프리셋으로 평가합니다. "
    "16과 20은 New QT를 사용하지 않는 단독 QT 조건이며, "
    "30/40과 단독 10도 유효한 독립 조건입니다. 현재 조건별 "
    "출력 유형은 현재 확인된 열두 AB1 쌍의 실제 결과를 기준으로 "
    "보수적으로 보정되어 있으며, 추가 사례에 따라 갱신해야 "
    "합니다. 20계열의 좋은 junction은 성공을 확정하지 않고 "
    "Contig2를 우선합니다. QT16은 New QT 미사용 조건이므로 "
    "짧은 overlap과 gap이 많은 정렬을 보수적으로 평가합니다. "
    "Primer 파일명은 판정에 사용하지 않고 Reverse 원본과 "
    "reverse-complement를 모두 비교해 정렬 방향을 선택합니다."
)

selected_condition = st.selectbox(
    "현재 분석 조건",
    options=COMPANY_CONDITIONS,
    index=COMPANY_CONDITIONS.index("16"),
)

qt_threshold, new_qt_threshold = parse_company_condition(
    selected_condition
)

with st.expander("조건 시뮬레이터 설정", expanded=False):
    qt_search_range = st.slider(
        "QT 탐색 범위",
        min_value=10,
        max_value=30,
        value=(10, 30),
        step=1,
    )

    new_qt_search_range = st.slider(
        "New QT 탐색 범위",
        min_value=10,
        max_value=40,
        value=(10, 40),
        step=1,
    )

    include_qt_only = st.checkbox(
        "New QT 미사용 조건도 탐색",
        value=True,
        help=(
            "QT만 사용하는 16, 20 등의 조건을 숫자 0과 구분해 "
            "별도 평가합니다."
        ),
    )

    st.caption(
        "지정 범위를 먼저 QT 2/New QT 5 단위로 탐색한 뒤 상위 "
        "후보 주변을 1단위로 정밀 분석합니다. 표준 조건은 실제 "
        "사례 보정값을 유지하고, 그 외 값은 확장 조건으로 "
        "표시합니다. 선택한 현재 조건은 범위 밖이어도 비교를 위해 "
        "자동 포함합니다."
    )

st.divider()


# --------------------------------------------------
# 파일 업로드
# --------------------------------------------------
left_column, right_column = st.columns(2)

with left_column:
    st.subheader("Forward AB1")

    forward_file = st.file_uploader(
        "Forward 방향 AB1 파일을 선택하세요.",
        type=["ab1", "abi"],
        key="forward_file",
    )

with right_column:
    st.subheader("Reverse AB1")

    reverse_file = st.file_uploader(
        "Reverse 방향 AB1 파일을 선택하세요.",
        type=["ab1", "abi"],
        key="reverse_file",
    )

st.divider()


# --------------------------------------------------
# 분석 실행
# --------------------------------------------------
if forward_file is not None and reverse_file is not None:

    if st.button(
        "AB1 파일 분석",
        type="primary",
        use_container_width=True,
    ):

        try:
            # AB1 읽기
            forward_data = read_ab1(forward_file)
            reverse_data = read_ab1(reverse_file)

            st.success(
                "F/R AB1 파일을 정상적으로 읽었습니다."
            )

            # 원본 요약
            summary_table = pd.DataFrame(
                [
                    make_summary(
                        forward_data,
                        "Forward",
                    ),
                    make_summary(
                        reverse_data,
                        "Reverse",
                    ),
                ]
            )

            st.subheader("AB1 원본 요약")

            st.dataframe(
                summary_table,
                use_container_width=True,
                hide_index=True,
            )

            st.divider()

            # Quality 그래프
            chart_left, chart_right = st.columns(2)

            with chart_left:
                show_quality_chart(
                    forward_data,
                    "Forward",
                )

            with chart_right:
                show_quality_chart(
                    reverse_data,
                    "Reverse",
                )

            st.divider()

            # 원본 Reverse read를 reverse-complement로 변환
            (
                reverse_complement_sequence,
                reverse_complement_qualities,
            ) = make_reverse_complement(
                reverse_data["sequence"],
                reverse_data["quality_scores"],
            )

            overlap_result = analyze_best_reverse_orientation(
                forward_data["sequence"],
                forward_data["quality_scores"],
                reverse_data["sequence"],
                reverse_data["quality_scores"],
                qt_threshold,
                new_qt_threshold,
            )

            current_prediction = classify_contig_prediction(
                overlap_result,
                qt_threshold,
                new_qt_threshold,
                selected_condition,
            )

            with st.expander(
                "현재 조건 F/R terminal overlap 분석",
                expanded=False,
            ):
                st.caption(
                    f"현재 조건: {selected_condition}. "
                    "원본 read를 유지하고 Reverse 원본/"
                    "reverse-complement 및 두 연결 방향을 모두 "
                    "평가했습니다."
                )

                if overlap_result is None:
                    st.error(
                        "F와 Reverse-complement R 사이에서 "
                        "terminal overlap 후보를 찾지 못했습니다."
                    )
                else:
                    overlap_table = pd.DataFrame(
                        [
                            {
                                "Alignment score": overlap_result[
                                    "score"
                                ],
                                "Overlap 염기 수": overlap_result[
                                    "paired_bases"
                                ],
                                "Gap 제외 Identity (%)": overlap_result[
                                    "base_identity"
                                ],
                                "Gap 포함 Identity (%)": overlap_result[
                                    "gap_included_identity"
                                ],
                                "Quality 가중 Identity (%)": overlap_result[
                                    "quality_weighted_identity"
                                ],
                                f"Q{qt_threshold} 지지 Match": overlap_result[
                                    "qt_supported_matches"
                                ],
                                f"Q{qt_threshold} 최장 연속 Match": overlap_result[
                                    "qt_supported_longest_match_run"
                                ],
                                "고품질 Mismatch/Gap": overlap_result[
                                    "qt_supported_conflicts"
                                ],
                                "Match": overlap_result["matches"],
                                "Mismatch": overlap_result["mismatches"],
                                "Gap": overlap_result["gaps"],
                            }
                        ]
                    )

                    st.dataframe(
                        overlap_table,
                        use_container_width=True,
                        hide_index=True,
                    )

                    prediction_message = (
                        f"{current_prediction['status']}: "
                        f"{current_prediction['reason']}"
                    )

                    if (
                        current_prediction["status"]
                        == "F+R 결합 성공 예상"
                    ):
                        st.success(prediction_message)
                    elif (
                        current_prediction["status"]
                        == "Contig2 예상"
                    ):
                        st.warning(prediction_message)
                    else:
                        st.error(prediction_message)

            if overlap_result is not None:
                junction_table = pd.DataFrame(
                    [
                        {
                            "연결 방향": overlap_result[
                                "connection_direction"
                            ],
                            "Reverse 처리": overlap_result[
                                "reverse_orientation"
                            ],
                            "왼쪽 junction soft-clip": overlap_result[
                                "left_tail_unaligned"
                            ],
                            "오른쪽 junction soft-clip": overlap_result[
                                "right_head_unaligned"
                            ],
                            "Soft-clip 합계": overlap_result[
                                "terminal_slack_total"
                            ],
                            "경계 기준 이상 soft-clip 염기": overlap_result[
                                "terminal_high_quality_bases"
                            ],
                            "Soft-clip 저품질 비율 (%)": round(
                                overlap_result[
                                    "terminal_low_quality_fraction"
                                ]
                                * 100,
                                2,
                            ),
                        }
                    ]
                )

                with st.expander(
                    "Junction 경계 평가",
                    expanded=False,
                ):
                    st.dataframe(
                        junction_table,
                        use_container_width=True,
                        hide_index=True,
                    )

                with st.expander("지표 설명", expanded=False):
                    st.markdown(
                        """
- **Junction**은 방향을 맞춘 두 read가 contig로 이어지는 연결 경계입니다.
- **왼쪽 junction soft-clip**은 왼쪽 read의 정렬 종료 뒤에 남아, 연결을 위해 제외해야 하는 말단 염기 수입니다.
- **오른쪽 junction soft-clip**은 오른쪽 read의 정렬 시작 전에 남아, 연결을 위해 제외해야 하는 말단 염기 수입니다.
- **Soft-clip 합계**는 위 두 값의 합입니다. 작을수록 두 read가 말단에서 직접 만나지만, 작다는 이유만으로 contig 성공이 확정되지는 않습니다.
- **경계 기준 이상 soft-clip 염기**는 제외 후보 중 현재 품질 경계 이상인 염기 수입니다. New QT 사용 조건에서는 New QT, 미사용 조건에서는 QT가 경계가 됩니다. 값이 크면 신뢰도 높은 서열을 많이 버려야 하므로 불리합니다.
- **Soft-clip 저품질 비율**은 제외 후보 중 현재 품질 경계 미만 염기의 비율입니다. 높을수록 말단 제외가 합리적이라는 보조 근거입니다.
- **QT 최장 연속 Match**는 양쪽 염기가 모두 현재 QT 이상이면서 정확히 일치하는 구간 중 가장 긴 연속 길이입니다. QT 지지 Match 총량이 많아도 이 값이 짧으면 고품질 anchor가 여러 조각으로 끊긴 상태입니다.
- **장거리 내부 overlap**은 양쪽 read 끝에 긴 read-through가 남아도 내부에서 250 bp 이상의 강하고 연속적인 overlap이 확인되는 유형입니다. Primer명과 무관하게 평가하며, 일반 terminal overlap보다 엄격한 Identity·gap·anchor 기준을 적용합니다.
                        """
                    )
                    st.info(
                        "Soft-clip은 원본 AB1에서 염기를 삭제한다는 "
                        "뜻이 아니라, 해당 alignment/조립 경계에서 "
                        "사용하지 않는 후보 구간입니다. 이 지표들은 "
                        "구조적 보조값이며 QT 조건별 성공을 단독으로 "
                        "결정하지 않습니다."
                    )

                    st.caption(
                        "Gap을 일률적으로 동일 감점하지 않고 해당 "
                        "염기의 Quality로 가중했습니다. 정렬 밖 "
                        "junction 염기도 현재 품질 경계 미만이면 저품질 "
                        "soft-clip 후보로 취급합니다."
                    )

                with st.expander("Alignment 상세 보기", expanded=False):
                    alignment_preview = format_alignment_preview(
                        overlap_result["aligned_left"],
                        overlap_result["markers"],
                        overlap_result["aligned_right"],
                        overlap_result["left_sequence"],
                        overlap_result["right_sequence"],
                    )

                    st.code(
                        alignment_preview,
                        language=None,
                    )

            # ----------------------------------
            # QT / New QT 범위 탐색
            # ----------------------------------
            with st.spinner(
                "QT/New QT 범위를 탐색하고 상위 후보를 "
                "정밀 분석하고 있습니다..."
            ):
                simulation_rows = simulate_condition_range(
                    forward_data,
                    reverse_data,
                    qt_search_range[0],
                    qt_search_range[1],
                    new_qt_search_range[0],
                    new_qt_search_range[1],
                    include_qt_only=include_qt_only,
                    selected_condition=selected_condition,
                )

            best_simulation = choose_best_simulation(
                simulation_rows
            )

            st.divider()
            st.subheader("Contig 시뮬레이션 결과")
            st.caption(
                "지정 범위에서 성공 가능 조건을 찾고, 인접 QT/New "
                "QT에서도 결과가 유지되는지 평가합니다. 표준 조건은 "
                "확인된 열두 AB1 쌍의 실제 결과로 보정했으며, 확장 "
                "조건은 구조 기반 탐색 후보입니다."
            )

            simulation_display_rows = [
                {
                    key: value
                    for key, value in row.items()
                    if not key.startswith("_")
                }
                for row in simulation_rows
            ]

            simulation_table = pd.DataFrame(
                simulation_display_rows
            )

            if best_simulation is not None:
                recommendation = (
                    f"추천 조건: {best_simulation['조건']} · "
                    f"{best_simulation['조건 유형']} 조건 · "
                    f"{best_simulation['Contig 예측']} · "
                    f"안정성 {best_simulation['추천 안정성']} "
                    f"(인접 성공 {best_simulation['인접 성공']}) · "
                    f"{best_simulation['판정 근거']}"
                )

                if (
                    best_simulation["Contig 예측"]
                    == "F+R 결합 성공 예상"
                ):
                    if best_simulation["조건 유형"] == "확장":
                        st.info(
                            recommendation
                            + " · 확장 조건은 아직 실제 결과 보정이 "
                            "없으므로 시험 적용 후보입니다."
                        )
                    elif best_simulation["추천 안정성"] == "낮음":
                        st.warning(
                            recommendation
                            + " · 단일 조건 성공일 수 있어 실제 적용 전 "
                            "검토가 필요합니다."
                        )
                    else:
                        st.success(recommendation)
                elif best_simulation["Contig 예측"] == "Contig2 예상":
                    st.warning(recommendation)
                else:
                    st.error(
                        "시험한 조건에서 생성 가능한 조합을 "
                        "찾지 못했습니다. "
                        + recommendation
                    )

            success_count = sum(
                row["Contig 예측"] == "F+R 결합 성공 예상"
                for row in simulation_rows
            )
            contig2_count = sum(
                row["Contig 예측"] == "Contig2 예상"
                for row in simulation_rows
            )
            no_contig_count = sum(
                row["Contig 예측"] == "No contig 예상"
                for row in simulation_rows
            )

            summary_success, summary_contig2, summary_fail = st.columns(3)
            summary_success.metric(
                "결합 성공 예상",
                f"{success_count}개 조건",
            )
            summary_contig2.metric(
                "Contig2 예상",
                f"{contig2_count}개 조건",
            )
            summary_fail.metric(
                "No contig 예상",
                f"{no_contig_count}개 조건",
            )

            st.markdown("#### 추천 후보 상위 조건")

            top_card_rows = simulation_rows[:12]

            for batch_start in range(0, len(top_card_rows), 4):
                card_rows = top_card_rows[
                    batch_start:batch_start + 4
                ]
                card_columns = st.columns(len(card_rows))

                for card_column, row in zip(
                    card_columns,
                    card_rows,
                ):
                    with card_column:
                        with st.container(border=True):
                            st.markdown(
                                f"### QT {row['조건']}"
                            )
                            st.caption(
                                f"{row['조건 유형']} 조건 · "
                                f"안정성 {row['추천 안정성']} · "
                                f"인접 성공 {row['인접 성공']}"
                            )

                            if (
                                row["Contig 예측"]
                                == "F+R 결합 성공 예상"
                            ):
                                st.success("✅ F+R 결합 성공 예상")
                            elif row["Contig 예측"] == "Contig2 예상":
                                st.warning("⚠️ Contig2 예상")
                            else:
                                st.error("❌ No contig 예상")

                            metric_left, metric_right = st.columns(2)
                            metric_left.metric(
                                "Overlap",
                                f"{row['Overlap']} bp",
                            )
                            metric_right.metric(
                                "가중 Identity",
                                (
                                    f"{row['Quality 가중 Identity (%)']}%"
                                ),
                            )

                            st.caption(
                                "QT 연속 Match "
                                f"{row['QT 최장 연속 Match']} bp"
                            )
                            st.caption(row["판정 근거"])

            with st.expander("상세 수치 표 보기"):
                st.dataframe(
                    simulation_table,
                    use_container_width=True,
                    hide_index=True,
                )

            simulation_csv = simulation_table.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                "시뮬레이션 결과 CSV 다운로드",
                data=simulation_csv,
                file_name="contig_simulation_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # ----------------------------------
            # 재반응 방향 및 기대 효과
            # ----------------------------------
            rerun_assessment = evaluate_rerun_scenarios(
                forward_data,
                reverse_data,
                overlap_result,
                current_prediction,
                simulation_rows,
                selected_condition,
                qt_threshold,
            )

            st.divider()
            st.subheader("재반응 시나리오")
            st.caption(
                "선택한 현재 조건과 전체 Contig 시뮬레이션 결과를 "
                "함께 평가합니다. 각 단계는 재반응의 상대적 기대 "
                "효과이며, 실제 성공 확률을 의미하지 않습니다."
            )

            recommendation_text = (
                f"추천: {rerun_assessment['recommendation']} · "
                f"판단 신뢰도 {rerun_assessment['confidence']} · "
                f"{rerun_assessment['recommendation_reason']}"
            )

            if rerun_assessment["recommendation"] == "재반응 불필요":
                st.success(recommendation_text)
            elif "조건" in rerun_assessment["recommendation"]:
                st.info(recommendation_text)
            elif "효과 낮음" in rerun_assessment["recommendation"]:
                st.error(recommendation_text)
            else:
                st.warning(recommendation_text)

            scenario_columns = st.columns(4)

            for scenario_column, scenario in zip(
                scenario_columns,
                rerun_assessment["scenarios"],
            ):
                with scenario_column:
                    with st.container(border=True):
                        st.markdown(f"#### {scenario['name']}")

                        if scenario["kind"] == "risk":
                            if scenario["level"] == "높음":
                                st.error("🔴 높음")
                            elif scenario["level"] == "중간":
                                st.warning("🟠 중간")
                            else:
                                st.success("🟢 낮음")
                        else:
                            if scenario["level"] == "높음":
                                st.success("🟢 높음")
                            elif scenario["level"] == "중간":
                                st.warning("🟠 중간")
                            else:
                                st.info("⚪ 낮음")

                        st.caption(scenario["reason"])

            with st.expander(
                "재반응 판단 근거 상세 보기",
                expanded=False,
            ):
                forward_profile = rerun_assessment[
                    "forward_profile"
                ]
                reverse_profile = rerun_assessment[
                    "reverse_profile"
                ]
                structural_support = rerun_assessment[
                    "structural_support"
                ]

                evidence_table = pd.DataFrame(
                    [
                        {
                            "방향": "Forward",
                            "재반응 필요도": forward_profile[
                                "need_level"
                            ],
                            "평균 Quality": forward_profile[
                                "average_quality"
                            ],
                            "Q20 비율 (%)": round(
                                forward_profile["q20_fraction"] * 100,
                                2,
                            ),
                            "Q30 비율 (%)": round(
                                forward_profile["q30_fraction"] * 100,
                                2,
                            ),
                            "Q20 read-through": forward_profile[
                                "readthrough_length"
                            ],
                            "Junction anchor 평균 Q": forward_profile[
                                "junction_anchor_mean_quality"
                            ],
                            "저품질 junction soft-clip": forward_profile[
                                "junction_low_quality_softclip"
                            ],
                        },
                        {
                            "방향": "Reverse",
                            "재반응 필요도": reverse_profile[
                                "need_level"
                            ],
                            "평균 Quality": reverse_profile[
                                "average_quality"
                            ],
                            "Q20 비율 (%)": round(
                                reverse_profile["q20_fraction"] * 100,
                                2,
                            ),
                            "Q30 비율 (%)": round(
                                reverse_profile["q30_fraction"] * 100,
                                2,
                            ),
                            "Q20 read-through": reverse_profile[
                                "readthrough_length"
                            ],
                            "Junction anchor 평균 Q": reverse_profile[
                                "junction_anchor_mean_quality"
                            ],
                            "저품질 junction soft-clip": reverse_profile[
                                "junction_low_quality_softclip"
                            ],
                        },
                    ]
                )

                st.dataframe(
                    evidence_table,
                    use_container_width=True,
                    hide_index=True,
                )
                st.info(
                    "Overlap 구조 지지: "
                    f"{structural_support['level']} · "
                    f"{structural_support['reason']}"
                )
                st.caption(
                    "F/R 방향은 파일명이나 primer 표기가 아니라 "
                    "Forward/Reverse 업로드 슬롯을 기준으로 표시합니다. "
                    "한쪽 재반응 후 실제 결과 사례가 누적되면 이 "
                    "규칙을 추가 보정할 수 있습니다."
                )

            st.divider()
            # 서열 확인
            with st.expander("Forward 원본 서열"):
                st.text_area(
                    "Forward original sequence",
                    value=forward_data["sequence"],
                    height=180,
                    key="forward_original_sequence",
                )

            with st.expander("Reverse 원본 방향 서열"):
                st.text_area(
                    "Reverse original sequence",
                    value=reverse_data["sequence"],
                    height=180,
                    key="reverse_original_sequence",
                )

            with st.expander("Reverse-complement 서열"):
                st.text_area(
                    "Reverse-complement sequence",
                    value=reverse_complement_sequence,
                    height=180,
                    key="reverse_complement_sequence",
                )

        except Exception as error:
            st.error(
                "AB1 파일을 분석하는 과정에서 오류가 발생했습니다."
            )
            st.exception(error)

else:
    st.info(
        "Forward와 Reverse AB1 파일을 모두 업로드해주세요."
    )
