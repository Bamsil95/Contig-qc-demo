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

    if new_qt_threshold > 0:
        terminal_high_quality_bases = sum(
            quality >= new_qt_threshold
            for quality in terminal_qualities
        )
        terminal_quality_burden = sum(
            max(quality - new_qt_threshold + 1, 0)
            / max(41 - new_qt_threshold, 1)
            for quality in terminal_qualities
        )
    else:
        # New QT 0에서는 soft clipping을 허용하지 않는 보수적 조건
        terminal_high_quality_bases = terminal_slack_total
        terminal_quality_burden = float(terminal_slack_total)

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
    "10",
    "20/0",
    "20/10",
    "20/20",
    "30/10",
    "30/20",
    "30/40",
]

CONDITION_THRESHOLDS = {
    "10": (10, 0),
    "20/0": (20, 0),
    "20/10": (20, 10),
    "20/20": (20, 20),
    "30/10": (30, 10),
    "30/20": (30, 20),
    "30/40": (30, 40),
}

# 현재까지 확인된 네 AB1 쌍의 실제 결과로 보정한 조건별
# 출력 유형입니다.
# 20계열은 terminal overlap이 좋아 보여도 성공으로 자동 승격하지
# 않고 F/R 개별 출력(Contig2)을 우선합니다.
CONTIG2_CALIBRATED_CONDITIONS = {
    "10",
    "20/0",
    "20/10",
    "20/20",
}

NO_CONTIG_CALIBRATED_CONDITIONS = {
    "30/10",
    "30/40",
}


def parse_company_condition(condition_label):
    if condition_label not in CONDITION_THRESHOLDS:
        raise ValueError(
            f"지원하지 않는 회사 조건입니다: {condition_label}"
        )

    return CONDITION_THRESHOLDS[condition_label]


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

    if total_slack > 350:
        structural_fail_reasons.append(
            f"junction soft-clip 후보 {total_slack} bp 초과"
        )

    if (
        new_qt_threshold > 0
        and terminal_high_quality
        > max(100, int(overlap_length * 0.75))
    ):
        structural_fail_reasons.append(
            f"New QT 이상 말단 염기 {terminal_high_quality}개"
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

    if new_qt_threshold > 0:
        terminal_pass = (
            terminal_high_quality <= 30
            and (
                total_slack <= 20
                or low_quality_fraction >= 0.65
            )
        )
    else:
        terminal_pass = total_slack <= 5

    pass_conditions = (
        overlap_length >= 25
        and weighted_identity >= 92
        and qt_matches >= 20
        and qt_conflicts <= pass_conflict_limit
        and terminal_pass
    )

    reverse_orientation = overlap_result.get(
        "reverse_orientation",
        "",
    )

    # 현재 확인된 NS1/NS24 및 785F/907R 성공 패턴입니다.
    # Reverse read를 reverse-complement한 연결이면서 30/20의
    # Quality 근거를 충족할 때만 성공 후보로 승격합니다.
    if (
        condition_label == "30/20"
        and reverse_orientation == "Reverse-complement 적용"
        and qt_longest_match_run >= 8
        and (pass_conditions or tight_terminal_overlap)
    ):
        return {
            "status": "F+R 결합 성공 예상",
            "rank": 2,
            "reason": (
                "30/20 성공 보정 조건과 Quality 가중 overlap 충족; "
                f"Q{qt_threshold} 최장 연속 match "
                f"{qt_longest_match_run} bp"
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
    ):
        no_contig_reasons.append(
            f"Q{qt_threshold} 최장 연속 match 8 bp 미만"
        )

    if qt_conflicts > pass_conflict_limit:
        no_contig_reasons.append(
            f"고품질 mismatch/gap {qt_conflicts}개"
        )

    if not terminal_pass:
        if new_qt_threshold > 0:
            no_contig_reasons.append(
                f"New QT 이상 soft-clip 염기 "
                f"{terminal_high_quality}개"
            )
        else:
            no_contig_reasons.append(
                f"New QT 0 junction 잔여 {total_slack} bp"
            )

    return {
        "status": "No contig 예상",
        "rank": 0,
        "reason": "; ".join(no_contig_reasons),
    }


# --------------------------------------------------
# 회사 조건 프리셋 시뮬레이션
# --------------------------------------------------
def simulate_company_conditions(
    forward_data,
    reverse_data,
    condition_labels,
):
    rows = []

    unique_conditions = [
        condition
        for condition in COMPANY_CONDITIONS
        if condition in set(condition_labels)
    ]

    for condition_label in unique_conditions:
        qt_value, current_new_qt = parse_company_condition(
            condition_label
        )

        overlap_result = analyze_best_reverse_orientation(
            forward_data["sequence"],
            forward_data["quality_scores"],
            reverse_data["sequence"],
            reverse_data["quality_scores"],
            qt_value,
            current_new_qt,
        )

        prediction = classify_contig_prediction(
            overlap_result,
            qt_value,
            current_new_qt,
            condition_label,
        )

        if overlap_result is None:
            overlap_result = {
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

        rows.append(
            {
                "회사 조건": condition_label,
                "QT": qt_value,
                "New QT": (
                    "-"
                    if condition_label == "10"
                    else str(current_new_qt)
                ),
                "Overlap": overlap_result["paired_bases"],
                "Gap 제외 Identity (%)": overlap_result[
                    "base_identity"
                ],
                "Gap 포함 Identity (%)": overlap_result[
                    "gap_included_identity"
                ],
                "Quality 가중 Identity (%)": overlap_result[
                    "quality_weighted_identity"
                ],
                "QT 지지 Match": overlap_result[
                    "qt_supported_matches"
                ],
                "QT 최장 연속 Match": overlap_result[
                    "qt_supported_longest_match_run"
                ],
                "고품질 Mismatch/Gap": overlap_result[
                    "qt_supported_conflicts"
                ],
                "Gap": overlap_result["gaps"],
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
                "New QT 이상 soft-clip 염기": overlap_result[
                    "terminal_high_quality_bases"
                ],
                "soft-clip 저품질 비율 (%)": round(
                    overlap_result[
                        "terminal_low_quality_fraction"
                    ]
                    * 100,
                    2,
                ),
                "Contig 예측": prediction["status"],
                "판정 근거": prediction["reason"],
                "_rank": prediction["rank"],
                "_condition_order": COMPANY_CONDITIONS.index(
                    condition_label
                ),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            -row["_rank"],
            -row["Quality 가중 Identity (%)"],
            -row["QT 지지 Match"],
            row["New QT 이상 soft-clip 염기"]
            if row["New QT 이상 soft-clip 염기"] is not None
            else float("inf"),
            -row["Overlap"],
            row["_condition_order"],
        ),
    )


def choose_best_simulation(simulation_rows):
    if not simulation_rows:
        return None

    return max(
        simulation_rows,
        key=lambda row: (
            row["_rank"],
            row["Quality 가중 Identity (%)"],
            row["QT 지지 Match"],
            -(
                row["New QT 이상 soft-clip 염기"]
                if row["New QT 이상 soft-clip 염기"]
                is not None
                else 10**9
            ),
            row["Overlap"],
            -row["_condition_order"],
        ),
    )


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
    junction soft-clipping 가능성을 평가합니다.
    """
)

st.divider()


# --------------------------------------------------
# 회사 프로그램 조건 설정
# --------------------------------------------------
st.subheader("회사 프로그램 조건 설정")

st.caption(
    "회사에서 실제 사용하는 조건을 개별 프리셋으로 평가합니다. "
    "30/40과 단독 10도 유효한 독립 조건입니다. 현재 조건별 "
    "출력 유형은 현재 확인된 네 AB1 쌍의 실제 결과를 기준으로 "
    "보수적으로 보정되어 있으며, 추가 사례에 따라 갱신해야 "
    "합니다. 20계열의 좋은 junction은 성공을 확정하지 않고 "
    "Contig2를 우선합니다. "
    "Primer 파일명은 판정에 사용하지 않고 Reverse 원본과 "
    "reverse-complement를 모두 비교해 정렬 방향을 선택합니다."
)

selected_condition = st.selectbox(
    "현재 분석 조건",
    options=COMPANY_CONDITIONS,
    index=COMPANY_CONDITIONS.index("30/20"),
)

qt_threshold, new_qt_threshold = parse_company_condition(
    selected_condition
)

with st.expander("조건 시뮬레이터 설정", expanded=False):
    simulation_conditions = st.multiselect(
        "자동 시험할 회사 조건",
        options=COMPANY_CONDITIONS,
        default=COMPANY_CONDITIONS,
    )

    st.caption(
        "선택한 조건을 모두 분석하고, 현재 조건은 목록에 없어도 "
        "자동으로 포함합니다. 결과는 F+R 결합 성공 예상 → "
        "Contig2 예상 → No contig 예상 순으로 정렬됩니다."
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

            st.subheader("현재 조건 F/R terminal overlap 분석")
            st.caption(
                f"현재 회사 조건: {selected_condition}. "
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
                            "New QT 이상 soft-clip 염기": overlap_result[
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

                st.subheader("Junction 경계 평가")
                st.dataframe(
                    junction_table,
                    use_container_width=True,
                    hide_index=True,
                )

                with st.expander("Junction / soft-clip 지표 설명"):
                    st.markdown(
                        """
- **Junction**은 방향을 맞춘 두 read가 contig로 이어지는 연결 경계입니다.
- **왼쪽 junction soft-clip**은 왼쪽 read의 정렬 종료 뒤에 남아, 연결을 위해 제외해야 하는 말단 염기 수입니다.
- **오른쪽 junction soft-clip**은 오른쪽 read의 정렬 시작 전에 남아, 연결을 위해 제외해야 하는 말단 염기 수입니다.
- **Soft-clip 합계**는 위 두 값의 합입니다. 작을수록 두 read가 말단에서 직접 만나지만, 작다는 이유만으로 contig 성공이 확정되지는 않습니다.
- **New QT 이상 soft-clip 염기**는 제외 후보 중 품질이 New QT 이상인 염기 수입니다. 값이 크면 신뢰도 높은 서열을 많이 버려야 하므로 불리합니다.
- **Soft-clip 저품질 비율**은 제외 후보 중 New QT 미만 염기의 비율입니다. 높을수록 말단 제외가 합리적이라는 보조 근거입니다.
- **QT 최장 연속 Match**는 양쪽 염기가 모두 현재 QT 이상이면서 정확히 일치하는 구간 중 가장 긴 연속 길이입니다. QT 지지 Match 총량이 많아도 이 값이 짧으면 고품질 anchor가 여러 조각으로 끊긴 상태입니다.
                        """
                    )
                    st.info(
                        "Soft-clip은 원본 AB1에서 염기를 삭제한다는 "
                        "뜻이 아니라, 해당 alignment/조립 경계에서 "
                        "사용하지 않는 후보 구간입니다. 이 지표들은 "
                        "구조적 보조값이며 QT 조건별 성공을 단독으로 "
                        "결정하지 않습니다."
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
                elif current_prediction["status"] == "Contig2 예상":
                    st.warning(prediction_message)
                else:
                    st.error(prediction_message)

                st.caption(
                    "Gap을 일률적으로 동일 감점하지 않고 해당 염기의 "
                    "Quality로 가중했습니다. 정렬 밖 junction 염기도 "
                    "New QT 미만이면 저품질 soft-clip 후보로 취급합니다."
                )

                with st.expander("Alignment 상세 보기"):
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
            # 회사 조건 프리셋 전체 시뮬레이션
            # ----------------------------------
            conditions_to_simulate = list(
                dict.fromkeys(
                    simulation_conditions
                    + [selected_condition]
                )
            )

            simulation_rows = simulate_company_conditions(
                forward_data,
                reverse_data,
                conditions_to_simulate,
            )

            best_simulation = choose_best_simulation(
                simulation_rows
            )

            st.divider()
            st.subheader("회사 조건별 Contig 시뮬레이터")
            st.caption(
                "각 행은 동일한 AB1 쌍을 회사 프리셋별로 평가한 "
                "결과입니다. F+R 결합 성공 예상, Contig2 예상, "
                "No contig 예상 순으로 표시합니다. 현재 조건별 "
                "prior는 한 개의 실제 보정 샘플에 기반합니다."
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

            st.dataframe(
                simulation_table,
                use_container_width=True,
                hide_index=True,
            )

            if best_simulation is not None:
                recommendation = (
                    f"추천 조건: {best_simulation['회사 조건']} · "
                    f"{best_simulation['Contig 예측']} · "
                    f"{best_simulation['판정 근거']}"
                )

                if (
                    best_simulation["Contig 예측"]
                    == "F+R 결합 성공 예상"
                ):
                    st.success(recommendation)
                elif best_simulation["Contig 예측"] == "Contig2 예상":
                    st.warning(recommendation)
                else:
                    st.error(
                        "시험한 조건에서 생성 가능한 조합을 "
                        "찾지 못했습니다. "
                        + recommendation
                    )

            simulation_csv = simulation_table.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                "시뮬레이션 결과 CSV 다운로드",
                data=simulation_csv,
                file_name="contig_company_condition_simulation.csv",
                mime="text/csv",
                use_container_width=True,
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
