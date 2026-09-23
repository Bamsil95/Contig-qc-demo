"""Contig 조건 비교: QT / New QT on-off / Window size를 분리한 버전.

실행: python -m streamlit run "app_Streamlit(5).py"
설치(Python 3.10 이상): python -m pip install streamlit pandas biopython

2026-09-23 사용자 확인: 20/10의 10은 Window size이며 Q10이 아니다.
회사 엔진: phrap 0.990319. 회사 trimming 구현과 실행 옵션은 미제공.
따라서 이전의 조건별 성공/Contig2 예외 규칙은 사용하지 않는다.
AB1 참고 분석의 trimming은 아래에 명시한 가정이며, 실제 회사 규칙이 아니다.
실제 phrap 실행 결과는 ACE의 F/R read 소속을 확인해서만 표시한다.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import pandas as pd
import streamlit as st
from Bio import SeqIO
from Bio.Align import PairwiseAligner
from Bio.Seq import Seq
from Bio.Sequencing import Ace

APP_VERSION = "2026.09.23-window-v1"
TARGET_PHRAP_VERSION = "0.990319"
MAX_READ_BASES = 5000
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
PREVIEW_METHODS = {
    "mean_q": "평균 Quality 기준 · 시범 가정",
    "mean_error": "평균 오류확률 기준 · 시범 가정",
}


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
                raise ValueError("New QT 사용 시 양의 정수 Window size가 필요합니다.")
        elif self.window_size is not None:
            raise ValueError("New QT 미사용 시 Window size는 적용하지 않습니다.")

    @property
    def key(self):
        return self.label.replace("/", "_")


# 화면 표기 16/0·10/0의 0은 New QT 해제의 약식 표기이다.
# 2nd QT는 제시된 화면처럼 네 조건 모두 해제한다. 알고리즘은 미확인.
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
            raise ValueError(f"이 도구는 read당 {MAX_READ_BASES:,} bp 이하의 F/R 쌍을 지원합니다.")
        if len(self.qualities) != len(self.sequence):
            raise ValueError("서열과 Quality 길이가 다릅니다. 누락 Quality를 임의로 채우지 않습니다.")
        if any(type(q) is not int or not 0 <= q <= 99 for q in self.qualities):
            raise ValueError("Quality에는 0~99의 정수만 사용할 수 있습니다.")
        if set(self.sequence.upper()) - set("ACGTRYSWKMBDHVNX"):
            raise ValueError("DNA 서열에 지원하지 않는 문자가 있습니다.")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_ab1(data: bytes, filename: str, read_id: str) -> Read:
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("AB1 파일이 20 MiB를 초과합니다.")
    record = SeqIO.read(io.BytesIO(data), "abi")
    qualities = record.letter_annotations.get("phred_quality")
    if qualities is None:
        raise ValueError(f"{filename}: 염기별 Quality가 없습니다.")
    return Read(read_id, str(record.seq).upper(), tuple(qualities), filename, sha256(data))


def preview_trim(read: Read, condition: Condition, method="mean_q"):
    """회사 구현 미확인 상태에서 비교용으로만 사용하는 명시적 가정.

    New QT off: 개별 Q가 QT 이상인 첫/마지막 염기 사이를 유지한다.
    New QT on: Window size개 염기를 1 bp씩 이동해 평가하고 통과한
    첫 window 시작~마지막 window 끝을 유지한다. 내부 저품질 염기는
    삭제하지 않는다. 실제 회사의 시작/종료점·재탐색 규칙과 다를 수 있다.
    """
    if condition.second_qt_enabled:
        raise ValueError("2nd QT 알고리즘이 확인되지 않아 시범 계산을 지원하지 않습니다.")
    if method not in PREVIEW_METHODS:
        raise ValueError("지원하지 않는 시범 trimming 방식입니다.")
    width = condition.window_size if condition.new_qt_enabled else 1
    n = len(read.sequence)
    start = end = 0
    if n >= width:
        values = (
            list(read.qualities) if method == "mean_q"
            else [10 ** (-q / 10.0) for q in read.qualities]
        )
        running = math.fsum(values[:width])
        passing = []
        cutoff = condition.qt if method == "mean_q" else 10 ** (-condition.qt / 10.0)
        for index in range(n - width + 1):
            average = running / width
            passes = (
                average >= cutoff - 1e-12 if method == "mean_q"
                else average <= cutoff + 1e-15
            )
            if passes:
                passing.append(index)
            if index + width < n:
                running += values[index + width] - values[index]
        if passing:
            start, end = passing[0], passing[-1] + width
    trimmed = Read(read.name, read.sequence[start:end], read.qualities[start:end],
                   read.file_name, read.source_sha256)
    return trimmed, {
        "source": "preview_assumption",
        "start_1based": start + 1 if end > start else None,
        "end_1based": end if end > start else None,
        "original_bases": n,
        "retained_bases": end - start,
        "removed_head": start if end > start else n,
        "removed_tail": n - end if end > start else 0,
        "effective_preview_window": width,
        "method": method,
        "note": "회사 전처리와 일치 여부 미확인",
    }


def encode_fasta_qual(reads):
    fasta, qual = [], []
    for read in reads:
        fasta.append(f">{read.name}\n{read.sequence}\n")
        qual.append(f">{read.name}\n{' '.join(map(str, read.qualities))}\n")
    return "".join(fasta).encode(), "".join(qual).encode()


def decode_fasta_qual(fasta: bytes, qual: bytes):
    """실제 실행 입력은 원본 bytes로 보존하고, 검증/참고 정렬만 파싱한다."""
    seq_records = list(SeqIO.parse(io.StringIO(fasta.decode("utf-8-sig")), "fasta"))
    qual_records = list(SeqIO.parse(io.StringIO(qual.decode("utf-8-sig")), "qual"))
    if len(seq_records) != 2 or len(qual_records) != 2:
        raise ValueError("각 조건의 FASTA와 QUAL에는 정확히 두 read가 있어야 합니다.")
    if len({r.id for r in seq_records}) != 2:
        raise ValueError("FASTA의 두 read ID가 중복됩니다.")
    reads = []
    for seq, quality in zip(seq_records, qual_records):
        if seq.id != quality.id:
            raise ValueError("FASTA와 QUAL의 read ID 및 순서가 일치해야 합니다.")
        reads.append(Read(seq.id, str(seq.seq), tuple(quality.letter_annotations["phred_quality"])))
    return tuple(reads)


def input_packet(condition, reads, source, trim_metadata=None, raw_bytes=None):
    fasta, qual = raw_bytes if raw_bytes is not None else encode_fasta_qual(reads)
    return {
        "condition": asdict(condition),
        "reads": reads,
        "input_source": source,
        "trim_metadata": trim_metadata or [],
        "fasta": fasta,
        "qual": qual,
        "fasta_sha256": sha256(fasta),
        "qual_sha256": sha256(qual),
    }


def prepare_preview(reads, method="mean_q"):
    packets = {}
    for condition in CONDITIONS:
        processed = [preview_trim(read, condition, method) for read in reads]
        packets[condition.label] = input_packet(
            condition, tuple(x[0] for x in processed), "preview_assumption",
            [x[1] for x in processed],
        )
    return packets


def load_input_zip(data: bytes):
    """ZIP은 디스크에 풀지 않는다. 알려진 네 경로의 입력만 읽는다."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("입력 ZIP이 20 MiB를 초과합니다.")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        if len(infos) > 100 or sum(i.file_size for i in infos) > MAX_UPLOAD_BYTES:
            raise ValueError("ZIP의 파일 수 또는 압축 해제 크기가 허용 범위를 초과합니다.")
        if any(info.flag_bits & 1 for info in infos):
            raise ValueError("암호화된 ZIP은 지원하지 않습니다.")
        names = [i.filename for i in infos]
        if len(names) != len(set(names)):
            raise ValueError("ZIP 내부에 중복 파일명이 있습니다.")
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError("ZIP 내부 경로가 올바르지 않습니다.")
        manifest = {}
        if "manifest.json" in names:
            manifest = json.loads(archive.read("manifest.json"))
            if not isinstance(manifest, dict):
                raise ValueError("manifest.json 형식이 올바르지 않습니다.")
        manifest_conditions = manifest.get("conditions", {})
        if not isinstance(manifest_conditions, dict):
            raise ValueError("manifest의 conditions는 조건별 객체여야 합니다.")
        packets = {}
        for condition in CONDITIONS:
            fasta_name = f"{condition.key}/reads.fasta"
            qual_name = fasta_name + ".qual"
            if fasta_name not in names and qual_name not in names:
                continue
            if fasta_name not in names or qual_name not in names:
                raise ValueError(f"{condition.label}: FASTA와 QUAL을 함께 넣어 주세요.")
            fasta, qual = archive.read(fasta_name), archive.read(qual_name)
            reads = decode_fasta_qual(fasta, qual)
            entry = manifest_conditions.get(condition.label, {})
            if not isinstance(entry, dict):
                raise ValueError(f"{condition.label}: manifest 항목 형식이 올바르지 않습니다.")
            trim_metadata = entry.get("trim_metadata", [])
            if not isinstance(trim_metadata, list) or any(not isinstance(item, dict) for item in trim_metadata):
                raise ValueError(f"{condition.label}: 전처리 좌표 정보가 올바르지 않습니다.")
            if entry.get("condition") and entry["condition"] != asdict(condition):
                raise ValueError(f"{condition.label}: manifest의 QT/Window 설정이 현재 조건과 다릅니다.")
            for key, payload in (("fasta_sha256", fasta), ("qual_sha256", qual)):
                if entry.get(key) and entry[key] != sha256(payload):
                    raise ValueError(f"{condition.label}: 입력 파일이 manifest 기록과 다릅니다.")
            # 이 앱이 만든 시범 입력을 재업로드해도 실제 사내 전처리로 승격하지 않는다.
            source = (
                "preview_assumption" if entry.get("input_source") == "preview_assumption"
                else "provided_unverified"
            )
            packets[condition.label] = input_packet(condition, reads, source,
                trim_metadata, (fasta, qual))
        if not packets:
            raise ValueError("16, 20_10, 30_20, 10 폴더의 reads.fasta/reads.fasta.qual을 찾지 못했습니다.")
        return packets


def normalize_sequence(sequence):
    return "".join(base if base in "ACGT" else "N" for base in sequence.upper())


def inspect_overlap(forward: Read, reverse: Read, qt: int):
    """전처리 후 두 read의 참고 정렬. phrap의 조립 성공 판정이 아니다."""
    if not forward.sequence or not reverse.sequence:
        return None
    aligner = PairwiseAligner(mode="local", match_score=2, mismatch_score=-3,
                              open_gap_score=-5, extend_gap_score=-1)
    aligner.wildcard = "N"
    fseq = normalize_sequence(forward.sequence)
    candidates = []
    for complemented in (True, False):
        rseq = normalize_sequence(
            str(Seq(reverse.sequence).reverse_complement()) if complemented else reverse.sequence
        )
        rq = reverse.qualities[::-1] if complemented else reverse.qualities
        alignments = aligner.align(fseq, rseq)
        try:
            alignment = alignments[0]
        except IndexError:
            continue
        matches = mismatches = gaps = ambiguous = qt_matches = qt_conflicts = 0
        longest_gap = gap_run = 0
        markers = []
        for i, j in zip(*alignment.indices):
            if i < 0 or j < 0:
                gaps += 1
                gap_run += 1
                longest_gap = max(longest_gap, gap_run)
                q = forward.qualities[i] if i >= 0 else rq[j]
                qt_conflicts += q >= qt
                markers.append(" ")
                continue
            gap_run = 0
            q = min(forward.qualities[i], rq[j])
            if fseq[i] not in "ACGT" or rseq[j] not in "ACGT":
                ambiguous += 1
                qt_conflicts += q >= qt
                markers.append("?")
            elif fseq[i] == rseq[j]:
                matches += 1
                qt_matches += q >= qt
                markers.append("|")
            else:
                mismatches += 1
                qt_conflicts += q >= qt
                markers.append(".")
        paired = matches + mismatches + ambiguous
        coordinates = alignment.coordinates
        fs, fe, rs, re_ = (int(coordinates[0, 0]), int(coordinates[0, -1]),
                            int(coordinates[1, 0]), int(coordinates[1, -1]))
        junction_fr = len(fseq) - fe + rs
        junction_rf = len(rseq) - re_ + fs
        candidates.append({
            "score": float(alignment.score), "paired_bases": paired,
            "base_identity": round(100 * matches / paired, 2) if paired else 0,
            "gap_identity": round(100 * matches / (paired + gaps), 2) if paired + gaps else 0,
            "matches": matches, "mismatches": mismatches, "gaps": gaps,
            "ambiguous": ambiguous, "longest_gap": longest_gap,
            "qt_matches": int(qt_matches), "qt_conflicts": int(qt_conflicts),
            "junction_unaligned": min(junction_fr, junction_rf),
            "connection": "F → R" if junction_fr <= junction_rf else "R → F",
            "reverse_orientation": "Reverse complement" if complemented else "원본 방향",
            "forward_start": fs, "forward_end": fe, "reverse_start": rs, "reverse_end": re_,
            "aligned_f": str(alignment[0]), "aligned_r": str(alignment[1]),
            "markers": "".join(markers),
        })
    if not candidates:
        return None
    # 길이 20 bp/80%는 짧은 우연 일치를 구분하는 참고 표시에만 사용한다.
    return max(candidates, key=lambda x: (
        x["paired_bases"] >= 20 and x["base_identity"] >= 80,
        x["score"], x["paired_bases"], -x["junction_unaligned"],
    ))


def overlap_label(overlap):
    if overlap is None:
        return "정렬 후보 없음"
    if overlap["paired_bases"] < 20 or overlap["base_identity"] < 80:
        return "짧거나 약한 국소 일치"
    return "겹침 후보 있음"


def split_phrap_options(text):
    """쉘을 사용하지 않으며 옵션명/숫자 값만 허용한다."""
    args = shlex.split(text)
    result = []
    for arg in args:
        if arg == "-new_ace":
            continue
        if arg in {"-old_ace", "-ace"}:
            raise ValueError("출력 판독에는 -new_ace를 사용합니다. 다른 ACE 옵션은 제거해 주세요.")
        if re.fullmatch(r"-[A-Za-z][A-Za-z0-9_]*", arg):
            result.append(arg)
            continue
        try:
            numeric = float(arg)
        except ValueError as exc:
            raise ValueError("추가 옵션에는 옵션명과 숫자만 넣어 주세요. 파일 경로는 지원하지 않습니다.") from exc
        if not math.isfinite(numeric):
            raise ValueError("옵션의 숫자는 유한한 값이어야 합니다.")
        result.append(arg)
    return result


def resolve_phrap(executable):
    candidate = shutil.which(executable)
    if candidate:
        return str(Path(candidate).resolve())
    path = Path(executable).expanduser()
    if path.is_file() and (os.name == "nt" or os.access(path, os.X_OK)):
        return str(path.resolve())
    raise FileNotFoundError("phrap 실행 파일을 찾지 못했습니다. 실행 환경의 경로를 확인해 주세요.")


def read_fasta_ids(text):
    return {record.id for record in SeqIO.parse(io.StringIO(text), "fasta")}


def parse_phrap_outputs(files, read_ids, console_text=""):
    """Contig 파일 개수로 결합을 추정하지 않고 ACE read 소속을 확인한다."""
    combined = console_text + "\n" + files.get("reads.fasta.ace", "")
    versions = sorted(set(re.findall(r"\bphrap\s+version\s+([0-9.]+)", combined, re.I)))
    version = versions[0] if len(versions) == 1 else None
    base = {
        "version": version,
        "version_matches": version == TARGET_PHRAP_VERSION,
        "contigs": [], "singlets": [],
    }
    if "reads.fasta.ace" not in files:
        return dict(base, state="unreadable", label="ACE 출력 없음 · 결합 판단 불가")
    try:
        ace = Ace.read(io.StringIO(files["reads.fasta.ace"]))
        groups = []
        for contig in ace.contigs:
            members = {read.rd.name for read in contig.reads}
            groups.append({"name": contig.name, "members": sorted(members), "bases": contig.nbases})
        singles = read_fasta_ids(files.get("reads.fasta.singlets", ""))
    except (ValueError, AssertionError, IndexError, StopIteration) as exc:
        return dict(base, state="unreadable", label=f"phrap 출력 해석 실패: {exc}")
    base.update(contigs=groups, singlets=sorted(singles))
    expected = set(read_ids)
    if any(expected <= set(group["members"]) for group in groups):
        return dict(base, state="joined", label="F/R 동일 contig 확인")
    seen = singles | {member for group in groups for member in group["members"]}
    if expected <= seen:
        return dict(base, state="separate", label="F/R 분리 출력 확인")
    missing = sorted(expected - seen)
    return dict(base, state="incomplete", label="일부 read 출력 미확인: " + ", ".join(missing))


def run_phrap(packet, executable, options="", timeout=60):
    if any(not read.sequence for read in packet["reads"]):
        return {"state": "skipped", "label": "빈 read 포함 · phrap 미실행", "files": {},
                "version": None, "version_matches": False, "log": ""}
    path = resolve_phrap(executable)
    args = [path, "reads.fasta", "-new_ace", *split_phrap_options(options)]
    with tempfile.TemporaryDirectory(prefix="contig_phrap_") as tmp:
        directory = Path(tmp)
        (directory / "reads.fasta").write_bytes(packet["fasta"])
        (directory / "reads.fasta.qual").write_bytes(packet["qual"])
        try:
            process = subprocess.run(args, cwd=directory, capture_output=True,
                text=True, errors="replace", timeout=timeout, shell=False)
        except subprocess.TimeoutExpired:
            return {"state": "error", "label": "phrap 실행 시간 초과", "files": {},
                    "version": None, "version_matches": False, "log": "timeout", "command": args}
        log = process.stdout + "\n" + process.stderr
        files = {}
        for suffix in ("ace", "contigs", "contigs.qual", "singlets", "log", "problems"):
            output = directory / f"reads.fasta.{suffix}"
            if output.is_file() and output.stat().st_size <= MAX_UPLOAD_BYTES:
                files[output.name] = output.read_text(errors="replace")
        if process.returncode:
            return {"state": "error", "label": f"phrap 실행 오류 (종료 코드 {process.returncode})",
                    "files": files, "version": None, "version_matches": False,
                    "log": log, "command": args}
        result = parse_phrap_outputs(files, [r.name for r in packet["reads"]], log)
        result.update(files=files, log=log, command=args)
        return result


def analyze_packets(packets, executable=None, options=""):
    if executable is not None:
        # 4회 실행 전에 설정 오류를 먼저 보고한다.
        resolve_phrap(executable)
        split_phrap_options(options)
    rows = []
    for condition in CONDITIONS:
        packet = packets.get(condition.label)
        if packet is None:
            rows.append({"condition": asdict(condition), "available": False,
                         "label": "해당 조건 입력 없음"})
            continue
        forward, reverse = packet["reads"]
        overlap = inspect_overlap(forward, reverse, condition.qt)
        engine = run_phrap(packet, executable, options) if executable else None
        rows.append({"condition": asdict(condition), "available": True,
            "source": packet["input_source"], "lengths": [len(forward.sequence), len(reverse.sequence)],
            "overlap": overlap, "label": overlap_label(overlap), "engine": engine,
            "company_result": "미검증",
        })
    return rows


def source_text(source):
    return "시범 전처리 · 회사 규칙 미확인" if source == "preview_assumption" else "제공된 전처리 입력 · 사내 일치 미확인"


def condition_table():
    return pd.DataFrame([{
        "조건": c.label, "QT": c.qt, "New QT": "사용" if c.new_qt_enabled else "미사용",
        "Window size": str(c.window_size) if c.new_qt_enabled else "미적용", "2nd QT": "미사용",
    } for c in CONDITIONS])


def summary_rows(rows):
    records = []
    for row in rows:
        condition = row["condition"]
        overlap = row.get("overlap") or {}
        engine = row.get("engine") or {}
        records.append({
            "조건": condition["label"], "QT": condition["qt"],
            "New QT 사용": condition["new_qt_enabled"], "Window size": condition["window_size"],
            "2nd QT 사용": condition["second_qt_enabled"],
            "입력": source_text(row["source"]) if row.get("available") else "없음",
            "F 길이": row.get("lengths", [None, None])[0],
            "R 길이": row.get("lengths", [None, None])[1],
            "겹침 후보 bp": overlap.get("paired_bases"), "Gap 포함 일치율": overlap.get("gap_identity"),
            "겹침 참고 평가": row["label"], "phrap 실행 결과": engine.get("label", "미실행"),
            "phrap 버전": engine.get("version"), "회사 결과": "미검증",
            "회사 실측 입력": "", "실측 입력 시각": "",
        })
    return records


def comparison_note(rows):
    executed = [row for row in rows if row.get("engine")]
    if not executed:
        return "phrap을 실행하지 않았습니다. 네 조건의 겹침 지표를 참고해 실제 결과를 확인해 주세요."
    joined = [row["condition"]["label"] for row in executed if row["engine"]["state"] == "joined"]
    if joined:
        return "이번 입력으로 phrap 결합이 확인된 조건: " + ", ".join(joined) + ". 회사 처리 결과와의 일치는 별도 확인이 필요합니다."
    return "이번 실행에서 F/R이 같은 contig에 속한 조건은 확인되지 않았습니다. 조건별 분리·오류·미출력 상태를 확인해 주세요."


def export_bundle(packets, rows):
    buffer = io.BytesIO()
    manifest = {
        "app_version": APP_VERSION, "target_phrap_version": TARGET_PHRAP_VERSION,
        "company_trimming_verified": False, "conditions": {},
    }
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for condition in CONDITIONS:
            packet = packets.get(condition.label)
            if packet is None:
                continue
            for name in ("fasta", "qual"):
                filename = "reads.fasta" if name == "fasta" else "reads.fasta.qual"
                archive.writestr(f"{condition.key}/{filename}", packet[name])
            manifest["conditions"][condition.label] = {
                key: packet[key] for key in ("condition", "input_source", "trim_metadata", "fasta_sha256", "qual_sha256")
            }
            manifest["conditions"][condition.label]["original_reads"] = [
                {"name": r.name, "file_name": r.file_name, "source_sha256": r.source_sha256}
                for r in packet["reads"]
            ]
        for row in rows:
            engine = row.get("engine")
            if not engine:
                continue
            prefix = row["condition"]["label"].replace("/", "_") + "/phrap_output/"
            for filename, data in engine.get("files", {}).items():
                archive.writestr(prefix + filename, data)
            archive.writestr(prefix + "console.txt", engine.get("log", ""))
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        archive.writestr("analysis.json", json.dumps(rows, ensure_ascii=False, indent=2))
        archive.writestr("comparison.csv", pd.DataFrame(summary_rows(rows)).to_csv(index=False).encode("utf-8-sig"))
        archive.writestr("README.txt", (
            "QT/Window 비교 자료\n\n"
            "20/10 = QT20, New QT 사용, Window size10. 30/20 = QT30, New QT 사용, Window size20.\n"
            "단독 표기 16과 10은 New QT 미사용이며 모든 조건에서 2nd QT를 사용하지 않습니다.\n"
            "preview_assumption 입력은 시범 trimming 결과입니다. 회사 전처리와 동일하다고 간주하지 마세요.\n"
            "provided_unverified 입력은 업로드된 FASTA/QUAL bytes를 그대로 사용한 결과입니다.\n"
            "회사 실측은 comparison.csv의 빈 칸에 입력합니다. 자동 예측이나 학습값으로 채우지 않습니다.\n"
            "재분석: 이 ZIP을 '전처리 입력 ZIP'으로 업로드할 수 있습니다.\n"
            "사내 입력은 조건 폴더(16,20_10,30_20,10)의 reads.fasta 및 reads.fasta.qual로 넣습니다.\n"
            "각 입력에는 F/R 두 read를 같은 ID/순서로 넣고, 각 read의 서열/Quality 길이를 맞춥니다.\n"
            "원본 입력 방향은 phrap에 그대로 전달합니다. 참고 정렬에서만 R 방향을 비교합니다.\n"
            "phrap 결과는 이 입력과 기록된 옵션에서 실제 실행한 결과이며 회사 결과의 재현을 보증하지 않습니다.\n"
        ))
    return buffer.getvalue()


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


def render_result(rows, packets):
    st.subheader("네 조건 비교")
    st.info(comparison_note(rows))
    if any(row.get("source") == "preview_assumption" for row in rows):
        st.caption("아래 길이와 겹침은 선택한 시범 전처리 방식의 결과입니다. 회사의 Contig/Contig2 예측으로 표시하지 않습니다.")
    # 가로 드래그 없이 네 조건을 확인한다. 결과가 없거나 오류인 조건도 숨기지 않는다.
    for start in (0, 2):
        columns = st.columns(2)
        for column, row in zip(columns, rows[start:start+2]):
            with column, st.container(border=True):
                condition = row["condition"]
                st.markdown(f"### {condition['label']}")
                window = str(condition["window_size"]) if condition["new_qt_enabled"] else "미적용"
                new_label = "사용" if condition["new_qt_enabled"] else "미사용"
                st.caption(f"QT {condition['qt']} · New QT {new_label} · Window {window} · 2nd QT 미사용")
                if not row["available"]:
                    st.write(row["label"])
                    continue
                st.write(f"분석 입력 길이: F **{row['lengths'][0]} bp** / R **{row['lengths'][1]} bp**")
                overlap = row["overlap"]
                if overlap:
                    st.write(f"겹침 후보 **{overlap['paired_bases']} bp** · Gap 포함 일치율 **{overlap['gap_identity']:.2f}%**")
                st.write(f"참고 평가: {row['label']}")
                engine = row["engine"]
                if not engine:
                    st.caption("phrap 미실행 · 회사 결과 미검증")
                else:
                    message = "phrap 실행: " + engine["label"]
                    if engine["state"] == "joined":
                        st.success(message)
                    elif engine["state"] in {"error", "unreadable"}:
                        st.error(message)
                    else:
                        st.warning(message)
                    if not engine["version_matches"]:
                        st.warning(f"실행 버전: {engine.get('version') or '미확인'} · 목표 버전 0.990319와 일치 확인 필요")
                    st.caption(source_text(row["source"]) + " · 회사 결과 미검증")

    with st.expander("전처리 구간·정렬 상세", expanded=False):
        selected = st.selectbox("상세 조건", [c.label for c in CONDITIONS], key="details_condition")
        packet = packets.get(selected)
        row = next(r for r in rows if r["condition"]["label"] == selected)
        if packet:
            st.caption(source_text(packet["input_source"]))
            if packet["trim_metadata"]:
                st.dataframe(pd.DataFrame(packet["trim_metadata"]), hide_index=True, use_container_width=True)
            else:
                st.caption("제공된 전처리 서열을 그대로 사용했습니다. 원본 AB1 내 절단 좌표는 알 수 없습니다.")
            overlap = row.get("overlap")
            if overlap:
                st.write(f"R 처리: {overlap['reverse_orientation']} · 연결 방향: {overlap['connection']}")
                st.write(f"Gap {overlap['gaps']} bp / 최장 연속 Gap {overlap['longest_gap']} bp / "
                         f"QT 이상 일치 {overlap['qt_matches']} bp / QT 이상 충돌 {overlap['qt_conflicts']}개")
                st.write(f"정렬 밖 연결 경계 잔여: {overlap['junction_unaligned']} bp")
                st.code(alignment_preview(overlap), language=None)
            engine = row.get("engine")
            if engine:
                st.write("실행 인수", engine.get("command", []))
                st.code(engine.get("log", ""), language=None)
        else:
            st.info("해당 조건의 입력이 없습니다.")

    st.subheader("재반응 검토")
    st.write("재반응 성공 확률은 계산하지 않습니다. 조건별로 남은 길이, 겹침 위치와 원본 파형을 함께 확인해 주세요.")
    st.caption("겹침 후보가 짧거나 내부 불일치가 많으면 F/R 파형을 우선 확인합니다. 참고 정렬의 gap만으로 실제 InDel 또는 혼합을 확정할 수 없습니다.")
    st.download_button("비교 결과와 phrap 입력 받기", export_bundle(packets, rows),
        file_name="contig_QT_window_comparison.zip", mime="application/zip", key="result_download")


def main():
    st.set_page_config(page_title="Contig 조건 비교", page_icon="🧬", layout="wide")
    st.title("🧬 Contig 조건 비교")
    st.write("QT와 Window size를 구분해 네 조건을 비교합니다. 실제 phrap 결과는 실행했을 때만 표시합니다.")
    st.caption(f"{APP_VERSION} · 대상 엔진 phrap {TARGET_PHRAP_VERSION}")
    st.subheader("조건 설정")
    st.dataframe(condition_table(), hide_index=True, use_container_width=True)
    st.caption("20/10의 10과 30/20의 20은 Window size입니다. 16·10은 New QT 미사용이며 2nd QT는 네 조건 모두 미사용입니다.")

    mode = st.radio("입력 종류", ("AB1 참고 분석", "전처리 입력 ZIP"), horizontal=True, key="input_mode")
    method = "mean_q"
    f_upload = r_upload = zip_upload = None
    if mode == "AB1 참고 분석":
        left, right = st.columns(2)
        with left:
            f_upload = st.file_uploader("Forward AB1", type=["ab1", "abi"], key="forward_file")
        with right:
            r_upload = st.file_uploader("Reverse AB1", type=["ab1", "abi"], key="reverse_file")
        st.info("사내 trimming 규칙이 아직 확인되지 않았습니다. AB1 참고 분석은 아래에 명시한 시범 규칙을 사용합니다.")
        with st.expander("시범 전처리 방식과 적용 범위", expanded=False):
            method = st.selectbox("Window 계산 방식", list(PREVIEW_METHODS),
                format_func=PREVIEW_METHODS.get, key="preview_method")
            st.markdown(
                "- **New QT 사용:** 지정한 Window size의 구간을 1 bp씩 이동하며 평가합니다.\n"
                "- **New QT 미사용:** 시범 계산에서는 각 염기의 Q로 양 끝 경계를 찾습니다.\n"
                "- 처음 통과한 구간부터 마지막 통과 구간까지 유지하며, 내부 저품질 염기를 삭제하지 않습니다.\n"
                "- 평균 Quality는 Q의 산술평균, 평균 오류확률은 평균 `10^(-Q/10)`을 QT의 오류확률과 비교합니다.\n"
                "- 두 방식 모두 회사 구현으로 확인된 방식이 아닙니다. Window size를 품질 임계값으로 사용하지 않습니다."
            )
    else:
        zip_upload = st.file_uploader("조건별 FASTA + QUAL ZIP", type=["zip"], key="processed_zip")
        st.caption("제공된 서열과 Quality를 추가 trimming 없이 그대로 분석·실행합니다. 각 조건은 F/R 두 read를 같은 ID와 순서로 포함해야 합니다.")
        with st.expander("입력 ZIP 구성", expanded=False):
            st.write("조건 폴더명: 16, 20_10, 30_20, 10. 각 폴더에 reads.fasta와 reads.fasta.qual을 넣어 주세요. 일부 조건만 넣어도 비교할 수 있습니다.")
            st.write("이 앱에서 내려받은 비교 ZIP도 다시 불러올 수 있습니다. 시범 전처리 자료는 재업로드해도 시범 자료로 표시됩니다.")

    with st.expander("phrap 실행 연결", expanded=False):
        execute = st.checkbox("분석할 때 phrap도 실제 실행", value=False, key="run_phrap")
        executable = st.text_input("phrap 실행 파일 경로", value=os.environ.get("PHRAP_EXECUTABLE", "phrap"), key="phrap_path")
        options = st.text_input("회사에서 사용하는 추가 실행 옵션", value="", key="phrap_options")
        st.caption("실행 파일은 이 Streamlit 앱이 돌아가는 컴퓨터에 있어야 합니다. -new_ace는 결과 판독을 위해 자동 추가합니다. 옵션이 비어 있으면 해당 실행 파일의 기본값을 사용합니다.")
        st.caption("시범 전처리 입력으로 phrap을 실행해도 회사 결과가 재현됐다는 뜻은 아닙니다. 동일한 입력·버전·옵션을 먼저 확인해야 합니다.")

    ready = (f_upload is not None and r_upload is not None) if mode == "AB1 참고 분석" else zip_upload is not None
    upload_bytes = (
        [f_upload.getvalue(), r_upload.getvalue()] if ready and mode == "AB1 참고 분석"
        else [zip_upload.getvalue()] if ready else []
    )
    fingerprint = sha256(json.dumps({"mode": mode, "method": method, "execute": execute,
        "executable": executable, "options": options, "hashes": [sha256(x) for x in upload_bytes]}, sort_keys=True).encode())
    clicked = st.button("네 조건 분석", type="primary", disabled=not ready, key="analyze_button")
    if clicked:
        st.session_state.pop("comparison_result", None)
        try:
            with st.spinner("네 조건의 입력과 겹침을 분석하고 있습니다…"):
                reads = None
                if mode == "AB1 참고 분석":
                    reads = (read_ab1(upload_bytes[0], f_upload.name, "F"),
                             read_ab1(upload_bytes[1], r_upload.name, "R"))
                    packets = prepare_preview(reads, method)
                else:
                    packets = load_input_zip(upload_bytes[0])
                rows = analyze_packets(packets, executable if execute else None, options)
                st.session_state["comparison_result"] = (fingerprint, reads, packets, rows)
        except (ValueError, OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
            st.error(str(exc))
    result = st.session_state.get("comparison_result")
    if result:
        saved_fingerprint, reads, packets, rows = result
        if saved_fingerprint != fingerprint:
            st.info("입력 또는 설정이 바뀌었습니다. 네 조건 분석을 다시 실행해 주세요.")
            return
        if reads:
            with st.expander("AB1 원본 Quality", expanded=False):
                st.dataframe(quality_summary(reads), hide_index=True, use_container_width=True)
                for column, read in zip(st.columns(2), reads):
                    with column:
                        st.caption(read.file_name)
                        frame = pd.DataFrame({"염기 위치": range(1, len(read.qualities)+1), "Quality": read.qualities})
                        st.line_chart(frame.set_index("염기 위치"))
        render_result(rows, packets)


if __name__ == "__main__":
    main()
