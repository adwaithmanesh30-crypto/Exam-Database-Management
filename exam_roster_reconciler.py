#!/usr/bin/env python3
"""
Reconcile OCR-extracted exam results against an authoritative student roster.

Unlike a row-quality auditor, this utility checks whether every expected student
has a result, whether every result belongs to a registered student, and whether
names agree for matching student IDs. It uses only the Python standard library
and can write a machine-readable JSON reconciliation report.

Example:
  python exam_roster_reconciler.py roster.csv results.csv \
      --roster-id RollNumber --results-id RollNumber \
      --roster-name StudentName --results-name StudentName \
      --report reconciliation.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class ReconciliationError(ValueError):
    """Raised when an input table cannot be reconciled safely."""


@dataclass(frozen=True)
class StudentRecord:
    """One validated student row and its source line."""

    student_id: str
    name: str
    line: int


def normalize_identifier(value: str) -> str:
    """Normalize an identifier for case-insensitive matching."""
    return unicodedata.normalize("NFKC", value).strip().casefold()


def normalize_name(value: str) -> str:
    """Normalize Unicode and whitespace without discarding name punctuation."""
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def resolve_column(requested: str, headers: list[str], source: Path) -> str:
    """Resolve a requested column without accepting ambiguous headers."""
    matches = [header for header in headers
               if header.casefold() == requested.strip().casefold()]
    if len(matches) != 1:
        raise ReconciliationError(
            f"{source}: column {requested!r} was not found uniquely"
        )
    return matches[0]


def load_students(path: Path, delimiter: str, id_column: str,
                  name_column: str) -> dict[str, StudentRecord]:
    """Read and validate a roster or result CSV, keyed by normalized ID."""
    students: dict[str, StudentRecord] = {}
    reader = None

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, delimiter=delimiter, strict=True)
            try:
                raw_headers = next(reader)
            except StopIteration:
                raise ReconciliationError(f"{path}: file is empty") from None

            headers = [header.strip() for header in raw_headers]
            if not headers or any(not header for header in headers):
                raise ReconciliationError(
                    f"{path}: header row contains an empty column name"
                )
            if len({header.casefold() for header in headers}) != len(headers):
                raise ReconciliationError(
                    f"{path}: duplicate column names make the file ambiguous"
                )

            id_header = resolve_column(id_column, headers, path)
            name_header = resolve_column(name_column, headers, path)
            id_index = headers.index(id_header)
            name_index = headers.index(name_header)

            for row in reader:
                line = reader.line_num
                if not row or all(not cell.strip() for cell in row):
                    continue
                if len(row) != len(headers):
                    raise ReconciliationError(
                        f"{path}: line {line} has {len(row)} fields; "
                        f"expected {len(headers)}"
                    )

                raw_id = row[id_index].strip()
                name = row[name_index].strip()
                if not raw_id:
                    raise ReconciliationError(
                        f"{path}: student ID is blank on line {line}"
                    )
                if not name:
                    raise ReconciliationError(
                        f"{path}: student name is blank on line {line}"
                    )

                key = normalize_identifier(raw_id)
                if key in students:
                    previous = students[key]
                    raise ReconciliationError(
                        f"{path}: duplicate student ID {raw_id!r} on line "
                        f"{line}; first seen on line {previous.line}"
                    )

                students[key] = StudentRecord(
                    student_id=raw_id,
                    name=name,
                    line=line,
                )
    except csv.Error as exc:
        line = reader.line_num if reader is not None else "unknown"
        raise ReconciliationError(
            f"{path}: malformed CSV near line {line}: {exc}"
        ) from exc

    return students


def reconcile(roster: dict[str, StudentRecord],
              results: dict[str, StudentRecord],
              roster_path: Path, results_path: Path) -> dict:
    """Compare roster and result records and return a JSON-safe report."""
    roster_keys = set(roster)
    result_keys = set(results)
    missing = [
        {
            "student_id": roster[key].student_id,
            "student_name": roster[key].name,
            "roster_line": roster[key].line,
        }
        for key in sorted(roster_keys - result_keys)
    ]
    unexpected = [
        {
            "student_id": results[key].student_id,
            "student_name": results[key].name,
            "results_line": results[key].line,
        }
        for key in sorted(result_keys - roster_keys)
    ]
    name_mismatches = []
    for key in sorted(roster_keys & result_keys):
        expected = roster[key]
        actual = results[key]
        if normalize_name(expected.name) != normalize_name(actual.name):
            name_mismatches.append({
                "student_id": expected.student_id,
                "roster_name": expected.name,
                "results_name": actual.name,
                "roster_line": expected.line,
                "results_line": actual.line,
            })

    return {
        "roster_file": str(roster_path),
        "results_file": str(results_path),
        "roster_students": len(roster),
        "result_students": len(results),
        "matched_students": len(roster_keys & result_keys),
        "missing_results": missing,
        "unexpected_results": unexpected,
        "name_mismatches": name_mismatches,
        "is_clean": not missing and not unexpected and not name_mismatches,
    }


def make_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Reconcile OCR exam results against a student roster."
    )
    parser.add_argument("roster_csv", type=Path,
                        help="authoritative list of enrolled students")
    parser.add_argument("results_csv", type=Path,
                        help="OCR-exported exam results to reconcile")
    parser.add_argument("--roster-id", default="StudentID",
                        metavar="COLUMN", help="student ID column in the roster")
    parser.add_argument("--results-id", default="StudentID",
                        metavar="COLUMN", help="student ID column in results")
    parser.add_argument("--roster-name", default="StudentName",
                        metavar="COLUMN", help="student name column in the roster")
    parser.add_argument("--results-name", default="StudentName",
                        metavar="COLUMN", help="student name column in results")
    parser.add_argument("--delimiter", default=",",
                        help="single-character CSV delimiter (default: comma)")
    parser.add_argument("--report", type=Path,
                        help="write a JSON report to a new file; never overwrites")
    return parser


def write_report(path: Path, report: dict,
                 input_paths: Sequence[Path]) -> None:
    """Write JSON exclusively so an existing file is never replaced."""
    resolved = path.resolve()
    if any(resolved == input_path.resolve() for input_path in input_paths):
        raise ValueError("report path must differ from both input files")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def print_report(report: dict) -> None:
    """Print a concise human-readable reconciliation summary."""
    print("Exam roster reconciliation")
    print(f"Roster students: {report['roster_students']}")
    print(f"Result students: {report['result_students']}")
    print(f"Matched students: {report['matched_students']}")
    print(f"Missing results: {len(report['missing_results'])}")
    print(f"Unexpected results: {len(report['unexpected_results'])}")
    print(f"Name mismatches: {len(report['name_mismatches'])}")

    for item in report["missing_results"]:
        print(f"[MISSING] {item['student_id']}: {item['student_name']}")
    for item in report["unexpected_results"]:
        print(f"[UNREGISTERED] {item['student_id']}: {item['student_name']}")
    for item in report["name_mismatches"]:
        print(
            f"[NAME MISMATCH] {item['student_id']}: roster="
            f"{item['roster_name']!r}, results={item['results_name']!r}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run validation, reconciliation, and optional report output."""
    parser = make_parser()
    args = parser.parse_args(argv)

    if len(args.delimiter) != 1 or args.delimiter in "\r\n":
        parser.error("--delimiter must be exactly one non-newline character")
    if args.roster_csv.resolve() == args.results_csv.resolve():
        parser.error("roster and results must be different files")

    try:
        roster = load_students(
            args.roster_csv, args.delimiter, args.roster_id, args.roster_name
        )
        results = load_students(
            args.results_csv, args.delimiter, args.results_id, args.results_name
        )
        report = reconcile(roster, results, args.roster_csv, args.results_csv)
        if args.report:
            write_report(args.report, report, (args.roster_csv, args.results_csv))
        print_report(report)
    except (OSError, UnicodeError, csv.Error, ReconciliationError, ValueError) as exc:
        print(f"Reconciliation failed: {exc}", file=sys.stderr)
        return 2

    return 0 if report["is_clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
