#!/usr/bin/env python3
"""
Validate OCR-exported exam CSV files before database import.

This is a pre-import quality gate, not an OCR engine or database manager. It
checks required fields, duplicate student identifiers, numeric mark ranges,
row widths, duplicate headers, and suspicious whitespace. It uses only the
Python standard library and can emit a JSON audit report.

Example:
  python exam_csv_quality_auditor.py marks.csv --required RollNumber --required StudentName --unique RollNumber --range Mathematics=0:100 --report audit.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Sequence


def parse_range(spec: str) -> tuple[str, Decimal, Decimal]:
    """Parse a range option formatted as COLUMN=MIN:MAX."""
    try:
        column, limits = spec.split("=", 1)
        low_text, high_text = limits.split(":", 1)
        low, high = Decimal(low_text), Decimal(high_text)
    except (ValueError, InvalidOperation):
        raise argparse.ArgumentTypeError(
            "use COLUMN=MIN:MAX, for example Marks=0:100"
        ) from None
    if not column.strip() or not low.is_finite() or not high.is_finite() or low > high:
        raise argparse.ArgumentTypeError("invalid column name or numeric limits")
    return column.strip(), low, high


def resolve_column(name: str, headers: list[str]) -> str:
    """Find a header case-insensitively while preserving its original spelling."""
    matches = [header for header in headers if header.casefold() == name.strip().casefold()]
    if len(matches) != 1:
        raise ValueError(f"column {name!r} was not found uniquely in the CSV header")
    return matches[0]


def add_issue(report: dict, limit: int, line: int | None, column: str | None,
              code: str, message: str, severity: str = "error") -> None:
    report[severity + "s"] += 1
    if len(report["issues"]) < limit:
        report["issues"].append({
            "severity": severity, "line": line, "column": column,
            "code": code, "message": message,
        })
    else:
        report["issues_omitted"] += 1


def audit_csv(path: Path, required: Sequence[str], unique: Sequence[str],
              ranges: Sequence[tuple[str, Decimal, Decimal]], delimiter: str,
              issue_limit: int) -> dict:
    report = {
        "source": str(path), "headers": [], "records_read": 0,
        "valid_records": 0, "invalid_records": 0, "blank_rows": 0,
        "errors": 0, "warnings": 0, "issues_omitted": 0,
        "issues": [], "fatal_error": None,
    }
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter=delimiter, strict=True)
        try:
            raw_headers = next(reader)
        except StopIteration:
            report["fatal_error"] = "CSV is empty; a header row is required."
            return report

        headers = [item.strip() for item in raw_headers]
        report["headers"] = headers
        if not headers or any(not item for item in headers):
            report["fatal_error"] = "Header row contains an empty column name."
            return report
        if len({item.casefold() for item in headers}) != len(headers):
            report["fatal_error"] = "Duplicate headers are ambiguous; rename them."
            return report

        for index, (raw, clean) in enumerate(zip(raw_headers, headers), start=1):
            if raw != clean:
                add_issue(report, issue_limit, 1, clean,
                          "header_whitespace", "Header whitespace was trimmed.", "warning")

        try:
            required_cols = [resolve_column(name, headers) for name in required]
            unique_cols = list(dict.fromkeys(resolve_column(name, headers) for name in unique))
            checked_ranges = [
                (resolve_column(name, headers), low, high)
                for name, low, high in ranges
            ]
        except ValueError as exc:
            report["fatal_error"] = str(exc)
            return report

        first_seen: dict[tuple[str, ...], int] = {}
        try:
            for row in reader:
                line = reader.line_num
                if not row or all(not cell.strip() for cell in row):
                    report["blank_rows"] += 1
                    continue
                report["records_read"] += 1
                failed = False
                if len(row) != len(headers):
                    add_issue(report, issue_limit, line, None, "column_count",
                              f"Expected {len(headers)} fields, received {len(row)}.")
                    failed = True

                values = {header: row[i] if i < len(row) else ""
                          for i, header in enumerate(headers)}
                for column, value in values.items():
                    if value != value.strip():
                        add_issue(report, issue_limit, line, column, "field_whitespace",
                                  "Value has leading or trailing whitespace.", "warning")
                    if any((ord(ch) < 32 and ch not in "\t\r\n") or ord(ch) == 127
                           for ch in value):
                        add_issue(report, issue_limit, line, column, "control_character",
                                  "Value contains a non-printable control character.")
                        failed = True

                for column in required_cols:
                    if not values[column].strip():
                        add_issue(report, issue_limit, line, column, "required_value",
                                  "Required value is blank.")
                        failed = True

                for column, low, high in checked_ranges:
                    raw_value = values[column].strip()
                    if not raw_value:
                        continue
                    try:
                        number = Decimal(raw_value)
                    except InvalidOperation:
                        add_issue(report, issue_limit, line, column, "not_numeric",
                                  f"Expected a number, received {raw_value!r}.")
                        failed = True
                        continue
                    if not number.is_finite() or not low <= number <= high:
                        add_issue(report, issue_limit, line, column, "out_of_range",
                                  f"Value {raw_value!r} must be between {low} and {high}.")
                        failed = True

                if unique_cols and all(values[col].strip() for col in unique_cols):
                    key = tuple(values[col].strip().casefold() for col in unique_cols)
                    if key in first_seen:
                        add_issue(report, issue_limit, line, ", ".join(unique_cols),
                                  "duplicate_key", f"Key first appeared on line {first_seen[key]}.")
                        failed = True
                    else:
                        first_seen[key] = line

                report["invalid_records" if failed else "valid_records"] += 1
        except csv.Error as exc:
            report["fatal_error"] = f"Malformed CSV near line {reader.line_num}: {exc}"
    return report


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit an OCR-generated exam CSV before database import."
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--required", action="append", default=[], metavar="COLUMN",
                        help="required non-empty column; repeatable")
    parser.add_argument("--unique", action="append", default=[], metavar="COLUMN",
                        help="unique key column; repeated options form a composite key")
    parser.add_argument("--range", dest="ranges", action="append", type=parse_range,
                        default=[], metavar="COLUMN=MIN:MAX",
                        help="numeric bounds for a column; repeatable")
    parser.add_argument("--delimiter", default=",",
                        help="single-character CSV delimiter (default: comma)")
    parser.add_argument("--max-issues", type=int, default=200,
                        help="maximum issue details to retain (default: 200)")
    parser.add_argument("--report", type=Path,
                        help="write JSON report to a new file; never overwrites")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    if len(args.delimiter) != 1 or args.delimiter in "\r\n":
        parser.error("--delimiter must be exactly one non-newline character")
    if args.max_issues < 1:
        parser.error("--max-issues must be at least 1")

    try:
        report = audit_csv(args.input_csv, args.required, args.unique, args.ranges,
                           args.delimiter, args.max_issues)
    except (OSError, UnicodeError, csv.Error) as exc:
        print(f"Cannot audit {args.input_csv}: {exc}", file=sys.stderr)
        return 2

    if args.report:
        try:
            if args.report.resolve() == args.input_csv.resolve():
                raise ValueError("report path must differ from the input file")
            with args.report.open("x", encoding="utf-8", newline="\n") as output:
                json.dump(report, output, indent=2, ensure_ascii=False)
                output.write("\n")
        except (OSError, ValueError) as exc:
            print(f"Cannot write report: {exc}", file=sys.stderr)
            return 2

    print(f"Exam CSV audit: {report['source']}")
    print(f"Records: {report['records_read']} | valid: {report['valid_records']} "
          f"| invalid: {report['invalid_records']} | blank: {report['blank_rows']}")
    print(f"Errors: {report['errors']} | warnings: {report['warnings']}")
    if report["fatal_error"]:
        print(f"Stopped: {report['fatal_error']}")
    for issue in report["issues"]:
        where = f"line {issue['line']}" if issue["line"] is not None else "file"
        if issue["column"]:
            where += f", {issue['column']}"
        print(f"[{issue['severity'].upper()}] {where}: {issue['code']} - {issue['message']}")
    if report["issues_omitted"]:
        print(f"{report['issues_omitted']} further issue details omitted.")
    if report["fatal_error"]:
        return 2
    return 1 if report["invalid_records"] or report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
