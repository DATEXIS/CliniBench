import logging
from enum import IntEnum, StrEnum
from typing import Iterable

import polars as pl
from polars import col, DataFrame, element

from utils import log_function_call

logger = logging.getLogger(__name__)


class MIMICSplit(StrEnum):
    hosp = "hosp"
    icu = "icu"


class ICDVersion(IntEnum):
    nine = 9
    ten = 10


@log_function_call(logger=logger)
def extract_admission_notes(notes_df: pl.DataFrame) -> pl.DataFrame:
    """
    Keep only Discharge Summaries and filter out Newborn admissions. Replace duplicates and join reports with
    their addendums. If admission_text_only is True, filter all sections that are not known at admission time.
    """
    notes_df = notes_df.with_columns(pl.col("text").str.strip_chars())

    # remove entries without subject id or text
    notes_df = notes_df.drop_nulls(subset=["subject_id", "text"])

    # reduce text to admission-only text
    notes_df = filter_admission_text(notes_df)

    return notes_df


def filter_admission_text(notes_df: pl.DataFrame) -> pl.DataFrame:
    """
    Filter text information by section and only keep sections that are known on admission time.
    """
    original_columns = notes_df.columns

    important_columns = [
        "Chief Complaint",
        "History of Present Illness",
        "Past Medical History",
    ]
    admission_sections = important_columns + [
        "Medications on Admission",
        "Allergies",
        "Physical Exam",
        "Family History",
        "Social History",
    ]

    notes_df = fix_mimic_iv_anonymization_mistakes(notes_df)

    notes_df = extract_sections(notes_df, admission_sections)

    notes_df = filter_if_all_columns_are_missing(notes_df, important_columns)

    notes_df = remove_discharge_data_from_physical_exam(notes_df)

    verify_no_other_section_in_admission_sections(notes_df, admission_sections)

    # add section headers and combine into admission note
    notes_df = notes_df.with_columns(
        pl.concat_str(
            [
                pl.lit(f"## {section.upper()}\n") + pl.col(section).cast(pl.Utf8)
                for section in admission_sections
            ],
            separator="\n\n",
        ).alias("text")
    )

    return notes_df.select(original_columns)


@log_function_call(logger=logger)
def fix_mimic_iv_anonymization_mistakes(df: pl.DataFrame):
    mistakes = {
        r"___\s*\nFamily History:": "___\n\nFamily History:",
        r"\n\s*\n___ Complaint:\n": "\n\nChief Complaint:\n",
        r"\n\s*\nPhysical ___:\n": "\n\nPhysical Exam:\n",
        r"\n\s*\n___ on Admission:\n": "\n\nMedication on Admission:\n",
    }

    expr = col("text")
    for mistake in mistakes.items():
        expr = expr.str.replace_all(*mistake)

    return df.with_columns(expr)


@log_function_call(logger=logger)
def extract_sections(df: pl.DataFrame, admission_sections: list[str]) -> DataFrame:
    return df.with_columns(
        "*",
        **{
            section: col("text")
            .str.extract_groups(
                rf"(?:^|\n\s*\n){section}:([\s\S]*?)\n\s*\n[^(\\|\d|\.)]+?:"
            )
            .struct["1"]
            .str.strip_chars()
            .fill_null("")
            for section in admission_sections
        },
    )


def remove_discharge_data_from_physical_exam(df: pl.DataFrame) -> DataFrame:
    lines_with_discharge_that_could_be_headlines = (
        r"([\s\S]*?)(?:(?:^|\n).*(?:D(?i)ischarge.*|discharge.*:)[\s\S]*)?$"
    )
    return df.with_columns(
        col("Physical Exam").str.extract(lines_with_discharge_that_could_be_headlines)
    )


@log_function_call(logger=logger)
def verify_no_other_section_in_admission_sections(
    df: pl.DataFrame, admission_sections: Iterable[str]
):
    section_names = set(
        collect_relevant_sections(df, 0.01).to_list() + admission_sections
    )
    for admission_section in admission_sections:
        for section in section_names:
            if admission_section == section:
                continue
            df_check = df.select(
                "hadm_id",
                col(admission_section),
                other_headline_in_admission_section=col(admission_section).str.contains(
                    rf"(?m)^{section}:"
                ),
            )

            other_headline_in_admission_section = df_check.filter(
                col("other_headline_in_admission_section")
            )
            if len(other_headline_in_admission_section) > 0:
                logger.debug(
                    f"Section headline '{section}' in '{admission_section}'!\n"
                    f"Offenders: {other_headline_in_admission_section}"
                )


def collect_relevant_sections(df: pl.DataFrame, min_frequency: float) -> pl.Series:
    section_name_counts = (
        df["text"]
        .str.extract_all(r"(?m)(?:^|\n\s*\n)((?:[A-Z]?[a-z]+ ?)*):\s*$")
        .list.eval(element().str.strip_chars().str.strip_chars_end(":"))
        .explode()
        .alias("section_name")
        .value_counts()
        .sort(by="count", descending=True)
    )
    section_name_counts = section_name_counts.select(
        "*", col("count").alias("relative") / len(df)
    )
    return section_name_counts.filter(col("relative") > min_frequency)["section_name"]


def extract_sections_old_behaviour(df, admission_sections):
    """
    Old van Aken behaviour.
    """
    return df.with_columns(
        *[_extract_section(key, section) for key, section in admission_sections.items()]
    )


def _extract_section(key, section):
    return (
        pl.col("text")
        .str.extract(rf"(?i){section}([\s\S]+?)\n\s*?\n[^(\\|\d|.)]+?:")
        .str.replace("\n", " ")
        .str.strip_chars()
        .fill_null("")
        .alias(key)
    )


def filter_if_all_columns_are_missing(df, columns_to_check):
    before_filter = len(df)
    df = df.filter(
        pl.any_horizontal([pl.col(col_name) != "" for col_name in columns_to_check])
    )
    logger.info(
        f"Filtered {before_filter - len(df)} admission notes due to missing all of {columns_to_check}."
    )
    return df
