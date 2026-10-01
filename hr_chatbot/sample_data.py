"""Generate synthetic candidates and jobs (all fictional) as real PDF and DOCX files.

    python -m hr_chatbot.sample_data            # writes into ./data
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf
from docx import Document

from .config import load_settings

# ----------------------------------------------------------------------------- PDF writer

_STYLES = {
    "title": ("hebo", 20, 0, 6),
    "h": ("hebo", 14, 0, 4),
    "bold": ("hebo", 10.5, 0, 2),
    "p": ("helv", 10.5, 0, 4),
    "b": ("helv", 10.5, 14, 2),
}


def _wrap(text: str, font: str, size: float, width: float) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if current and pymupdf.get_text_length(trial, fontname=font, fontsize=size) > width:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def write_pdf(path: Path, blocks: list[tuple[str, str]]) -> None:
    """blocks: (kind, text) with kind in title|h|bold|p|b|pagebreak."""
    width, height, margin = 595, 842, 54
    doc = pymupdf.open()
    page = doc.new_page(width=width, height=height)
    y = margin
    for kind, text in blocks:
        if kind == "pagebreak":
            page, y = doc.new_page(width=width, height=height), margin
            continue
        font, size, indent, gap = _STYLES[kind]
        if kind in ("h", "title") and y > margin:
            y += 8
        prefix = "- " if kind == "b" else ""
        for i, line in enumerate(_wrap(prefix + text, font, size, width - 2 * margin - indent)):
            if y + size * 1.4 > height - margin:
                page, y = doc.new_page(width=width, height=height), margin
            page.insert_text((margin + indent + (8 if kind == "b" and i else 0), y + size), line, fontname=font, fontsize=size)
            y += size * 1.4
        y += gap
    doc.save(str(path))
    doc.close()


def write_docx(path: Path, blocks: list[tuple[str, str]]) -> None:
    """blocks: (kind, text) with kind in title|h|p|b."""
    document = Document()
    for kind, text in blocks:
        if kind == "title":
            document.add_heading(text, level=0)
        elif kind == "h":
            document.add_heading(text, level=1)
        elif kind == "b":
            document.add_paragraph(text, style="List Bullet")
        else:
            document.add_paragraph(text)
    document.save(str(path))


# ----------------------------------------------------------------------------- content

CANDIDATE_001_RESUME = [
    ("title", "Jordan Ellis"),
    ("p", "Senior Data Platform Engineer | jordan.ellis@example.com | github.com/jordan-ellis-example"),
    ("h", "Summary"),
    ("p", "Backend and data platform engineer with 8 years of experience building Python services, batch and "
          "streaming data pipelines, and internal developer tooling. Led a team of four engineers at Northwind Analytics."),
    ("h", "Experience"),
    ("bold", "Staff Data Engineer, Northwind Analytics (2021 - Present)"),
    ("b", "Designed a Kafka and Spark Structured Streaming pipeline processing 2 billion events per day, reducing "
          "end-to-end latency from 15 minutes to 40 seconds."),
    ("b", "Built a Python FastAPI service layer over the data warehouse, used by 12 internal teams, and owned the on-call rotation."),
    ("b", "Mentored four engineers and introduced code review and testing standards (pytest, 85% coverage)."),
    ("pagebreak", ""),
    ("bold", "Software Engineer, Contoso Retail (2017 - 2021)"),
    ("b", "Migrated nightly ETL jobs from cron scripts to Apache Airflow, cutting failed runs by 70%."),
    ("b", "Developed PostgreSQL schemas and query optimizations for order analytics."),
    ("h", "Skills"),
    ("p", "Python, SQL, PostgreSQL, Apache Airflow, Apache Kafka, Spark, FastAPI, Docker, Kubernetes, "
          "AWS (S3, Glue, EMR), Terraform. Limited experience with Go and Rust."),
    ("h", "Education"),
    ("p", "BSc Computer Science, University of Example (2013 - 2017)"),
    ("h", "Certifications"),
    ("p", "AWS Certified Data Analytics - Specialty (2022)"),
]

CANDIDATE_001_COVER = [
    ("title", "Cover Letter"),
    ("p", "Dear Hiring Team,"),
    ("p", "I am excited to apply for the Senior Python Engineer role on your Data Platform team. Over the last eight "
          "years I have built and operated data pipelines in production, and I enjoy turning messy operational data "
          "into reliable services other teams can build on."),
    ("p", "I am particularly drawn to your focus on data quality and to the mentoring responsibilities in the role. "
          "I have not worked with dbt, although I have built similar in-house transformation frameworks in Python and SQL."),
    ("p", "Kind regards, Jordan Ellis"),
]

CANDIDATE_001_PORTFOLIO = [
    ("title", "Selected Projects"),
    ("h", "StreamKit"),
    ("p", "Open-source Python library that simplifies writing resilient Kafka consumers, with retry policies, dead-letter "
          "queues and metrics. Used in production at Northwind Analytics."),
    ("h", "Pipeline Observability Dashboard"),
    ("p", "Internal tool that tracks freshness, volume and schema drift for Airflow-managed datasets and alerts data owners "
          "through Slack. Built with FastAPI, PostgreSQL and a small React front end."),
]

CANDIDATE_002_RESUME = [
    ("title", "Priya Nair"),
    ("p", "Business Intelligence Analyst | priya.nair@example.com"),
    ("h", "Summary"),
    ("p", "Analyst with 3 years of experience turning sales and marketing data into dashboards and recommendations "
          "for revenue teams. Strong SQL and Tableau skills; learning Python for automation."),
    ("h", "Experience"),
    ("p", "Business Intelligence Analyst, Fabrikam Software (2023 - Present)"),
    ("b", "Built 20+ Tableau dashboards on Snowflake data used weekly by sales leadership."),
    ("b", "Automated monthly reporting with Python and pandas, saving about 12 hours per month."),
    ("p", "Junior Analyst, Litware Marketing (2021 - 2023)"),
    ("b", "Wrote SQL reports on campaign performance and presented findings to account managers."),
    ("h", "Skills"),
    ("p", "SQL, Tableau, Excel, Snowflake, basic Python (pandas), stakeholder communication. No experience with Kafka or Airflow."),
    ("h", "Education"),
    ("p", "BA Economics, Example State University (2017 - 2021)"),
]

JOB_001 = [
    ("title", "Senior Python Engineer - Data Platform"),
    ("h", "About the role"),
    ("p", "You will help build the internal data platform that powers analytics and machine learning across the company, "
          "owning pipelines end to end and raising engineering standards on a small, senior team."),
    ("h", "Responsibilities"),
    ("b", "Design, build and operate batch and streaming data pipelines."),
    ("b", "Develop Python services and APIs that expose data to other teams."),
    ("b", "Mentor engineers and lead code reviews."),
    ("h", "Required Skills"),
    ("b", "6+ years of professional Python experience."),
    ("b", "Hands-on experience designing production data pipelines, both batch and streaming."),
    ("b", "Strong SQL and PostgreSQL knowledge."),
    ("b", "Experience with a workflow orchestrator such as Apache Airflow."),
    ("b", "Docker and Kubernetes; experience with a major cloud provider (AWS preferred)."),
    ("h", "Preferred Skills"),
    ("b", "dbt, Apache Kafka, Terraform, Go."),
    ("b", "Prior experience mentoring or leading engineers."),
]

JOB_002 = [
    ("title", "Data Analyst - Revenue Operations"),
    ("h", "About the role"),
    ("p", "Partner with sales and finance leaders to measure pipeline health and forecast accuracy."),
    ("h", "Required Skills"),
    ("b", "2+ years of experience in an analytics role."),
    ("b", "Advanced SQL and experience with a BI tool such as Tableau or Looker."),
    ("b", "Clear written and verbal communication with non-technical stakeholders."),
    ("h", "Preferred Skills"),
    ("b", "Python for automation, experience with Snowflake."),
]


def generate(data_dir: Path) -> list[Path]:
    data_dir = Path(data_dir)
    written: list[Path] = []

    def out(rel: str) -> Path:
        path = data_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        written.append(path)
        return path

    write_pdf(out("candidates/candidate-001/resume.pdf"), CANDIDATE_001_RESUME)
    write_docx(out("candidates/candidate-001/cover-letter.docx"), CANDIDATE_001_COVER)
    write_pdf(out("candidates/candidate-001/portfolio.pdf"), CANDIDATE_001_PORTFOLIO)
    write_docx(out("candidates/candidate-002/resume.docx"), CANDIDATE_002_RESUME)
    write_docx(out("jobs/job-001/job-description.docx"), JOB_001)
    write_docx(out("jobs/job-002/job-description.docx"), JOB_002)
    return written


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else load_settings().data_dir
    files = generate(target)
    print(f"Wrote {len(files)} sample files under {target}")
